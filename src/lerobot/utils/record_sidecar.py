# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-episode side recording kept next to a LeRobot dataset, never inside it.

Selected with ``lerobot-record --sidecar_dir=...``. It exists for two reasons:

* A dataset being recorded cannot be read until ``finalize()``: its parquet files have
  no footer yet and its episode metadata is buffered in memory. So nothing can look at
  a session's demos while the session is still running.
* A camera left out of the dataset (``--dataset_cameras``) would otherwise not be kept
  at all.

The sidecar covers both. For every episode the dataset *saves*, it writes and closes::

    <dir>/session.json                        fps, task, joint names, cameras
    <dir>/episodes/ep_000042/data.parquet     timestamp, frame_index, episode_index,
                                              observation.state, action -- the very frames
                                              the dataset was given
    <dir>/episodes/ep_000042/<camera>.mp4     every robot camera, resized to ``width``

Episode numbers match the dataset. Each episode is assembled in a hidden temporary
folder and renamed into place once it is complete, so a reader sees whole episodes or
nothing. A re-recorded episode is deleted.

It is deliberately not a training format: there is no ``meta/info.json``, so nothing
will mistake it for a LeRobotDataset.

Threading. :meth:`SessionSidecar.add_frame` runs on the record loop and only enqueues;
one writer thread resizes and encodes, in order. A full-HD camera therefore costs the
loop a queue put, not a resize. The frames enqueued must not be mutated afterwards,
which holds for the freshly captured arrays the loop passes.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "lerobot-record-sidecar/1"
EPISODE_DIR = "ep_{index:06d}"


class SessionSidecar:
    def __init__(
        self,
        root: str | Path,
        fps: int,
        cameras: list[str],
        state_names: list[str] | None = None,
        action_names: list[str] | None = None,
        task: str = "",
        dataset_root: str = "",
        width: int = 640,
    ) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.cameras = list(cameras)
        self.width = width
        # Left over from a session that crashed mid-episode: never saved, so never valid.
        for stale in self.episodes_dir.glob(".ep_*"):
            shutil.rmtree(stale, ignore_errors=True)
        info = dict(
            format=FORMAT, fps=fps, task=task, dataset_root=dataset_root, cameras=self.cameras,
            width=width, state_names=state_names, action_names=action_names,
        )
        (self.root / "session.json").write_text(json.dumps(info, indent=2) + "\n")

        self._queue: queue.Queue = queue.Queue()
        self._frames_in_episode = 0
        self._warned_backlog = False
        self._thread = threading.Thread(target=self._run, name="record-sidecar", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------- record-loop side --

    def start_episode(self, index: int) -> None:
        self._frames_in_episode = 0
        self._queue.put(("start", index))

    def add_frame(self, frame: dict, observation: dict) -> None:
        """Frame-hook signature: the dataset frame, plus the observation it came from."""
        images = {cam: observation.get(cam) for cam in self.cameras}
        state, action = np.asarray(frame["observation.state"]), np.asarray(frame["action"])
        self._queue.put(("frame", (self._frames_in_episode, state, action, images)))
        self._frames_in_episode += 1
        # The writer normally keeps up easily; a backlog means it cannot, and memory grows.
        backlog = self._queue.qsize()
        if backlog > 4 * self.fps and not self._warned_backlog:
            logger.warning("Sidecar writer is %d frames behind the record loop.", backlog)
            self._warned_backlog = True

    def save_episode(self) -> None:
        self._queue.put(("save", None))

    def discard_episode(self) -> None:
        self._queue.put(("discard", None))

    def close(self) -> None:
        """Finish everything queued; an episode started but never saved is dropped."""
        self._queue.put(("discard", None))
        self._queue.put(("stop", None))
        self._thread.join()

    # ---------------------------------------------------------------- writer side --

    def _run(self) -> None:
        episode = None
        while True:
            op, arg = self._queue.get()
            try:
                if op == "start":
                    if episode is not None:
                        episode.discard()
                    episode = _EpisodeWriter(self.episodes_dir, arg, self.fps, self.width)
                elif op == "frame" and episode is not None:
                    episode.add(*arg)
                elif op == "save" and episode is not None:
                    episode.save()
                    episode = None
                elif op == "discard" and episode is not None:
                    episode.discard()
                    episode = None
                elif op == "stop":
                    return
            except Exception:
                # A broken side recording must never take the session down with it.
                logger.exception("Sidecar writer failed on %r; dropping this episode.", op)
                if episode is not None:
                    episode.discard()
                    episode = None


class _EpisodeWriter:
    """One episode, written into ``.ep_<index>.tmp`` and renamed on save."""

    def __init__(self, episodes_dir: Path, index: int, fps: int, width: int) -> None:
        self.index = index
        self.fps = fps
        self.width = width
        self.final = episodes_dir / EPISODE_DIR.format(index=index)
        self.tmp = episodes_dir / f".{self.final.name}.tmp"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        self.rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.videos: dict = {}

    def add(self, frame_index: int, state: np.ndarray, action: np.ndarray, images: dict) -> None:
        import cv2

        self.rows.append((frame_index, state, action))
        for cam, img in images.items():
            if img is None:
                continue
            img = np.asarray(img)
            if img.ndim != 3 or img.shape[2] != 3:  # e.g. depth; not reviewed
                continue
            h, w = img.shape[:2]
            size = (self.width, 2 * round(self.width * h / w / 2)) if w > self.width else (w, h)
            writer = self.videos.get(cam)
            if writer is None:
                path = str(self.tmp / f"{cam}.mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
                self.videos[cam] = writer
            if (w, h) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _release(self) -> None:
        for writer in self.videos.values():
            writer.release()
        self.videos = {}

    def save(self) -> None:
        self._release()
        if not self.rows:
            self.discard()
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        idx = np.array([r[0] for r in self.rows], dtype=np.int64)
        table = pa.table({
            "timestamp": (idx / self.fps).astype(np.float32),
            "frame_index": idx,
            "episode_index": np.full(len(idx), self.index, dtype=np.int64),
            "observation.state": [r[1].astype(np.float32).tolist() for r in self.rows],
            "action": [r[2].astype(np.float32).tolist() for r in self.rows],
        })
        pq.write_table(table, self.tmp / "data.parquet")
        # A re-recorded index replaces what was there (only possible across sessions).
        shutil.rmtree(self.final, ignore_errors=True)
        self.tmp.rename(self.final)

    def discard(self) -> None:
        self._release()
        shutil.rmtree(self.tmp, ignore_errors=True)
