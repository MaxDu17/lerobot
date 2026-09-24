#!/usr/bin/env python

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

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pa_parquet = pytest.importorskip("pyarrow.parquet")

from lerobot.utils.record_sidecar import SessionSidecar  # noqa: E402

CAMS = ["front", "overhead"]


def _frame(i: int) -> tuple[dict, dict]:
    frame = {"observation.state": np.full(6, i, np.float32), "action": np.full(6, -i, np.float32)}
    obs = {
        "front": np.full((1080, 1920, 3), i % 255, np.uint8),  # downscaled to the sidecar width
        "overhead": np.full((360, 640, 3), i % 255, np.uint8),
    }
    return frame, obs


def _record(sidecar: SessionSidecar, index: int, n: int, save: bool = True) -> None:
    sidecar.start_episode(index)
    for i in range(n):
        sidecar.add_frame(*_frame(i))
    sidecar.save_episode() if save else sidecar.discard_episode()


def _frame_count(path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return n, size


def test_saved_episode_is_complete(tmp_path):
    sidecar = SessionSidecar(tmp_path, fps=30, cameras=CAMS, task="t", width=320)
    _record(sidecar, 0, 12)
    sidecar.close()

    ep = tmp_path / "episodes" / "ep_000000"
    table = pa_parquet.read_table(ep / "data.parquet").to_pydict()
    assert table["frame_index"] == list(range(12))
    assert table["episode_index"] == [0] * 12
    assert table["observation.state"][5] == [5.0] * 6
    assert table["action"][5] == [-5.0] * 6
    assert _frame_count(ep / "front.mp4") == (12, (320, 180))
    assert _frame_count(ep / "overhead.mp4") == (12, (320, 180))
    info = json.loads((tmp_path / "session.json").read_text())
    assert info["cameras"] == CAMS and info["fps"] == 30


def test_discarded_and_unsaved_episodes_leave_nothing(tmp_path):
    sidecar = SessionSidecar(tmp_path, fps=30, cameras=CAMS, width=320)
    _record(sidecar, 0, 5, save=False)  # re-recorded
    _record(sidecar, 0, 7)  # the retake keeps the index
    sidecar.start_episode(1)  # interrupted: never saved
    sidecar.add_frame(*_frame(0))
    sidecar.close()

    assert sorted(p.name for p in (tmp_path / "episodes").iterdir()) == ["ep_000000"]
    rows = pa_parquet.read_table(tmp_path / "episodes" / "ep_000000" / "data.parquet").num_rows
    assert rows == 7


def test_an_empty_episode_is_not_written(tmp_path):
    sidecar = SessionSidecar(tmp_path, fps=30, cameras=CAMS)
    _record(sidecar, 0, 0)
    sidecar.close()
    assert list((tmp_path / "episodes").iterdir()) == []


def test_dataset_observation_features_drops_left_out_cameras():
    from lerobot.scripts.lerobot_record import dataset_observation_features

    robot = SimpleNamespace(
        cameras={"front": None, "overhead": None},
        observation_features={
            "shoulder_pan.pos": float,
            "front": (360, 640, 3),
            "overhead": (360, 640, 3),
            "overhead_depth": (360, 640, 1),
        },
    )
    assert dataset_observation_features(robot, []) == robot.observation_features
    assert dataset_observation_features(robot, ["front"]) == {
        "shoulder_pan.pos": float,
        "front": (360, 640, 3),
    }
    with pytest.raises(ValueError, match="not robot cameras"):
        dataset_observation_features(robot, ["wrist"])


def test_review_command_substitution():
    pytest.importorskip("PyQt5")
    from lerobot.utils.record_display import ReviewSettings

    settings = ReviewSettings(
        command="review {sidecar_dir} --last {last} --json '{\"keep\": 1}'",
        substitutions={"sidecar_dir": "/data/x.sidecar"},
    )
    assert settings.render(7) == "review /data/x.sidecar --last 7 --json '{\"keep\": 1}'"
