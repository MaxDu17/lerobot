# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

Interactive mode (`--interactive=true`) drives the session from the keyboard instead of
running from startup to Ctrl+C, and records each start->stop span as one episode:

    c  start the policy (and begin a new episode)
    s  stop the policy and save the episode
    d  stop the policy and discard the episode
    r  move to the folded rest pose      (idle only)
    h  move to a home pose               (idle only)
    q  save and exit

Stopping does not disconnect the robot: torque stays on and the arm holds position, so
`s` is an episode boundary rather than a shutdown. The checkpoint still loads once at
startup; `c` then re-handshakes against the server's resident-policy cache, which is
near-instant and resets per-episode policy state.

Keys are read straight from the controlling TTY, so the terminal must be focused and
keystrokes are not echoed while the client runs. `--keyboard_backend=auto` switches to
pynput's global hook instead, which also fires when the terminal is in the background;
it needs macOS Accessibility permission and does not work reliably inside an Electron
host such as VS Code's integrated terminal. If the process is killed hard enough to
skip cleanup, `stty sane` restores the terminal.

```shell
python src/lerobot/async_inference/robot_client.py \
    ... \
    --interactive=true \
    --record_root=rollouts/ToyPlate --record_repo_id=me/rollout_ToyPlate \
    --rest_pose="{shoulder_pan.pos: 0.0, shoulder_lift.pos: -98.0, ...}" \
    --home_source=dataset --home_dataset=datasets/ToyPlateLeftOnly
```
"""

import contextlib
import json
import logging
import pickle  # nosec
import random
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

# `lerobot.cameras.realsense` imports pyrealsense2 eagerly whenever the package is merely
# installed, and on some platforms it is installed but not loadable (e.g. a Jetson wheel built
# against a newer glibc). Probe the dependency itself rather than the camera module, so a missing
# realsense only costs us that camera type, while any other import error still surfaces.
try:
    import pyrealsense2  # noqa: F401
except ImportError as e:
    logging.warning("realsense camera type unavailable: pyrealsense2 failed to import (%s)", e)
else:
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
    unitree_g1,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.import_utils import (
    _datasets_available,
    _pandas_available,
    register_third_party_plugins,
    require_package,
)
from lerobot.utils.keyboard_input import TerminalKeyListener, create_key_listener

# Recording and dataset-sampled home poses live behind the `dataset` extra. The client
# itself does not need it: without these the policy still runs, only --record_root and
# --home_source=dataset are unavailable.
if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets import LeRobotDataset, VideoEncodingManager

if TYPE_CHECKING or _pandas_available:
    import pandas as pd

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


def smooth_move_to(
    robot: Robot, current: dict[str, float], target: dict[str, float], duration_s: float, fps: int
):
    """Drive the arm from `current` to `target` by linear joint-space interpolation.

    A local copy of `lerobot.common.control_utils.follower_smooth_move_to`, duplicated
    rather than imported because that module pulls in `lerobot.policies` — a heavy
    dependency the robot client deliberately does not have, since policies run on the
    server. Keep the two in sync if the interpolation ever changes.

    There is no collision checking: this sweeps joints along a straight line, so a move
    that starts mid-task can drag the gripper through the scene. Callers should keep
    `duration_s` slow enough that the operator can cut power.
    """
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)


def load_episode_start_poses(dataset_root: str | Path) -> dict[int, dict[str, float]]:
    """Map episode index -> that episode's first-frame joint positions.

    Reads the dataset's parquet shards directly instead of instantiating a
    `LeRobotDataset`, which would set up video decoding for frames we discard. The
    stored `observation.state` is already in the robot's normalized action units, so
    the returned dict can be handed straight to `robot.send_action`.
    """
    require_package("pandas", "dataset")

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No dataset metadata at {info_path}")

    with open(info_path) as f:
        info = json.load(f)
    names = info["features"]["observation.state"]["names"]

    poses: dict[int, dict[str, float]] = {}
    for shard in sorted((root / "data").rglob("*.parquet")):
        df = pd.read_parquet(shard, columns=["frame_index", "episode_index", "observation.state"])
        first = df[df["frame_index"] == 0]
        for episode, state in zip(first["episode_index"], first["observation.state"], strict=True):
            poses[int(episode)] = {n: float(v) for n, v in zip(names, state, strict=True)}

    if not poses:
        raise ValueError(f"No episode start frames found under {root / 'data'}")

    return poses


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        # --- Interactive control ------------------------------------------------------
        # `shutdown_event` means "tear down the process". `run_event` is a separate,
        # repeatable gate meaning "the policy is driving the arm right now", so a stop
        # can end an episode without disconnecting the robot (which would cut torque
        # and drop the arm) or closing the channel.
        self.run_event = threading.Event()
        if not config.interactive:
            self.run_event.set()  # non-interactive: drive from startup to Ctrl+C

        # Keyboard callbacks fire on the listener thread, which must never touch the
        # serial bus: the control loop is already writing to it. Callbacks only enqueue
        # here, and the control-loop thread performs the work.
        self.commands: Queue[str] = Queue()

        self.listener = None
        self.dataset = None
        self.dataset_features: dict[str, dict] | None = None
        self._home_poses: dict[int, dict[str, float]] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    # ------------------------------------------------------------------ interactive --

    CONTROLS_HELP = "c=start, s=stop+save, d=discard, r=rest, h=home, q=quit"

    def setup_interactive(self) -> None:
        """Open the rollout dataset and attach the keyboard listener."""
        if self.config.record_root:
            self._setup_dataset()

        if self.config.home_source == "dataset":
            # Load now, not on the first 'h': a missing or malformed dataset should fail
            # before the arm is live, not while the operator is waiting for it to move.
            self._home_poses = load_episode_start_poses(self.config.home_dataset)
            self.logger.info(
                f"Loaded {len(self._home_poses)} episode start poses from {self.config.home_dataset}"
            )

        if self.config.interactive:
            self.listener = self._make_listener()
            self.logger.info(
                f"Interactive mode ({self.config.keyboard_backend} keyboard). Controls: {self.CONTROLS_HELP}"
            )
            # Deliberately not "press c" yet: the initial handshake below still has to
            # load the checkpoint, which takes minutes for a VLA. Keys pressed meanwhile
            # queue up and fire once the control loop starts, which looks like a hang.
            self.logger.info("Waiting for the server to load the policy before accepting commands...")

    def _make_listener(self):
        """Start the configured keyboard backend.

        `create_key_listener`'s auto-selection prefers pynput's global hook and only
        falls back to the TTY when macOS reports the process as untrusted. That check
        is best-effort: it asks whether Accessibility is granted, not whether the event
        tap actually delivers anything. Inside an Electron host like VS Code's
        integrated terminal the two disagree -- TCC attributes the tap to a helper
        binary that was never granted, so the trust probe passes, pynput wins the
        selection, and no key ever arrives. The fallback never engages because nothing
        reported failure.

        Reading the controlling TTY has no such ambiguity, which is why it is the
        default here. It also scopes the controls to the focused window: with the
        global hook, a stray 'c' typed into a browser would start the arm.
        """
        if self.config.keyboard_backend == "auto":
            listener = create_key_listener(self._on_key, controls_help=self.CONTROLS_HELP)
            if listener is None:
                raise RuntimeError(
                    "interactive=True but no keyboard backend is available (not a TTY, and "
                    "pynput cannot capture). Run from an interactive terminal, or use "
                    "interactive=False."
                )
            return listener

        if not sys.stdin.isatty():
            # TerminalKeyListener.start() is a silent no-op in this case, which would
            # leave the session with a listener object that never fires.
            raise RuntimeError(
                "keyboard_backend=terminal needs stdin to be an interactive terminal, but it "
                "is redirected or piped. Launch the script directly rather than through a "
                "pipeline, or use --interactive=false."
            )

        listener = TerminalKeyListener(self._on_key)
        listener.start()
        return listener

    def _setup_dataset(self) -> None:
        require_package("datasets", "dataset")

        features = {
            **hw_to_dataset_features(self.robot.observation_features, OBS_STR, self.config.record_video),
            **hw_to_dataset_features(self.robot.action_features, ACTION, self.config.record_video),
        }
        root = Path(self.config.record_root)
        num_cameras = len(getattr(self.robot, "cameras", {}) or {})
        writer_threads = self.config.num_image_writer_threads * num_cameras

        if (root / "meta" / "info.json").exists():
            self.logger.info(f"Appending rollouts to existing dataset at {root}")
            dataset = LeRobotDataset.resume(
                self.config.record_repo_id,
                root=root,
                image_writer_threads=writer_threads,
                streaming_encoding=True,
                encoder_threads=2,
            )
            if dataset.fps != self.config.fps:
                raise ValueError(
                    f"Existing dataset at {root} was recorded at {dataset.fps} fps, "
                    f"but this client runs at {self.config.fps} fps."
                )
            missing = set(features) - set(dataset.features)
            if missing:
                raise ValueError(
                    f"Existing dataset at {root} is missing features {sorted(missing)}. "
                    "Point --record_root at a new directory."
                )
        else:
            self.logger.info(f"Creating rollout dataset at {root}")
            dataset = LeRobotDataset.create(
                self.config.record_repo_id,
                self.config.fps,
                root=root,
                robot_type=self.robot.name,
                features=features,
                use_videos=self.config.record_video,
                image_writer_threads=writer_threads,
                streaming_encoding=True,
                encoder_threads=2,
            )

        self.dataset = dataset
        self.dataset_features = dataset.features

    def _on_key(self, key: str) -> None:
        """Keyboard callback. Runs on the listener thread, so it only enqueues."""
        action = {
            "c": "start",
            "s": "stop",
            "d": "discard",
            "r": "rest",
            "h": "home",
            "q": "quit",
            "esc": "quit",
        }.get(key.lower())

        if action is not None:
            self.commands.put(action)

    def _handle_commands(self) -> None:
        """Drain one queued keyboard command. Runs on the control-loop thread."""
        try:
            command = self.commands.get_nowait()
        except Empty:
            return

        if command == "quit":
            self._stop_policy(save=True)
            self.shutdown_event.set()
        elif command == "start":
            self._start_policy()
        elif command == "stop":
            self._stop_policy(save=True)
        elif command == "discard":
            self._stop_policy(save=False)
        elif command in ("rest", "home"):
            self._move_to_pose(command)

    def _start_policy(self) -> None:
        if self.run_event.is_set():
            self.logger.info("Already running - ignoring 'c'.")
            return

        self._reset_episode_state()

        # Acknowledge the keypress BEFORE the handshake. Both calls below are blocking
        # and run on the control-loop thread, so without this 'c' produces no output at
        # all until they return -- and if the server predates the resident-policy cache
        # it reloads the checkpoint here, which is minutes of total silence that looks
        # exactly like a dead keyboard.
        self.logger.info("Starting policy - handshaking with the server...")

        handshake_start = time.perf_counter()
        try:
            # Ready() flushes the server's observation queue, its predicted-timestep set
            # and its last processed observation. SendPolicyInstructions() then hits the
            # server's resident-policy cache: it skips the checkpoint reload and calls
            # policy.reset(), clearing per-episode state. Both are cheap; together they
            # make a restart behave like a fresh connection.
            self.stub.Ready(services_pb2.Empty())
            self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(self.policy_config)))
        except grpc.RpcError as e:
            self.logger.error(f"Could not start policy - server handshake failed: {e}")
            return

        handshake_s = time.perf_counter() - handshake_start
        if handshake_s > 10:
            # A cache hit is milliseconds. Anything this slow means the server reloaded
            # the checkpoint, i.e. it is running a build without the reuse-on-reconnect
            # support -- every 'c' will cost this much until the server is updated.
            self.logger.warning(
                f"Handshake took {handshake_s:.0f}s - the server reloaded the checkpoint "
                "instead of reusing the resident one. Update and restart the policy server."
            )

        self.run_event.set()
        self.logger.info("Policy RUNNING - press 's' to stop and save.")

    def _stop_policy(self, save: bool = True) -> None:
        if not self.run_event.is_set():
            return

        # Clear first so the control loop stops issuing actions before anything else.
        # The arm holds its last commanded position: the servos stay torqued, and we
        # deliberately do not call robot.disconnect(), which would cut torque and drop it.
        self.run_event.clear()
        self._drain_action_queue()
        self.logger.info("Policy STOPPED - arm holding position.")

        if self.dataset is None:
            return

        if not self.dataset.has_pending_frames():
            self.logger.info("No frames recorded - nothing to save.")
            return

        if save:
            self.logger.info("Saving episode (encoding, this blocks for a moment)...")
            self.dataset.save_episode()
            self.logger.info(f"Saved. Dataset now holds {self.dataset.num_episodes} episodes.")
        else:
            self.dataset.clear_episode_buffer()
            self.logger.info("Discarded episode.")

    def _drain_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()

    def _reset_episode_state(self) -> None:
        """Clear every piece of per-episode state before a restart.

        Any of these left stale makes the next episode misbehave. `latest_action` is the
        worst: `_aggregate_action_queues` discards incoming actions whose timestep is
        `<= latest_action`, so a leftover value silently drops the whole first chunk.
        """
        self._drain_action_queue()
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1
        self.must_go.set()
        self.action_queue_size = []
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

    def _resolve_pose(self, which: str) -> dict[str, float] | None:
        if which == "rest":
            if not self.config.rest_pose:
                self.logger.warning(
                    "No rest pose configured. Capture one with tools/capture_pose.py and pass it "
                    "as --rest_pose, or set DEFAULT_REST_POSE."
                )
                return None
            return dict(self.config.rest_pose)

        source = self.config.home_source
        if source == "none":
            self.logger.info("home_source=none - 'h' does nothing. Set --home_source to enable it.")
            return None
        if source == "manual":
            return dict(self.config.home_pose)

        episode = self.config.home_episode
        if episode is None:
            episode = random.choice(sorted(self._home_poses))
        elif episode not in self._home_poses:
            self.logger.error(f"Episode {episode} not in {self.config.home_dataset}.")
            return None
        self.logger.info(f"Home pose sampled from episode {episode}.")
        return dict(self._home_poses[episode])

    def _move_to_pose(self, which: str) -> None:
        if self.run_event.is_set():
            self.logger.warning(f"Refusing '{which[0]}' while the policy is running - press 's' first.")
            return

        target = self._resolve_pose(which)
        if target is None:
            return

        observation = self.robot.get_observation()
        current = {k: v for k, v in observation.items() if k.endswith(".pos")}

        unknown = set(target) - set(current)
        if unknown:
            self.logger.error(f"Pose names not on this robot: {sorted(unknown)}")
            return

        self.logger.info(f"Moving to {which} pose over {self.config.move_duration_s:.1f}s...")
        smooth_move_to(self.robot, current, target, self.config.move_duration_s, self.config.fps)
        self.logger.info(f"At {which} pose.")

    def _record_frame(self, observation: RawObservation, action: dict[str, Any], task: str) -> None:
        observation_frame = build_dataset_frame(self.dataset_features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset_features, action, prefix=ACTION)
        self.dataset.add_frame({**observation_frame, **action_frame, "task": task})

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            if not self.run_event.is_set():
                # Idle between episodes: nothing is sending observations, so GetActions
                # would just block for obs_queue_timeout and return Empty. Poll the gate.
                time.sleep(0.1)
                continue

            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                if not self.run_event.is_set():
                    # Stopped while this chunk was in flight. Dropping it here keeps
                    # `_stop_policy`'s queue drain from being immediately undone.
                    self.logger.debug("Discarding action chunk that arrived after stop")
                    continue

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")
                # Back off before retrying. Without this a persistent error (server down,
                # tunnel dropped) spins this thread at full speed against the channel.
                time.sleep(self.config.environment_dt)

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        """Capture an observation and ship it to the server."""
        return self.send_captured_observation(self.robot.get_observation(), task, verbose)

    def send_captured_observation(
        self, raw_observation: RawObservation, task: str, verbose: bool = False
    ) -> RawObservation:
        """Ship an already-captured observation to the server.

        Split out from `control_loop_observation` so the interactive loop can capture
        once per tick and use the same frame for both the dataset and the upload,
        rather than reading the bus twice.
        """
        try:
            start_time = time.perf_counter()

            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_prepare_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Preparing observation took {obs_prepare_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations.

        The loop ticks at `fps` for the whole session. `run_event` gates whether a tick
        drives the arm: while it is clear the loop stays alive and responsive so it can
        service keyboard commands (which is also why rest/home moves execute here rather
        than on the listener thread — only this thread may touch the serial bus).
        """
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            # May block for seconds on a rest/home move, but only ever while idle.
            self._handle_commands()

            if self.run_event.is_set():
                _captured_observation, _performed_action = self.control_loop_tick(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action

    def control_loop_tick(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """One policy-driven tick: observe, act, upload, record.

        The observation is captured once at the top and reused for both the dataset
        frame and the server upload. Upstream only captured on upload ticks, which made
        observations arrive at a ragged fraction of the control rate — fine for
        inference, useless as a recording cadence.

        Order matters for the dataset: observe *before* acting, so the stored pair is
        (state at t, action applied at t) rather than an action paired with its own
        result.
        """
        raw_observation: RawObservation = self.robot.get_observation()

        performed_action = self.control_loop_action(verbose) if self.actions_available() else None

        if self._ready_to_send_observation():
            # Copies because send_captured_observation stamps "task" onto the dict and
            # hands it to the serializer; the dataset frame should not carry that key.
            self.send_captured_observation(dict(raw_observation), task, verbose)

        # Record only on ticks that actually moved the arm, so the dataset's frame rate
        # matches the `fps` it declares.
        if performed_action is not None and self.dataset is not None:
            self._record_frame(raw_observation, performed_action, task)

        return raw_observation, performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)
    client.setup_interactive()

    if client.start():
        if cfg.interactive:
            # Reached only after the checkpoint is resident, so 'c' is now genuinely fast.
            client.logger.info(f"Policy loaded. IDLE - controls: {client.CONTROLS_HELP}")

        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            if client.dataset is not None:
                # VideoEncodingManager flushes pending video and calls finalize() on the
                # way out, so do NOT finalize again below. The rescue-save has to sit
                # *inside* it: on the exception path __exit__ cancels pending videos, so
                # an episode saved after it would lose its footage.
                with VideoEncodingManager(client.dataset):
                    try:
                        client.control_loop(task=cfg.task)
                    finally:
                        # Rescue a half-recorded episode from a crash or Ctrl+C. In the
                        # normal 'q' path the stop already saved and cleared the buffer,
                        # so this finds nothing pending.
                        with contextlib.suppress(Exception):
                            if client.dataset.has_pending_frames():
                                client.logger.info("Saving in-progress episode before exit...")
                                client.dataset.save_episode()
            else:
                client.control_loop(task=cfg.task)

        finally:
            if client.listener is not None:
                client.listener.stop()

            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
