# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove`` (then connect the
Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``).

For a lighter local window, add ``--display_mode=pyqt``: one tile per camera, the recording
controls as buttons, and a panel that flags a stalled camera or a loop running under its target
fps.  Unlike the other two it runs in-process and takes the main thread, so the record loop moves
to a worker; ``--display_fps`` and ``--display_tile_width`` size it.

``--dataset_cameras='[front]'`` keeps the other cameras out of the dataset: they are still
captured and shown, but never become training data. ``--sidecar_dir=<dir>`` writes each saved
episode's joint data plus every camera to a side recording that, unlike the dataset, is readable
while the session is still running (``lerobot.utils.record_sidecar``). ``--review_command`` adds
a "Review last N" button to the pyqt window that runs a command in the background -- typically a
reviewer reading the sidecar -- and shows its stdout as markdown.

``--wait_for_start=true`` holds the first episode until ``n`` is pressed, instead of recording
the moment the robot connects; the arm can be teleoperated, unrecorded, until then.
``--dataset.reset_time_s=inf`` does the same for every reset: the next episode starts on ``n`` only.

Pass ``--rest_pose="{shoulder_pan.pos: 0.0, ...}"`` to park the arm before disconnecting, on every
exit path.  Disconnecting cuts torque, so without it the arm drops when the session ends -- after
the last episode, on ``q``, and on Ctrl+C alike.  Capture a pose with ``tools/capture_pose.py``;
``--rest_on_exit=false`` turns the park off again.

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.rgb_encoder.vcodec=h264 \\
    --dataset.rgb_encoder.preset=fast \\
    --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import json
import logging
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    follower_smooth_move_to,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.datasets.utils import INFO_PATH
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.keyboard_input import apply_recording_control, init_keyboard_listener
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


# Third display_mode, alongside visualization_utils' "rerun" and "foxglove". It is not
# one of theirs: those stream logged observations to an external viewer, while this owns
# a window in-process and needs the main thread for it, so `record` dispatches on it
# before the visualization backends are ever consulted.
PYQT_DISPLAY_MODE = "pyqt"


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun", "foxglove" or "pyqt".
    # "pyqt" opens the lightweight in-process window from `lerobot.utils.record_display`:
    # one tile per camera, the controls as buttons, and a panel that flags a stalled
    # camera or a slow loop. The other two stream to an external viewer as before.
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # Backend for the Right/Left/Esc (and n/r/q) recording controls.
    #   terminal -> read this TTY directly; needs THIS WINDOW FOCUSED, and your keystrokes
    #               stop echoing while recording runs (that is how you know it is live).
    #   auto     -> prefer pynput's global hook, so keys also register when the terminal is
    #               in the background. On macOS it can silently capture nothing when the
    #               process is hosted by an app TCC attributes differently (VS Code's
    #               integrated terminal), which is why it is not the default here.
    # Matches `RobotClientConfig.keyboard_backend`, which defaults the same way.
    keyboard_backend: str = "terminal"
    # Hold the first episode until 'n' (the window's "Start demo collection") instead of
    # recording the moment the robot connects. The arm is teleoperated meanwhile and
    # nothing is recorded; 'r' does nothing yet and 'q' ends the session. Ignored with
    # neither a key listener nor a window, since nothing could then start the recording.
    wait_for_start: bool = False

    # --- pyqt window (display_mode="pyqt") -------------------------------------------
    # Deliberately below the control rate: redrawing at 30 Hz costs more than it shows,
    # and the window is for watching the scene, not for measuring it.
    display_fps: float = 15.0
    display_tile_width: int = 480
    # Camera tiles per row. One stacks them, which keeps the window narrow enough to leave
    # room for the review panel beside it.
    display_columns: int = 1
    # Shell command behind the window's "Review last N" button, run in the background
    # (at low priority, so the loop keeps its rate). Its stdout is shown in the window as
    # markdown. {last}, {sidecar_dir} and {dataset_root} are substituted. Unset: no button.
    review_command: str | None = None
    review_default_last: int = 5

    # --- Cameras and the sidecar ------------------------------------------------------
    # Cameras written into the dataset. Empty (the default) means every robot camera.
    # Any other camera is still captured, shown in the window and written to the sidecar,
    # but is not a dataset feature: a view kept for review, never something to train on.
    # Same idea as `--policy_cameras` in the async client.
    dataset_cameras: list[str] = field(default_factory=list)
    # Directory for the per-episode side recording (`lerobot.utils.record_sidecar`): the
    # saved episodes' joint data plus every camera at `sidecar_width`, readable while the
    # session is still running, which the dataset is not. Unset: no sidecar.
    sidecar_dir: str | None = None
    sidecar_width: int = 640

    # --- Rest pose --------------------------------------------------------------------
    # Disconnecting cuts torque, so without this the arm drops on EVERY exit - finishing
    # the last episode, pressing q, Ctrl+C and crashes alike. Parking first costs a few
    # seconds and saves the arm. Needs rest_pose; without one this warns and skips.
    rest_on_exit: bool = True
    # Joint target in the robot's own action space, e.g. {shoulder_pan.pos: 0.0, ...}.
    # Empty by default because a folded pose is a physical property of your bench, not
    # something that can be guessed: capture yours with tools/capture_pose.py. Same
    # units and same format as `--rest_pose` in the async client.
    rest_pose: dict[str, float] = field(default_factory=dict)
    # Seconds to interpolate the park over. The move is a straight line in joint space
    # with no collision checking, so keep this slow enough to reach the power switch.
    move_duration_s: float = 3.0
    # Seconds to ease the follower onto the leader when the session starts. Without it the
    # first command is the leader's pose, wherever the follower is, and the motors close
    # that gap at full speed. The command blends from the follower's pose to the leader's
    # *live* pose, so the leader may move meanwhile. Nothing is recorded; 'n' and 'r' are
    # ignored until it finishes, 'q' still ends the session. 0 disables it.
    ease_in_s: float = 0.0

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    display_hook: Callable[[RobotObservation, bool], None] | None = None,
    frame_hook: Callable[[dict, RobotObservation], None] | None = None,
    timer: CycleTimer | None = None,
    action_filter: Callable[[RobotAction, RobotObservation], RobotAction] | None = None,
):
    """Drive the robot from the teleoperator at *fps*, optionally recording each frame.

    *display_hook* is called once per tick with the processed observation and whether
    the teleoperator produced nothing, and is how the pyqt window is fed.  It is
    independent of *display_data*, which drives the rerun/foxglove loggers: a caller
    picks one or the other, never both.

    *frame_hook* is called with every frame handed to the dataset, and the processed
    observation it came from (which also carries cameras the dataset leaves out).  It is
    how the sidecar sees exactly the frames the dataset does.

    *timer* lets a caller that runs several phases — :func:`record` records one episode
    per call, with an unrecorded reset phase in between — keep one
    :class:`~lerobot.utils.cycle_timer.CycleTimer` across all of them, so the cadence
    statistics span the whole session and are reported per episode.  Without it each
    call gets a private timer: identical pacing and identical slow-loop warnings, just
    no end-of-run summary, since a single phase has no run to summarise.

    *action_filter* rewrites each command, given the raw observation, just before it is
    sent (:class:`EaseIn` uses it).  A dataset still records the teleop action, so it is
    meant for unrecorded phases.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if timer is None:
        timer = CycleTimer(fps, records_data=dataset is not None)

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        # Checked before `tick()`: this iteration is not a control tick, so it should not
        # be timed as one.
        if events["exit_early"]:
            events["exit_early"] = False
            break

        timer.tick()

        with timer.section("observe"):
            # Get robot observation
            obs = robot.get_observation()

        with timer.section("process_obs"):
            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        with timer.section("teleop"):
            # Get action from teleop
            if isinstance(teleop, Teleoperator):
                act = teleop.get_action()
                if robot.name == "unitree_g1":
                    teleop.send_feedback(obs)

                # Applies a pipeline to the raw teleop action, default is IdentityProcessor
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

            elif isinstance(teleop, list):
                arm_action = teleop_arm.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))
                action_values = act_processed_teleop
                robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
            else:
                robot_action_to_send = None
                no_action_count += 1
                if no_action_count == 1 or no_action_count % 10 == 0:
                    logging.warning(
                        "No teleoperator provided, skipping action generation. "
                        "This is likely to happen when resetting the environment without a teleop device. "
                        "The robot won't be at its rest position at the start of the next episode."
                    )

        if display_hook is not None:
            # Before the no-action guard below, so the window keeps updating through a
            # phase that produces no action instead of freezing on the last good frame.
            with timer.section("telemetry"):
                display_hook(obs_processed, robot_action_to_send is None)

        # Nothing to send and nothing to record, but the phase still has to be paced and
        # still has to end: `continue`ing straight past the tail of the loop body used to
        # spin at full CPU speed on a `control_time_s` that never advanced.
        if robot_action_to_send is None:
            timer.wait()
            timestamp = time.perf_counter() - start_episode_t
            continue

        if action_filter is not None:
            robot_action_to_send = action_filter(robot_action_to_send, obs)

        with timer.section("send"):
            # Send action to robot
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset. action = postprocessor.process(action)
            # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
            _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            with timer.section("record"):
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame)
                if frame_hook is not None:
                    frame_hook(frame, obs_processed)

        if display_data:
            with timer.section("telemetry"):
                log_visualization_data(
                    display_mode,
                    observation=obs_processed,
                    action=action_values,
                    compress_images=display_compressed_images,
                )

        timer.wait()

        timestamp = time.perf_counter() - start_episode_t


class EaseIn:
    """An ``action_filter`` that eases the follower onto the leader over *duration_s*.

    The command is ``(1 - a) * start + a * leader``, where *start* is the follower's pose
    at the first tick and *a* rises from 0 to 1 along a smoothstep, which has zero slope
    at both ends: no jolt as the move begins, and no step when plain teleop takes over.
    Blending towards the leader's live pose, rather than moving to a snapshot of it the
    way :func:`follower_smooth_move_to` does, is what lets the operator hold the leader
    naturally meanwhile. Keys the observation lacks pass through unchanged.

    The clock starts at the first command, and only then, so a phase that is cut short
    and resumed carries on where it left off.
    """

    def __init__(self, duration_s: float):
        self.duration_s = duration_s
        self._start: dict[str, float] | None = None
        self._t0 = 0.0

    @property
    def remaining_s(self) -> float:
        if self._start is None:
            return self.duration_s
        return max(self.duration_s - (time.perf_counter() - self._t0), 0.0)

    def __call__(self, action: RobotAction, observation: RobotObservation) -> RobotAction:
        if self._start is None:
            self._t0 = time.perf_counter()
            self._start = {k: float(observation[k]) for k in action if k in observation}
        s = 1.0 - self.remaining_s / self.duration_s if self.duration_s > 0 else 1.0
        a = s * s * (3.0 - 2.0 * s)
        return {k: (1.0 - a) * self._start[k] + a * v if k in self._start else v for k, v in action.items()}


class _RecordDisplay:
    """Owns the pyqt window's plumbing: the bridge, the health monitor, and the state
    the panel reads.

    Split out of the window itself because everything here runs on the record loop's
    thread and must not touch a widget.  The one method the loop calls per tick is
    :meth:`tick`; everything else is the session announcing what phase it is in.
    """

    # Window commands -> the canonical control names `apply_recording_control` takes,
    # so a button press and the TTY's 'n' land on exactly the same code path.
    _COMMANDS = {"next": "right", "rerecord": "left", "quit": "esc"}

    def __init__(self, cfg: RecordConfig, robot: Robot):
        from lerobot.utils.record_display import DisplayBridge, StreamMonitor

        cameras = getattr(robot, "cameras", {}) or {}
        self.camera_names = list(cameras)
        # A camera with no declared fps is paced by the loop, so that is its target.
        self._camera_targets = {
            name: float(getattr(cam, "fps", None) or cfg.dataset.fps) for name, cam in cameras.items()
        }

        self.cfg = cfg
        self.bridge = DisplayBridge()
        self.monitor = StreamMonitor(float(cfg.dataset.fps), self._camera_targets)
        self._period = 1.0 / max(float(cfg.display_fps), 1e-6)
        self._last_push = 0.0
        self._last_warnings: list[str] = []

        # Filled in by the session once they exist. Until then the window renders an
        # empty panel and its buttons no-op, which is the honest thing to show while
        # the robot is still connecting.
        self.events: dict | None = None
        self.dataset: LeRobotDataset | None = None

        self.state = "STARTING"
        self.message = "connecting..."
        self.episode = 0
        self.saved_episodes = 0
        self.window = None

    def build_window(self):
        """Construct the Qt window. MUST be called on the GUI (main) thread."""
        from lerobot.utils.record_display import RecordWindow, ReviewSettings

        cfg = self.cfg
        review = None
        if cfg.review_command:
            review = ReviewSettings(
                command=cfg.review_command,
                default_last=cfg.review_default_last,
                substitutions={
                    "sidecar_dir": str(Path(cfg.sidecar_dir).resolve()) if cfg.sidecar_dir else "",
                    "dataset_root": str(cfg.dataset.root or ""),
                },
            )
        left_out = [c for c in self.camera_names if cfg.dataset_cameras and c not in cfg.dataset_cameras]
        notes = dict.fromkeys(left_out, "not in dataset")
        self.window = RecordWindow(
            camera_names=self.camera_names,
            on_command=self.dispatch,
            tile_width=cfg.display_tile_width,
            columns=cfg.display_columns,
            camera_notes=notes,
            review=review,
        )
        self.bridge.updated.connect(self.window.on_update)
        return self.window

    def dispatch(self, command: str) -> None:
        """Handle a button or window keystroke. Runs on the GUI thread."""
        if self.events is None:
            logging.info("Ignoring %r: the recording session has not started yet.", command)
            return
        control = self._COMMANDS.get(command)
        if control is None:
            logging.warning("Unknown window command %r", command)
            return
        apply_recording_control(control, self.events)

    def request_stop(self) -> None:
        """Ask the loop to wind up, used when the window closes or Ctrl+C lands."""
        if self.events is not None:
            self.events["stop_recording"] = True
            self.events["exit_early"] = True

    def set_state(self, state: str, message: str = "") -> None:
        """Announce a phase change, and push it immediately rather than at the next tick.

        Phases like SAVING and PARKING take seconds and run with the loop stopped, so a
        throttled update would show them only after they finished, if at all.
        """
        self.state = state
        self.message = message
        self._push(force=True)

    def tick(self, observation: RobotObservation, no_action: bool) -> None:
        """Feed the monitor and push a frame if one is due. Record-loop thread."""
        frames = {name: observation.get(name) for name in self.camera_names}
        self.monitor.tick(frames)
        self.monitor.note_no_action(no_action)
        self._push(frames=frames)

    def _buffered_frames(self) -> int:
        if self.dataset is None:
            return 0
        buffer = getattr(getattr(self.dataset, "writer", None), "episode_buffer", None)
        return buffer.get("size", 0) if buffer else 0

    def _push(self, frames: dict | None = None, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_push) < self._period:
            return
        self._last_push = now

        warnings = self.monitor.warnings()
        # Also log them, once per change: the window is the fast signal, but a session
        # is usually reviewed afterwards from the terminal scrollback.
        if warnings != self._last_warnings:
            for warning in warnings:
                if warning not in self._last_warnings:
                    logging.warning("Recording health: %s", warning)
            self._last_warnings = warnings

        from lerobot.utils.record_display import RecordStatus

        status = RecordStatus(
            state=self.state,
            recording=self.state == "RECORDING",
            episode=self.episode,
            target_episodes=self.cfg.dataset.num_episodes,
            saved_episodes=self.saved_episodes,
            frames=self._buffered_frames(),
            fps=self.monitor.loop_fps,
            target_fps=float(self.cfg.dataset.fps),
            camera_fps=self.monitor.camera_fps,
            camera_target_fps=dict(self._camera_targets),
            task=self.cfg.dataset.single_task or "",
            dataset_root=str(self.cfg.dataset.root or ""),
            message=self.message,
            warnings=warnings,
        )
        # Emitting rather than touching widgets: Qt objects belong to the GUI thread.
        self.bridge.updated.emit(frames or {}, status)


def dataset_observation_features(robot: Robot, dataset_cameras: list[str]) -> dict:
    """``robot.observation_features`` without the cameras left out of *dataset_cameras*.

    Empty *dataset_cameras* keeps every camera.  A camera that is not a feature never
    reaches the dataset: frames are built from the feature spec, so its images are simply
    not picked up, while the robot still captures them for the window and the sidecar.
    """
    features = robot.observation_features
    if not dataset_cameras:
        return features
    cameras = list(getattr(robot, "cameras", {}) or {})
    unknown = sorted(set(dataset_cameras) - set(cameras))
    if unknown:
        raise ValueError(f"dataset_cameras {unknown} are not robot cameras: {cameras}")
    dropped = {c for c in cameras if c not in dataset_cameras}
    # a depth-enabled camera also contributes "<name>_depth"
    return {
        k: v
        for k, v in features.items()
        if k not in dropped and not (k.endswith("_depth") and k[: -len("_depth")] in dropped)
    }


def _park_at_rest(robot: Robot, cfg: RecordConfig) -> None:
    """Drive the arm to ``cfg.rest_pose`` before it is disconnected.

    Disconnecting cuts torque, so every exit path drops the arm without this.  The move
    is straight-line joint interpolation with NO collision checking: it sweeps from
    wherever the arm stopped, which mid-task can drag the gripper across the scene.
    ``move_duration_s`` is the operator's window to cut power, so keep it slow.

    Never raises.  It runs from the session's ``finally``, where an exception would
    replace whatever actually ended the recording with a misleading one, and would skip
    the disconnect and the dataset finalize that follow it.
    """
    if not cfg.rest_on_exit:
        return
    if not cfg.rest_pose:
        logging.warning(
            "No --rest_pose configured - skipping the park, so the arm will drop when torque "
            "cuts. Capture one with tools/capture_pose.py."
        )
        return
    if not robot.is_connected:
        logging.warning("Robot is already disconnected - cannot park at the rest pose.")
        return

    try:
        current = {k: v for k, v in robot.get_observation().items() if k.endswith(".pos")}
        unknown = sorted(set(cfg.rest_pose) - set(current))
        if unknown:
            logging.error("rest_pose names are not on this robot: %s - skipping the park.", unknown)
            return

        log_say("Parking at rest", cfg.play_sounds)
        logging.info("Moving to the rest pose over %.1fs...", cfg.move_duration_s)
        follower_smooth_move_to(
            robot, current, dict(cfg.rest_pose), cfg.move_duration_s, cfg.dataset.fps
        )
        logging.info("At the rest pose.")
    except Exception:
        logging.exception("Failed to park at the rest pose; disconnecting anyway.")


def _confirm_overwrite(cfg: RecordConfig) -> None:
    """Offer to delete the dataset already at ``--dataset.root`` when not resuming.

    ``LeRobotDataset.create`` refuses a root that exists, which used to end the session in
    a FileExistsError. This asks on the terminal instead, before anything connects and
    before the key listener takes the TTY over; 'n' exits without touching anything. The
    sidecar goes too on 'y', since it would otherwise mix the old episodes into the new
    ones under the same numbers, and carry on the old session's review chat.

    Only a folder that is a LeRobot dataset (or empty) is offered: anything else at that
    path is left for the original error rather than deleted on a 'y'. With no terminal
    to ask on, nothing changes either.
    """
    if cfg.resume:
        return
    if cfg.dataset.root is not None:
        root = Path(cfg.dataset.root)
    elif cfg.dataset.no_stamp:
        root = HF_LEROBOT_HOME / cfg.dataset.repo_id
    else:
        return  # the name gets a timestamp at creation, so it cannot collide
    is_dataset = (root / INFO_PATH).is_file()
    is_empty = root.is_dir() and not any(root.iterdir())
    if not (is_dataset or is_empty) or not sys.stdin.isatty():
        return

    episodes = "empty"
    if is_dataset:
        try:
            count = json.loads((root / INFO_PATH).read_text())["total_episodes"]
            episodes = f"{count} episode{'' if count == 1 else 's'}"
        except (OSError, ValueError, KeyError):
            episodes = "episode count unreadable"
    sidecar = Path(cfg.sidecar_dir) if cfg.sidecar_dir else None
    print(f"\n  dataset: {root.resolve()}  ({episodes})")
    if sidecar is not None and sidecar.exists():
        print(f"  sidecar: {sidecar.resolve()}  (deleted with it)")
    while True:
        try:
            answer = input(
                f"Existing dataset found and --resume is not set. Do you want to overwrite {root.name} y/n? "
            )
        except EOFError:
            answer = "n"
        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            break
        if answer in ("n", "no"):
            sys.exit(
                f"Not overwriting {root.name}. Add --resume=true to keep recording into it, "
                "or choose another --dataset.root."
            )

    shutil.rmtree(root)
    if sidecar is not None and sidecar.exists():
        shutil.rmtree(sidecar)
    logging.info("Deleted %s to record it afresh.", root)


def _run_session(
    cfg: RecordConfig,
    robot: Robot,
    teleop: Teleoperator | None,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset_features: dict,
    display_compressed_images: bool,
    display: "_RecordDisplay | None",
) -> LeRobotDataset:
    """Record the whole session and tear it down.

    Factored out of `record` because it runs on the main thread normally, but on a
    worker when the Qt window owns the main thread.
    """
    dataset = None
    listener = None
    sidecar = None
    # Image writers are per dataset camera; a camera kept out of the dataset needs none.
    num_cameras = sum(1 for ft in dataset_features.values() if ft["dtype"] in ("image", "video"))
    # One timer for the whole session, so its statistics describe the recording rather
    # than one episode's slice of it.  The reset phases below deliberately run on their
    # own private timers: they write no frames, so folding their ticks in would dilute
    # every number that answers "did I record at `fps`?".
    timer = CycleTimer(cfg.dataset.fps)

    # With the Qt window the frames reach the display through `display_hook`, so the
    # rerun/foxglove logger stays switched off: it has never heard of "pyqt" and would
    # raise on the mode string.
    log_to_visualizer = cfg.display_data and display is None
    display_hook = display.tick if display is not None else None

    try:
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        if display is not None:
            display.dataset = dataset

        if cfg.sidecar_dir:
            from lerobot.utils.record_sidecar import SessionSidecar

            sidecar = SessionSidecar(
                cfg.sidecar_dir,
                fps=cfg.dataset.fps,
                cameras=list(getattr(robot, "cameras", {}) or {}),
                state_names=dataset.features.get(f"{OBS_STR}.state", {}).get("names"),
                action_names=dataset.features.get(ACTION, {}).get("names"),
                task=cfg.dataset.single_task or "",
                dataset_root=str(dataset.root),
                width=cfg.sidecar_width,
            )
        frame_hook = sidecar.add_frame if sidecar is not None else None

        # Connect the teleoperator before the robot so the robot isn't left idle (and possibly
        # tripping a firmware watchdog) during teleop init. Matches lerobot_teleoperate.py.
        if teleop is not None:
            teleop.connect()
        robot.connect()

        listener, events = init_keyboard_listener(cfg.keyboard_backend)
        if display is not None:
            # The window's buttons now have somewhere to write. Clicks before this point
            # were dropped with a log line -- there was no session yet to steer.
            display.events = events

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        def teleoperate(control_time_s: float, action_filter=None) -> None:
            """Drive the arm without recording: the ease-in, the wait for the first 'n', and the resets."""
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=control_time_s,
                single_task=cfg.dataset.single_task,
                display_data=log_to_visualizer,
                display_mode=cfg.display_mode,
                display_hook=display_hook,
                action_filter=action_filter,
            )

        with VideoEncodingManager(dataset):
            if cfg.ease_in_s > 0:
                logging.info("Easing the follower onto the leader over %.1fs...", cfg.ease_in_s)
                if display is not None:
                    display.set_state("SYNCING", "easing the follower onto the leader")
                ease = EaseIn(cfg.ease_in_s)
                # 'n' and 'r' end a phase, but cutting this one short would bring the jump back
                while ease.remaining_s > 0 and not events["stop_recording"]:
                    teleoperate(ease.remaining_s, action_filter=ease)
                    events["rerecord_episode"] = False

            if cfg.wait_for_start and (listener is not None or display is not None):
                log_say("Ready to record", cfg.play_sounds)
                logging.info("Press 'n' to start recording episode %d.", dataset.num_episodes)
                if display is not None:
                    display.set_state("READY", "set up the scene, then Start demo collection (n)")
                while not events["stop_recording"]:
                    teleoperate(float("inf"))
                    # 'r' ends the phase too, but there is nothing to re-record yet.
                    if not events["rerecord_episode"]:
                        break
                    events["rerecord_episode"] = False

            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                if display is not None:
                    display.episode = recorded_episodes
                    display.set_state("RECORDING", f"episode {episode_index}")
                if sidecar is not None:
                    sidecar.start_episode(episode_index)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=log_to_visualizer,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                    display_hook=display_hook,
                    frame_hook=frame_hook,
                    timer=timer,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    if display is not None:
                        display.set_state("RESET", "reset the scene, then Start demo collection (n)")
                    # 'r' while recording has already asked for a discard. 'r' during the reset
                    # asks for one too, but must not end the reset as it used to: the next demo
                    # starts on 'n' (or the reset timer) only.
                    discard = events["rerecord_episode"]
                    events["rerecord_episode"] = False
                    while True:
                        teleoperate(cfg.dataset.reset_time_s)
                        if not events["rerecord_episode"] or events["stop_recording"]:
                            break
                        discard = True
                        events["rerecord_episode"] = False
                        logging.info("Discarding the last episode; still resetting.")
                        if display is not None:
                            display.set_state(
                                "RESET", "demo discarded; reset the scene, then Start demo collection (n)"
                            )
                    events["rerecord_episode"] = discard

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    if display is not None:
                        display.set_state("DISCARDED", "re-recording the last episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if sidecar is not None:
                        sidecar.discard_episode()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    continue

                if display is not None:
                    display.set_state("SAVING", f"writing episode {episode_index}")
                dataset.save_episode()
                if sidecar is not None:
                    # After the dataset, so a sidecar episode never exists without its
                    # dataset counterpart.
                    sidecar.save_episode()
                recorded_episodes += 1
                if display is not None:
                    display.saved_episodes = recorded_episodes
                # Close the window on the episode just saved.  The digest is emitted on
                # the next episode's first tick, so the reset phase, `save_episode` and
                # the spoken prompts in between are excluded from the cadence instead of
                # being charged to whichever episode they sit next to.  `restart()` then
                # exempts that first tick, whose cameras have been idle for seconds.
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()
    finally:
        # First, and in `finally`: ^C is how most recording sessions end, and the summary
        # is most useful before the video encoding and the hub upload scroll it away.
        timer.log_run_summary()

        log_say("Stop recording", cfg.play_sounds, blocking=True)

        # Before finalize(), which can spend minutes encoding video: get the arm safe
        # first, then do the bookkeeping.
        if display is not None:
            display.set_state("PARKING", "moving to the rest pose")
        _park_at_rest(robot, cfg)

        if sidecar is not None:
            # Drops an episode that was started but never saved (a crash mid-episode).
            sidecar.close()

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if log_to_visualizer:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

        if display is not None:
            display.set_state("DONE", "session finished")
    return dataset


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    _confirm_overwrite(cfg)
    use_window = cfg.display_data and cfg.display_mode == PYQT_DISPLAY_MODE
    if cfg.display_data and not use_window:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=dataset_observation_features(robot, cfg.dataset_cameras)
            ),
            use_videos=cfg.dataset.video,
        ),
    )
    cameras = list(getattr(robot, "cameras", {}) or {})
    left_out = [c for c in cameras if cfg.dataset_cameras and c not in cfg.dataset_cameras]
    if left_out:
        logging.info(f"Cameras kept out of the dataset (window and sidecar only): {left_out}")

    display = None
    if use_window:
        require_package("PyQt5", "viz")
        display = _RecordDisplay(cfg, robot)

    def session() -> LeRobotDataset:
        return _run_session(
            cfg,
            robot,
            teleop,
            teleop_action_processor,
            robot_action_processor,
            robot_observation_processor,
            dataset_features,
            display_compressed_images,
            display,
        )

    if display is None:
        return session()

    # With a window, Qt takes the main thread: widgets may only be touched from the
    # thread that created them, and on macOS that has to be the main one. The record
    # loop moves to a worker and reaches the GUI only through the bridge's signals.
    from PyQt5.QtCore import QTimer

    from lerobot.utils.record_display import make_app

    app = make_app()
    window = display.build_window()
    display.bridge.closed.connect(app.quit)
    window.show()

    # Ctrl+C needs help here, for two reasons. Qt's event loop blocks inside C, where
    # Python never gets to run its signal handlers, so the no-op timer below surfaces
    # often enough for one to fire. And the default handler's KeyboardInterrupt would
    # then be raised inside a Qt slot, which PyQt5 turns into an abort() -- killing the
    # process with the arm unparked and the dataset unfinalized. So Ctrl+C asks the
    # session to wind up instead, which is the same path the window's Stop button takes.
    def on_sigint(signum, frame):
        logging.warning("Ctrl+C - winding up the session; press again to force quit.")
        signal.signal(signal.SIGINT, previous_sigint)  # a second press is not ours
        display.request_stop()

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_sigint)

    outcome: dict = {}
    finished = threading.Event()

    def worker() -> None:
        try:
            outcome["dataset"] = session()
        except BaseException as e:  # carried across the thread boundary and re-raised
            outcome["error"] = e
        finally:
            # Released from here rather than from inside the session, so a failure in
            # the session's own teardown cannot leave the window hanging.
            outcome.setdefault("dataset", None)
            finished.set()
            display.bridge.closed.emit()

    # One timer, two jobs. It hands control back to Python often enough for the SIGINT
    # handler above to run. And it is the race-free half of noticing that the session
    # is over: a session that fails early -- a bad dataset path, a port that will not
    # open -- can emit `closed` before `exec_()` is even entered, and that signal would
    # simply be missed, leaving a dead window on screen with no way out.
    def poll() -> None:
        if finished.is_set():
            app.quit()

    poll_timer = QTimer()
    poll_timer.timeout.connect(poll)
    poll_timer.start(100)

    thread = threading.Thread(target=worker, name="lerobot-record-session")
    thread.start()
    try:
        app.exec_()
    finally:
        # Closing the window (or Ctrl+C) asks the loop to wind up; make sure it really
        # did before returning, so the arm is parked and the dataset is finalized.
        display.request_stop()
        thread.join()
        signal.signal(signal.SIGINT, previous_sigint)

    if "error" in outcome:
        raise outcome["error"]
    return outcome["dataset"]


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
