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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig

from .constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})
    client_device: str = field(
        default="cpu",
        metadata={
            "help": "Device to move actions to after receiving from server (e.g., for downstream planners)"
        },
    )

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    # --- Interactive keyboard control -------------------------------------------------
    # Opt-in: with `interactive=False` the client runs the policy from startup to Ctrl+C,
    # which is the historical behavior.
    interactive: bool = field(
        default=False,
        metadata={"help": "Drive start/stop/rest/home from the keyboard (c/s/r/h/q)"},
    )

    # "terminal" reads the controlling TTY directly and requires the window to be
    # focused. "auto" prefers pynput's global hook, which also fires when the terminal
    # is in the background -- convenient, but it means a stray 'c' in any application
    # starts the arm, and on macOS it can silently capture nothing when the process is
    # hosted by an app TCC attributes differently (VS Code's integrated terminal).
    keyboard_backend: str = field(
        default="terminal",
        metadata={"help": "Keyboard backend: terminal (focused window only) | auto"},
    )

    # --- Policy inputs ----------------------------------------------------------------
    # Which of the robot's cameras the policy actually consumes. Empty means all of
    # them, which is the historical behaviour.
    #
    # This matters because the server indexes `policy_image_features[key]` directly for
    # every image the client sends, so a camera the checkpoint was not trained on is a
    # KeyError mid-rollout, not a silently ignored extra. Cameras left out here are
    # still captured, recorded and displayed - they just never reach the policy, which
    # also keeps them off the wire (observations are raw uint8, so each 640x360 stream
    # is another 0.66 MB per upload, and the upload blocks the control loop).
    policy_cameras: list[str] = field(
        default_factory=list,
        metadata={"help": "Cameras the policy consumes (default: all of the robot's cameras)"},
    )

    # --- Live display -----------------------------------------------------------------
    # An OpenCV window with the camera feed and rollout status. It is also a second
    # key source, so the controls work whether the window or the terminal has focus.
    display: bool = field(default=False, metadata={"help": "Show the camera + status window"})
    display_cameras: list[str] = field(
        default_factory=list,
        metadata={"help": "Cameras to show, in order (default: all of the robot's cameras)"},
    )
    display_fps: float = field(
        default=15.0,
        metadata={"help": "Window refresh rate. Below the control rate on purpose - drawing costs time"},
    )
    display_tile_width: int = field(
        default=480, metadata={"help": "Width in pixels of each camera tile in the window"}
    )

    # --- Episodic recording -----------------------------------------------------------
    # Each start->stop span is one episode. Unset `record_root` to run without recording.
    record_root: str | None = field(
        default=None, metadata={"help": "Directory to write the rollout dataset to (unset = no recording)"}
    )
    record_repo_id: str = field(
        default="", metadata={"help": "repo_id for the rollout dataset (required when record_root is set)"}
    )
    record_video: bool = field(
        default=True, metadata={"help": "Encode camera streams as video rather than individual images"}
    )
    num_image_writer_threads: int = field(
        default=4, metadata={"help": "Image-writer threads per camera while recording"}
    )

    # --- Rest / home poses ------------------------------------------------------------
    # Joint targets in the robot's own action space, e.g. {shoulder_pan.pos: 0.0, ...}.
    # Body joints are normalized to [-100, 100] and the gripper to [0, 100], the same
    # units a recorded dataset's `observation.state` uses.
    rest_pose: dict[str, float] = field(
        default_factory=dict,
        metadata={"help": "Folded rest pose for the 'r' key. Capture one with tools/capture_pose.py"},
    )
    home_source: str = field(
        default="none",
        metadata={"help": "Where the 'h' pose comes from: none | manual | dataset"},
    )
    home_pose: dict[str, float] = field(
        default_factory=dict, metadata={"help": "Explicit home pose, used when home_source=manual"}
    )
    home_dataset: str | None = field(
        default=None,
        metadata={"help": "Dataset root to sample episode start poses from (home_source=dataset)"},
    )
    home_episode: int | None = field(
        default=None,
        metadata={"help": "Pin the sampled home pose to this episode index (default: sample at random)"},
    )
    move_duration_s: float = field(
        default=3.0, metadata={"help": "Seconds to interpolate a rest/home move over"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if not self.client_device:
            raise ValueError("client_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

        if self.keyboard_backend not in ("terminal", "auto"):
            raise ValueError(f"keyboard_backend must be terminal or auto, got {self.keyboard_backend!r}")

        if self.home_source not in ("none", "manual", "dataset"):
            raise ValueError(f"home_source must be one of none/manual/dataset, got {self.home_source!r}")

        # Fail at parse time rather than on the first 'h' press, when the arm is live.
        if self.home_source == "manual" and not self.home_pose:
            raise ValueError("home_source=manual requires --home_pose")

        if self.home_source == "dataset" and not self.home_dataset:
            raise ValueError("home_source=dataset requires --home_dataset")

        if self.record_root and not self.record_repo_id:
            raise ValueError("record_root requires --record_repo_id")

        if self.move_duration_s <= 0:
            raise ValueError(f"move_duration_s must be positive, got {self.move_duration_s}")

        if self.display_fps <= 0:
            raise ValueError(f"display_fps must be positive, got {self.display_fps}")

        if self.display_tile_width <= 0:
            raise ValueError(f"display_tile_width must be positive, got {self.display_tile_width}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "client_device": self.client_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
        }
