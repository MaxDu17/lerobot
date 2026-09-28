#!/usr/bin/env python

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

import re
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import EaseIn, RecordConfig, record, record_loop
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.utils.keyboard_input import apply_recording_control
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def _ticks(summary: str) -> int:
    """Sample size out of a cadence report — every other number is an average over it."""
    return int(re.search(r"(\d+) ticks", summary).group(1))


def _step_calls(summary: str, step: str) -> int:
    """How many ticks ran *step*, off the loop-body breakdown of a run summary."""
    return int(re.search(rf"\n\s+{step}\s+.*· (\d+) calls", summary).group(1))


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate(cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)

    # A teleop session has no episodes, so there is one cadence block for the whole run,
    # and the steps it names are the ones the loop wraps.
    (summary,) = cadence_log
    assert summary.startswith("Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick):")
    for step in ("observe", "teleop", "send"):
        assert step in summary, step


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

    # Replay has to hit the dataset's frame rate or the trajectory plays back at the
    # wrong speed, so it reports its cadence like every other loop.  Its block is the
    # last one and names its own steps.
    assert cadence_log[-1].startswith("Cadence summary — whole run · target 30 Hz")
    assert "read_frame" in cadence_log[-1]


def test_record_reports_a_cadence_summary_per_episode_and_for_the_run(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "cadence",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    record(cfg)

    assert len(cadence_log) == 3
    per_episode, run = cadence_log[:2], cadence_log[2]
    assert [m.split(":")[0] for m in per_episode] == ["Cadence (episode 0)", "Cadence (episode 1)"]
    assert run.startswith("Cadence summary — whole run, 2 episodes")
    # Windows partition the session, so the episodes account for every tick of the run...
    assert _ticks(run) == sum(_ticks(m) for m in per_episode)
    # ...and every one of those ticks wrote a frame.  The reset phase paces at the same
    # fps but records nothing, so it runs on its own timer rather than diluting the
    # numbers that answer "did I record at `fps`?".
    assert _step_calls(run, "record") == _step_calls(run, "observe") == _ticks(run)


@pytest.mark.parametrize(
    "keys, phases, episodes",
    [
        # 'r' has nothing to re-record yet, so the wait goes on until 'n'
        (["left", "right"], ["wait", "wait", "episode"], 1),
        # 'q' ends the session before anything is recorded
        (["esc"], ["wait"], 0),
    ],
)
def test_record_waits_for_n_before_the_first_episode(tmp_path, keys, phases, episodes):
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "wait",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        wait_for_start=True,
    )
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    seen = []

    def loop_pressing_keys(**kwargs):
        if kwargs.get("dataset") is not None:
            seen.append("episode")
        elif kwargs["control_time_s"] == float("inf"):
            seen.append("wait")
            # one key per wait phase; running out raises instead of waiting forever
            apply_recording_control(keys[len(seen) - 1], events)
        else:
            seen.append("reset")
        return record_loop(**kwargs)

    with (
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(MagicMock(), events)),
        patch("lerobot.scripts.lerobot_record.record_loop", side_effect=loop_pressing_keys),
    ):
        dataset = record(cfg)

    assert seen == phases
    assert dataset.num_episodes == episodes


def test_an_untimed_reset_ends_on_n_only_and_r_discards_without_leaving_it(tmp_path):
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "reset",
            num_episodes=2,
            episode_time_s=0.1,
            reset_time_s=float("inf"),
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    # one key per reset: discard the first demo, then start its re-recording, then the next
    keys = iter(["left", "right", "right"])
    seen = []

    def loop_pressing_keys(**kwargs):
        if kwargs.get("dataset") is not None:
            seen.append("episode")
        else:
            seen.append("reset")
            apply_recording_control(next(keys), events)  # running out raises, not hangs
        return record_loop(**kwargs)

    with (
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(MagicMock(), events)),
        patch("lerobot.scripts.lerobot_record.record_loop", side_effect=loop_pressing_keys),
    ):
        dataset = record(cfg)

    assert seen == ["episode", "reset", "reset", "episode", "reset", "episode"]
    assert dataset.num_episodes == 2


def _record_twice_cfg(tmp_path) -> RecordConfig:
    return RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "ds",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
        sidecar_dir=str(tmp_path / "ds.sidecar"),
    )


@pytest.mark.parametrize("answer", ["y", "n"])
def test_record_asks_before_overwriting_an_existing_dataset(tmp_path, answer):
    record(_record_twice_cfg(tmp_path))
    old_sidecar_marker = tmp_path / "ds.sidecar" / "from_the_old_session"
    old_sidecar_marker.mkdir()

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    with (
        patch("sys.stdin") as stdin,
        patch("builtins.input", return_value=answer) as ask,
        # the real listener would try to put the mocked stdin into cbreak mode
        patch("lerobot.scripts.lerobot_record.init_keyboard_listener", return_value=(MagicMock(), events)),
    ):
        stdin.isatty.return_value = True
        if answer == "n":
            with pytest.raises(SystemExit):
                record(_record_twice_cfg(tmp_path))
        else:
            dataset = record(_record_twice_cfg(tmp_path))

    ask.assert_called_once()
    assert "Do you want to overwrite ds y/n?" in ask.call_args.args[0]
    if answer == "y":
        assert dataset.num_episodes == 1
        assert not old_sidecar_marker.exists()
    else:
        assert (tmp_path / "ds" / "meta" / "info.json").is_file()
        assert old_sidecar_marker.exists()


def test_record_without_a_terminal_still_refuses_an_existing_dataset(tmp_path):
    record(_record_twice_cfg(tmp_path))
    with patch("sys.stdin") as stdin, pytest.raises(FileExistsError):
        stdin.isatty.return_value = False
        record(_record_twice_cfg(tmp_path))


def test_ease_in_blends_from_the_follower_to_the_live_leader():
    clock = [100.0]
    follower = {"a.pos": 0.0, "front": object()}
    with patch("lerobot.scripts.lerobot_record.time.perf_counter", lambda: clock[0]):
        ease = EaseIn(2.0)
        # the first command is where the follower already is; keys it lacks pass through
        assert ease({"a.pos": 50.0, "b": 7}, follower) == {"a.pos": 0.0, "b": 7}
        clock[0] += 1.0  # halfway, where the smoothstep is 0.5, towards a leader that moved
        assert ease({"a.pos": 80.0}, follower)["a.pos"] == pytest.approx(40.0)
        clock[0] += 1.0
        assert ease({"a.pos": 80.0}, follower)["a.pos"] == pytest.approx(80.0)
        assert ease.remaining_s == 0


def test_record_eases_the_follower_onto_the_leader_before_recording(tmp_path):
    cfg = RecordConfig(
        robot=MockRobotConfig(random_values=False, static_values=[0.0, 0.0, 0.0]),
        teleop=MockTeleopConfig(random_values=False, static_values=[90.0, 90.0, 90.0]),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
            single_task="Dummy task",
            root=tmp_path / "ease",
            num_episodes=1,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        play_sounds=False,
        ease_in_s=0.3,
    )
    sent = []

    def make_spied_robot(config):
        robot = make_robot_from_config(config)
        send = robot.send_action
        robot.send_action = lambda action: sent.append(action["motor_1.pos"]) or send(action)
        return robot

    with patch("lerobot.scripts.lerobot_record.make_robot_from_config", side_effect=make_spied_robot):
        dataset = record(cfg)

    assert sent[0] == pytest.approx(0.0, abs=1e-3)
    assert sent == sorted(sent)
    # ~9 ticks cover the 90-unit gap; one tick taking a third of it would be a jump
    assert max(b - a for a, b in zip(sent, sent[1:], strict=False)) < 30
    assert sent[-3:] == [90.0, 90.0, 90.0]
    # the episode records the leader's commands, not the eased ones
    assert dataset[0]["action"].tolist() == [90.0, 90.0, 90.0]


def test_record_loop_without_a_teleoperator_paces_and_terminates():
    # Regression: the no-teleop branch used to `continue` past both the pacing sleep and
    # the `timestamp` update, so a reset phase with no teleop device spun as fast as the
    # CPU allowed and never reached `control_time_s` at all.
    robot = make_robot_from_config(MockRobotConfig())
    robot.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    calls = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal calls
        calls += 1
        assert calls <= 20, "loop is spinning: 20 iterations of a 0.1 s phase at 30 Hz"
        return real_get_observation()

    robot.get_observation = counted_get_observation

    try:
        record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=None,
            control_time_s=0.1,
        )
    finally:
        robot.disconnect()

    # 0.1 s at 30 Hz is 3 ticks; the upper bound is what proves the phase was paced.
    assert 1 <= calls <= 6
