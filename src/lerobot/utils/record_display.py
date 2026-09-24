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

"""Live camera + status window for teleoperated data collection, built on PyQt5.

Selected with ``--display_data=true --display_mode=pyqt``. It is the lightweight
alternative to the rerun viewer: one tile per camera, the recording controls as
labelled buttons, and a panel that says whether the session is actually healthy.

This is the recording counterpart of
:mod:`lerobot.async_inference.rollout_display`, which serves policy rollouts. The
two are deliberately separate files rather than one parameterised widget: they show
different things (episodes and stream health here, action-queue depth there) and the
rollout window is the one already trusted on hardware, so it is left untouched. The
Qt chrome they share is small -- :func:`_numpy_to_pixmap` and :class:`CameraTile` --
and worth folding into a common module if a third window ever appears.

Extending it:

* a new status value -> add a field to :class:`RecordStatus`, populate it in
  ``_status()`` in ``lerobot_record.py``, and add one entry to :data:`PANEL_ROWS`;
* a new warning -> emit it from :class:`StreamMonitor`, which owns every check;
* a new camera -> nothing, the window builds one tile per key it is given;
* a new control -> add a button in :meth:`RecordWindow._build_controls`, which routes
  through the same ``on_command`` callback the keyboard uses.

Threading. Qt owns the main thread and the record loop runs on a worker, which is the
inverse of the usual lerobot arrangement but the only way Qt works: widgets may only
be touched from the thread that created them, and on macOS that has to be the main
one. The worker therefore never calls into the window directly -- it emits
:class:`DisplayBridge.updated`, and Qt marshals the payload onto the GUI thread. The
frames handed over must not be mutated afterwards; the loop passes freshly captured
arrays, which satisfies that.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_BG = "#1c1c1e"
_PANEL = "#242428"
_TEXT = "#e1e1e1"
_DIM = "#8e8e93"
_RUNNING = "#5ed46e"
_IDLE = "#f5a623"
_REC = "#ff453a"
_WARN_BG = "#4a2c00"
_WARN_TEXT = "#ffd479"

PANEL_W = 280

# A loop or camera slower than this fraction of its target is called out. Recording
# below the declared fps silently produces a dataset whose timestamps do not match
# its `fps` field, which is the kind of thing you want to find during collection
# rather than during training.
HEALTHY_RATE_FRACTION = 0.9

# A camera handing back a byte-identical frame for this long is treated as stalled.
# Generous on purpose: a genuinely static scene at 30 fps still jitters in sensor
# noise, so identical frames mean the capture pipeline stopped, not that nothing moved.
STALE_FRAME_S = 1.5


@dataclass
class RecordStatus:
    """Everything the panel can show. Extend here, then add a row to PANEL_ROWS."""

    state: str = "IDLE"
    recording: bool = False
    # Index of the episode being recorded, and how many the session was asked for.
    episode: int = 0
    target_episodes: int = 0
    saved_episodes: int = 0
    # Frames buffered for the episode in progress.
    frames: int = 0
    fps: float = 0.0
    target_fps: float = 30.0
    # Measured delivery rate per camera, and the rate each was configured for.
    camera_fps: dict[str, float] = field(default_factory=dict)
    camera_target_fps: dict[str, float] = field(default_factory=dict)
    task: str = ""
    dataset_root: str = ""
    message: str = ""
    warnings: list[str] = field(default_factory=list)


# (label, value-formatter). Add a tuple to add a row; nothing else needs to change.
# Per-camera rates are appended separately, since their labels are only known at runtime.
PANEL_ROWS: list[tuple[str, Callable[[RecordStatus], str]]] = [
    ("Episode", lambda s: f"{s.episode + 1} / {s.target_episodes}" if s.target_episodes else "-"),
    ("Frames", lambda s: str(s.frames) if s.recording else "-"),
    ("Saved", lambda s: str(s.saved_episodes)),
    ("Loop Hz", lambda s: f"{s.fps:.1f} / {s.target_fps:.0f}"),
]

# Buttons rendered under the panel, as (label, command). Commands are the same strings
# the keyboard produces, so both paths converge on one handler.
CONTROL_BUTTONS: list[tuple[str, str]] = [
    ("Keep + next  (n)", "next"),
    ("Re-record  (r)", "rerecord"),
    ("Stop recording  (q)", "quit"),
]

# The arrow/Esc spellings are accepted too, so the window and the TTY listener take the
# same keys. `apply_recording_control` documents the arrow mapping these mirror.
_KEY_TO_COMMAND = {
    Qt.Key_N: "next",
    Qt.Key_Right: "next",
    Qt.Key_R: "rerecord",
    Qt.Key_Left: "rerecord",
    Qt.Key_Q: "quit",
    Qt.Key_Escape: "quit",
}


class StreamMonitor:
    """Measures loop and camera rates on the control-loop thread, and flags problems.

    Kept out of the window because it is plain bookkeeping with no Qt in it, and
    because the checks are the useful part: a window that only mirrors frames tells
    you nothing a camera app would not. Rates are computed over a trailing window
    rather than the session, so a stall shows up while it is happening instead of
    being averaged away.
    """

    def __init__(self, target_fps: float, camera_target_fps: dict[str, float], window_s: float = 2.0):
        self.target_fps = target_fps
        self.camera_target_fps = dict(camera_target_fps)
        self._window_s = window_s
        self._ticks: deque[float] = deque()
        # Per camera: timestamps of frames that differed from their predecessor, plus
        # the last signature seen and when it last changed.
        self._frames: dict[str, deque[float]] = {}
        self._last_signature: dict[str, float] = {}
        self._last_change: dict[str, float] = {}
        self._no_action = False

    @staticmethod
    def _signature(frame: np.ndarray) -> float:
        """Cheap content fingerprint, strided so full-HD frames stay affordable."""
        return float(np.asarray(frame)[::32, ::32].sum())

    def _trim(self, stamps: deque[float], now: float) -> None:
        while stamps and now - stamps[0] > self._window_s:
            stamps.popleft()

    @staticmethod
    def _rate(stamps: deque[float]) -> float:
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 1e-6 else 0.0

    def tick(self, frames: dict[str, np.ndarray] | None = None) -> None:
        """Record one control-loop iteration, and any camera frames it carried."""
        now = time.perf_counter()
        self._ticks.append(now)
        self._trim(self._ticks, now)

        for name, frame in (frames or {}).items():
            if frame is None:
                continue
            stamps = self._frames.setdefault(name, deque())
            self._last_change.setdefault(name, now)
            try:
                signature = self._signature(frame)
            except Exception as e:  # a malformed frame must not stop the loop
                logger.debug("Could not fingerprint frame from %r: %s", name, e)
                continue
            # Only frames that actually changed count toward the rate: a camera that
            # keeps returning its last buffer would otherwise read as perfectly healthy.
            if signature != self._last_signature.get(name):
                self._last_signature[name] = signature
                self._last_change[name] = now
                stamps.append(now)
            self._trim(stamps, now)

    def note_no_action(self, missing: bool) -> None:
        """Flag that the teleoperator produced nothing on the last tick."""
        self._no_action = missing

    @property
    def loop_fps(self) -> float:
        return self._rate(self._ticks)

    @property
    def camera_fps(self) -> dict[str, float]:
        return {name: self._rate(stamps) for name, stamps in self._frames.items()}

    def _settled(self) -> bool:
        """True once enough of the window has elapsed for a rate to mean anything.

        Without this every session opens with a spurious "too slow", because the first
        few ticks span far less than the averaging window.
        """
        return len(self._ticks) > 2 and (self._ticks[-1] - self._ticks[0]) > self._window_s / 2

    def warnings(self) -> list[str]:
        """Human-readable problems with the stream right now, worst first."""
        now = time.perf_counter()
        problems: list[str] = []

        frozen = {
            name: now - last for name, last in self._last_change.items() if now - last > STALE_FRAME_S
        }
        problems += [f"{name}: frozen for {held:.1f}s" for name, held in sorted(frozen.items())]

        loop_fps = self.loop_fps
        loop_slow = self._settled() and loop_fps < self.target_fps * HEALTHY_RATE_FRACTION
        if loop_slow:
            problems.append(f"loop at {loop_fps:.1f} Hz, target {self.target_fps:.0f}")

        for name, measured in sorted(self.camera_fps.items()):
            target = self.camera_target_fps.get(name)
            if target is None or name in frozen or len(self._frames.get(name, ())) <= 2:
                continue
            # A camera cannot outrun the loop that samples it, so it is held to the
            # loop's *measured* rate when that is the lower of the two. Otherwise a
            # slow loop would be reported twice: once as itself, and once per camera.
            ceiling = min(target, loop_fps) if loop_fps > 0 else target
            if measured < ceiling * HEALTHY_RATE_FRACTION:
                problems.append(f"{name}: {measured:.1f} Hz, expected {ceiling:.0f}")

        if self._no_action:
            problems.append("no teleop action - is the leader arm connected?")

        return problems


class DisplayBridge(QObject):
    """Thread-safe conduit from the record loop to the window.

    A queued signal is the supported way to cross into the GUI thread; calling the
    widget directly from the worker is undefined behaviour that usually presents as a
    crash under load rather than an exception.
    """

    updated = pyqtSignal(object, object)  # frames: dict[str, np.ndarray], RecordStatus
    closed = pyqtSignal()


def _numpy_to_pixmap(frame: np.ndarray, width: int) -> QPixmap:
    """RGB uint8 array -> QPixmap scaled to `width`."""
    height, original_width = frame.shape[:2]
    # QImage does not copy, and the array may be reused by the caller, so copy here.
    image = QImage(
        np.ascontiguousarray(frame).data, original_width, height, 3 * original_width, QImage.Format_RGB888
    ).copy()
    return QPixmap.fromImage(image).scaledToWidth(width, Qt.SmoothTransformation)


class CameraTile(QWidget):
    """One camera feed with a caption that doubles as its health readout."""

    def __init__(self, name: str, width: int) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.name = name
        self.caption = QLabel(name)
        self.caption.setStyleSheet(f"color: {_DIM}; font-size: 11px;")

        self.view = QLabel()
        self.view.setFixedWidth(width)
        self.view.setMinimumHeight(int(width * 9 / 16))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(f"background: #000; border: 1px solid {_PANEL};")
        self.view.setText("waiting for frames…")

        layout.addWidget(self.caption)
        layout.addWidget(self.view)
        self._width = width

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        self.view.setPixmap(_numpy_to_pixmap(frame, self._width))

    def set_health(self, measured: float | None, target: float | None) -> None:
        """Put the measured rate in the caption, in red when it is below target."""
        if measured is None:
            self.caption.setText(self.name)
            self.caption.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
            return
        suffix = f"  •  {measured:.1f} Hz"
        healthy = target is None or measured >= target * HEALTHY_RATE_FRACTION
        colour = _DIM if healthy else _REC
        self.caption.setText(self.name + suffix)
        self.caption.setStyleSheet(f"color: {colour}; font-size: 11px;")


class RecordWindow(QMainWindow):
    """Camera tiles on the left, status panel and controls on the right."""

    def __init__(
        self,
        camera_names: list[str],
        on_command: Callable[[str], None],
        tile_width: int = 480,
        columns: int = 2,
    ) -> None:
        super().__init__()
        self.setWindowTitle("LeRobot recording")
        self._on_command = on_command
        self.extra_rows: list[tuple[str, Callable[[RecordStatus], str]]] = []

        central = QWidget()
        central.setStyleSheet(f"background: {_BG};")
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # One tile per camera, wrapping into `columns`, so adding a third camera needs
        # no layout change here.
        grid = QGridLayout()
        grid.setSpacing(10)
        self.tiles: dict[str, CameraTile] = {}
        for i, name in enumerate(camera_names):
            tile = CameraTile(name, tile_width)
            self.tiles[name] = tile
            # Top-aligned, otherwise the grid spreads each tile over the full row
            # height and the caption ends up floating far above its image.
            grid.addWidget(tile, i // columns, i % columns, Qt.AlignTop | Qt.AlignLeft)
        if not camera_names:
            grid.addWidget(QLabel("no cameras configured"), 0, 0)
        # Soak up leftover vertical space below the last row of tiles.
        grid.setRowStretch(grid.rowCount(), 1)

        root.addLayout(grid)
        root.addWidget(self._build_panel())
        self.setCentralWidget(central)

        # Keystrokes must reach keyPressEvent even when a button has been clicked,
        # which would otherwise take focus and swallow them.
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------------------------------------------------------- construction --

    def _build_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(PANEL_W)
        panel.setStyleSheet(f"background: {_PANEL}; border-radius: 6px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self.state_label = QLabel("IDLE")
        state_font = QFont()
        state_font.setPointSize(20)
        state_font.setBold(True)
        self.state_label.setFont(state_font)

        self.rec_label = QLabel("")
        self.rec_label.setStyleSheet(f"color: {_REC}; font-size: 12px; font-weight: bold;")

        header = QHBoxLayout()
        header.addWidget(self.state_label)
        header.addStretch(1)
        header.addWidget(self.rec_label)
        layout.addLayout(header)
        layout.addWidget(self._rule())

        # Rows are rebuilt on demand rather than cached, so per-camera rows and any
        # extra_rows added after construction still show up.
        self.rows_container = QVBoxLayout()
        self.rows_container.setSpacing(6)
        self._row_labels: dict[str, QLabel] = {}
        layout.addLayout(self.rows_container)

        layout.addWidget(self._rule())
        layout.addWidget(self._build_warnings())

        task_caption = QLabel("TASK")
        task_caption.setStyleSheet(f"color: {_DIM}; font-size: 10px; letter-spacing: 1px;")
        self.task_label = QLabel("")
        self.task_label.setWordWrap(True)
        self.task_label.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
        layout.addWidget(task_caption)
        layout.addWidget(self.task_label)

        self.root_label = QLabel("")
        self.root_label.setWordWrap(True)
        self.root_label.setStyleSheet(f"color: {_DIM}; font-size: 10px;")
        layout.addWidget(self.root_label)

        layout.addStretch(1)
        layout.addWidget(self._build_controls())

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        layout.addWidget(self.message_label)

        return panel

    def _build_warnings(self) -> QWidget:
        """Amber banner, hidden entirely while the session is healthy."""
        self.warning_box = QFrame()
        self.warning_box.setStyleSheet(f"background: {_WARN_BG}; border-radius: 4px;")
        layout = QVBoxLayout(self.warning_box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(f"color: {_WARN_TEXT}; font-size: 11px; font-weight: bold;")
        layout.addWidget(self.warning_label)

        self.warning_box.setVisible(False)
        return self.warning_box

    def _build_controls(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label, command in CONTROL_BUTTONS:
            button = QPushButton(label)
            button.setStyleSheet(
                f"QPushButton {{ background: #32323a; color: {_TEXT}; border: none;"
                f" padding: 7px; border-radius: 4px; font-size: 12px; }}"
                f"QPushButton:hover {{ background: #3e3e48; }}"
            )
            # Buttons emit the same commands as the keys, so there is one code path.
            button.clicked.connect(lambda _, c=command: self._dispatch(c))
            # Otherwise the button keeps focus and the next keystroke re-triggers it.
            button.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(button)
        return box

    @staticmethod
    def _rule() -> QFrame:
        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setStyleSheet("color: #3a3a40;")
        return rule

    # -------------------------------------------------------------------- behaviour --

    def _dispatch(self, command: str) -> None:
        try:
            self._on_command(command)
        except Exception:
            logger.exception("Command %r from the window failed", command)

    def keyPressEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        command = _KEY_TO_COMMAND.get(event.key())
        if command is not None:
            self._dispatch(command)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        # Closing the window ends the session. The episode in progress is still saved
        # and the arm still parks, because this takes the same path as 'q'.
        self._dispatch("quit")
        super().closeEvent(event)

    def on_update(self, frames: dict[str, np.ndarray], status: RecordStatus) -> None:
        """Slot for DisplayBridge.updated. Runs on the GUI thread."""
        for name, tile in self.tiles.items():
            tile.set_frame(frames.get(name))
            tile.set_health(status.camera_fps.get(name), status.camera_target_fps.get(name))

        colour = _RUNNING if status.recording else _IDLE
        self.state_label.setText(status.state)
        self.state_label.setStyleSheet(f"color: {colour};")
        self.rec_label.setText("● REC" if status.recording else "")
        self.task_label.setText(status.task)
        self.root_label.setText(status.dataset_root)
        self.message_label.setText(status.message)
        self.message_label.setStyleSheet(f"color: {colour}; font-size: 11px;")

        # Cap the banner: past the first few, the list stops being readable at a glance
        # and the point is to be noticed, not to be complete. The log has them all.
        self.warning_box.setVisible(bool(status.warnings))
        if status.warnings:
            shown = status.warnings[:3]
            extra = len(status.warnings) - len(shown)
            text = "\n".join(f"⚠ {w}" for w in shown)
            self.warning_label.setText(text + (f"\n+{extra} more" if extra else ""))

        camera_rows = [
            (name, (lambda s, n=name: f"{s.camera_fps.get(n, 0.0):.1f} Hz"))
            for name in sorted(status.camera_fps)
        ]
        for label, formatter in [*PANEL_ROWS, *camera_rows, *self.extra_rows]:
            try:
                value = formatter(status)
            except Exception as e:  # a bad row must not take the window down
                logger.debug("Panel row %r failed: %s", label, e)
                value = "?"
            self._set_row(label, value)

    def _set_row(self, label: str, value: str) -> None:
        widget = self._row_labels.get(label)
        if widget is None:
            row = QHBoxLayout()
            caption = QLabel(label)
            caption.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
            widget = QLabel(value)
            widget.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
            row.addWidget(caption)
            row.addStretch(1)
            row.addWidget(widget)
            self.rows_container.addLayout(row)
            self._row_labels[label] = widget
        widget.setText(value)


def make_app() -> QApplication:
    """The process-wide QApplication, created once."""
    return QApplication.instance() or QApplication([])
