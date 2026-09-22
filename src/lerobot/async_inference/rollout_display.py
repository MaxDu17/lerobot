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

"""Live camera + status window for interactive policy rollouts, built on PyQt5.

Extending it:

* a new status value -> add a field to :class:`RolloutStatus`, populate it in
  ``RobotClient._status()``, and add one entry to :data:`PANEL_ROWS`;
* a row the caller owns -> append to ``window.extra_rows``; no edit here;
* a new camera -> nothing, the window builds one tile per key it is given;
* a new control -> add a button in :meth:`RolloutWindow._build_controls`, which
  routes through the same ``on_command`` callback the keyboard uses.

Threading. Qt owns the main thread and the control loop runs on a worker, which is
the inverse of the usual lerobot arrangement but the only way Qt works: widgets may
only be touched from the thread that created them. The worker therefore never calls
into the window directly -- it emits :class:`DisplayBridge.updated`, and Qt marshals
the payload onto the GUI thread. The frames handed over must not be mutated
afterwards; the client passes freshly captured arrays, which satisfies that.
"""

from __future__ import annotations

import logging
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

PANEL_W = 260


@dataclass
class RolloutStatus:
    """Everything the panel can show. Extend here, then add a row to PANEL_ROWS."""

    state: str = "IDLE"
    running: bool = False
    recording: bool = False
    episodes: int = 0
    frames: int = 0
    queue: int = 0
    chunk: int = 0
    fps: float = 0.0
    target_fps: float = 30.0
    task: str = ""
    message: str = ""
    # Cameras the policy actually consumes, so the tiles can say which is which.
    policy_cameras: list[str] = field(default_factory=list)


# (label, value-formatter). Add a tuple to add a row; nothing else needs to change.
PANEL_ROWS: list[tuple[str, Callable[[RolloutStatus], str]]] = [
    ("Episodes", lambda s: str(s.episodes)),
    ("Frames", lambda s: str(s.frames) if s.recording else "-"),
    ("Queue", lambda s: f"{s.queue}/{s.chunk}" if s.chunk > 0 else str(s.queue)),
    ("Loop Hz", lambda s: f"{s.fps:.1f} / {s.target_fps:.0f}"),
]

# Buttons rendered under the panel, as (label, command). Commands are the same
# strings the keyboard produces, so both paths converge on one handler.
CONTROL_BUTTONS: list[tuple[str, str]] = [
    ("Start  (c)", "start"),
    ("Stop + save  (s)", "stop"),
    ("Discard  (d)", "discard"),
    ("Rest  (r)", "rest"),
    ("Home  (h)", "home"),
]

_KEY_TO_COMMAND = {
    Qt.Key_C: "start",
    Qt.Key_S: "stop",
    Qt.Key_D: "discard",
    Qt.Key_R: "rest",
    Qt.Key_H: "home",
    Qt.Key_Q: "quit",
    Qt.Key_Escape: "quit",
}


class DisplayBridge(QObject):
    """Thread-safe conduit from the control loop to the window.

    A queued signal is the supported way to cross into the GUI thread; calling the
    widget directly from the worker is undefined behaviour that usually presents as a
    crash under load rather than an exception.
    """

    updated = pyqtSignal(object, object)  # frames: dict[str, np.ndarray], RolloutStatus
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
    """One camera feed with a caption."""

    def __init__(self, name: str, width: int, policy_input: bool) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        suffix = "  •  policy input" if policy_input else "  •  recorded only"
        caption = QLabel(name + suffix)
        caption.setStyleSheet(f"color: {_DIM}; font-size: 11px;")

        self.view = QLabel()
        self.view.setFixedWidth(width)
        self.view.setMinimumHeight(int(width * 9 / 16))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(f"background: #000; border: 1px solid {_PANEL};")
        self.view.setText("waiting for frames…")

        layout.addWidget(caption)
        layout.addWidget(self.view)
        self._width = width

    def set_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        self.view.setPixmap(_numpy_to_pixmap(frame, self._width))


class RolloutWindow(QMainWindow):
    """Camera tiles on the left, status panel and controls on the right."""

    def __init__(
        self,
        camera_names: list[str],
        policy_cameras: list[str],
        on_command: Callable[[str], None],
        tile_width: int = 480,
        columns: int = 2,
    ) -> None:
        super().__init__()
        self.setWindowTitle("LeRobot rollout")
        self._on_command = on_command
        self.extra_rows: list[tuple[str, Callable[[RolloutStatus], str]]] = []

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
            tile = CameraTile(name, tile_width, policy_input=name in policy_cameras)
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

        # Rows are rebuilt on demand rather than cached, so extra_rows added after
        # construction still show up.
        self.rows_container = QVBoxLayout()
        self.rows_container.setSpacing(6)
        self._row_labels: dict[str, QLabel] = {}
        layout.addLayout(self.rows_container)

        layout.addWidget(self._rule())

        task_caption = QLabel("TASK")
        task_caption.setStyleSheet(f"color: {_DIM}; font-size: 10px; letter-spacing: 1px;")
        self.task_label = QLabel("")
        self.task_label.setWordWrap(True)
        self.task_label.setStyleSheet(f"color: {_TEXT}; font-size: 12px;")
        layout.addWidget(task_caption)
        layout.addWidget(self.task_label)

        layout.addStretch(1)
        layout.addWidget(self._build_controls())

        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        layout.addWidget(self.message_label)

        return panel

    def _build_controls(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label, command in CONTROL_BUTTONS:
            button = QPushButton(label)
            button.setStyleSheet(
                f"QPushButton {{ background: #32323a; color: {_TEXT}; border: none;"
                f" padding: 6px; border-radius: 4px; font-size: 11px; }}"
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
        # Closing the window ends the session; leaving the arm live with no visible
        # controls is worse than shutting down.
        self._dispatch("quit")
        super().closeEvent(event)

    def on_update(self, frames: dict[str, np.ndarray], status: RolloutStatus) -> None:
        """Slot for DisplayBridge.updated. Runs on the GUI thread."""
        for name, tile in self.tiles.items():
            tile.set_frame(frames.get(name))

        colour = _RUNNING if status.running else _IDLE
        self.state_label.setText(status.state)
        self.state_label.setStyleSheet(f"color: {colour};")
        self.rec_label.setText("● REC" if status.recording else "")
        self.task_label.setText(status.task)
        self.message_label.setText(status.message)
        self.message_label.setStyleSheet(f"color: {colour}; font-size: 11px;")

        for label, formatter in [*PANEL_ROWS, *self.extra_rows]:
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
