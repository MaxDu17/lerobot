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

"""Live camera + status window for interactive policy rollouts.

Kept deliberately small and declarative so it can grow:

* to show a new value, add a field to :class:`RolloutStatus` and one entry to
  :data:`PANEL_ROWS` (or append to ``display.extra_rows`` from the caller, which
  needs no edit to this file);
* to draw on the camera image itself, append to ``display.overlays``.

Everything renders with OpenCV drawing primitives into a single canvas, so there is
no GUI toolkit or event loop to reason about beyond ``cv2.waitKey``.

Threading: ``cv2.imshow`` must be called from the main thread on macOS, which is
where ``RobotClient.control_loop`` already runs. Do not move rendering onto the
action-receiver thread.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np

logger = logging.getLogger(__name__)

# BGR, because the canvas is handed straight to cv2.imshow.
_BG = (28, 28, 30)
_TEXT = (225, 225, 225)
_DIM = (145, 145, 145)
_RUNNING = (120, 215, 110)
_IDLE = (70, 180, 245)
_REC = (70, 70, 240)
_RULE = (60, 60, 64)

PANEL_W = 300
FOOTER_H = 34
_PAD = 16


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


# (label, value-formatter). Add a tuple to add a row; nothing else needs to change.
PANEL_ROWS: list[tuple[str, Callable[[RolloutStatus], str]]] = [
    ("Episodes", lambda s: str(s.episodes)),
    ("Frames", lambda s: str(s.frames) if s.recording else "-"),
    ("Queue", lambda s: f"{s.queue}/{s.chunk}" if s.chunk > 0 else str(s.queue)),
    ("Loop Hz", lambda s: f"{s.fps:.1f} / {s.target_fps:.0f}"),
]


def _put(canvas, text: str, org: tuple[int, int], scale: float, color, thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _text_w(text: str, scale: float) -> int:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]


def _wrap(text: str, max_px: int, scale: float, max_lines: int) -> list[str]:
    """Greedy word wrap to a pixel width, ellipsising anything past `max_lines`.

    Measured rather than counted in characters, so the panel stays correct if PANEL_W
    or the font scale changes.
    """
    if not text:
        return []

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if _text_w(candidate, scale) <= max_px:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            current = ""
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    if lines and sum(len(line.split()) for line in lines) < len(words):
        # Trim the tail until the ellipsis itself fits.
        last = lines[-1]
        while last and _text_w(last + "...", scale) > max_px:
            last = last[:-1]
        lines[-1] = last + "..."
    return lines


class RolloutDisplay:
    """A camera feed with a status panel, refreshed at a fraction of the control rate.

    The control loop runs at 30 Hz but the window does not need to: rendering and
    ``cv2.waitKey`` cost a millisecond or two each, which is real budget inside a
    33 ms tick. :meth:`due` lets the caller skip most ticks.
    """

    def __init__(
        self,
        window_name: str = "LeRobot rollout",
        refresh_hz: float = 15.0,
        scale: float = 1.0,
        controls_help: str = "",
    ) -> None:
        self.window_name = window_name
        self.scale = scale
        self.controls_help = controls_help
        self._period = 1.0 / refresh_hz if refresh_hz > 0 else 0.0
        self._last_draw = 0.0
        self._closed = False

        # Extension seams: callers can add rows or draw on the frame without editing
        # this module. `overlays` receive the (already BGR, already scaled) frame.
        self.extra_rows: list[tuple[str, Callable[[RolloutStatus], str]]] = []
        self.overlays: list[Callable[[np.ndarray, RolloutStatus], None]] = []

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    @property
    def closed(self) -> bool:
        """True once the user has closed the window (or it failed to open)."""
        if self._closed:
            return True
        try:
            # < 1 means destroyed; a backend without window properties raises instead.
            self._closed = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            self._closed = True
        return self._closed

    def due(self) -> bool:
        """Whether enough time has passed to be worth redrawing."""
        return (time.perf_counter() - self._last_draw) >= self._period

    def update(self, frame: np.ndarray | None, status: RolloutStatus) -> None:
        """Draw one frame. `frame` is an RGB camera image, or None for a placeholder."""
        if self.closed:
            return

        self._last_draw = time.perf_counter()
        canvas = self._compose(frame, status)
        try:
            cv2.imshow(self.window_name, canvas)
        except cv2.error as e:
            logger.warning("Display update failed, disabling window: %s", e)
            self._closed = True

    def poll_key(self) -> str | None:
        """Pump the GUI event loop and return a pressed key name, if any.

        Must be called regularly or the window will not repaint. Returns the same
        canonical names the keyboard listeners emit, so the caller can feed it
        straight into the same handler.
        """
        if self._closed:
            return None
        try:
            code = cv2.waitKey(1)
        except cv2.error:
            self._closed = True
            return None
        if code == -1:
            return None
        if code == 27:
            return "esc"
        char = chr(code & 0xFF)
        return char if char.isprintable() else None

    def close(self) -> None:
        self._closed = True
        try:
            cv2.destroyWindow(self.window_name)
            # destroyWindow only queues the teardown; waitKey lets the backend run it.
            cv2.waitKey(1)
        except cv2.error:
            pass

    # ------------------------------------------------------------------ rendering --

    def _compose(self, frame: np.ndarray | None, status: RolloutStatus) -> np.ndarray:
        if frame is None:
            view = np.full((360, 640, 3), 40, dtype=np.uint8)
            _put(view, "no camera frame", (200, 185), 0.6, _DIM)
        else:
            # Cameras hand back RGB; cv2.imshow wants BGR.
            view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.scale != 1.0:
                view = cv2.resize(view, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        for overlay in self.overlays:
            overlay(view, status)

        h, w = view.shape[:2]
        canvas = np.full((h + FOOTER_H, w + PANEL_W, 3), _BG, dtype=np.uint8)
        canvas[:h, :w] = view

        self._draw_panel(canvas, status, x0=w, height=h)
        self._draw_footer(canvas, y0=h, width=w + PANEL_W)
        return canvas

    def _draw_panel(self, canvas: np.ndarray, status: RolloutStatus, x0: int, height: int) -> None:
        x = x0 + _PAD
        y = _PAD + 14

        colour = _RUNNING if status.running else _IDLE
        _put(canvas, status.state, (x, y + 10), 0.95, colour, 2)

        if status.recording:
            # Filled dot reads as "armed" faster than the word REC alone.
            cv2.circle(canvas, (x0 + PANEL_W - _PAD - 46, y + 3), 6, _REC, -1)
            _put(canvas, "REC", (x0 + PANEL_W - _PAD - 34, y + 9), 0.55, _REC, 1)

        y += 34
        cv2.line(canvas, (x, y), (x0 + PANEL_W - _PAD, y), _RULE, 1)
        y += 26

        for label, formatter in [*PANEL_ROWS, *self.extra_rows]:
            try:
                value = formatter(status)
            except Exception as e:  # a bad row must not take the window down
                logger.debug("Panel row %r failed: %s", label, e)
                value = "?"
            _put(canvas, label, (x, y), 0.48, _DIM)
            _put(canvas, value, (x + 108, y), 0.52, _TEXT)
            y += 26

        y += 10
        cv2.line(canvas, (x, y), (x0 + PANEL_W - _PAD, y), _RULE, 1)
        y += 24

        inner_w = PANEL_W - 2 * _PAD

        _put(canvas, "TASK", (x, y), 0.42, _DIM)
        y += 20
        for line in _wrap(status.task, inner_w, 0.46, 3):
            _put(canvas, line, (x, y), 0.46, _TEXT)
            y += 18

        if status.message:
            # Pinned to the bottom so it does not shift as the task wraps.
            my = height - _PAD - 4
            for line in reversed(_wrap(status.message, inner_w, 0.46, 2)):
                _put(canvas, line, (x, my), 0.46, colour)
                my -= 18

    def _draw_footer(self, canvas: np.ndarray, y0: int, width: int) -> None:
        cv2.line(canvas, (0, y0), (width, y0), _RULE, 1)
        if self.controls_help:
            _put(canvas, self.controls_help, (_PAD, y0 + 22), 0.48, _DIM)
