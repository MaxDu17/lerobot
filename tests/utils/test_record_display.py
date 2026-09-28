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

import os
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required (install lerobot[viz])")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before the QApplication exists

from PyQt5.QtCore import QProcess  # noqa: E402

from lerobot.utils.record_display import (  # noqa: E402
    REVIEW_FONT_PT,
    REVIEW_FONT_RANGE,
    REVIEW_FONT_STEP,
    RecordStatus,
    RecordWindow,
    ReviewSettings,
    make_app,
    spoken_summary,
)


@pytest.fixture(scope="module")
def app():
    return make_app()


def _window(command: str = "true") -> RecordWindow:
    return RecordWindow(["front"], on_command=lambda _: None, review=ReviewSettings(command=command))


def test_spoken_summary_is_the_verdict_without_its_markup(app):
    reply = "**Verdict:** clean demos; `ep 3` is the one to check.\n\n**Recent demos**\n- ep 3: regrasp"
    assert spoken_summary(reply) == "Verdict: clean demos; ep 3 is the one to check."


def test_spoken_summary_skips_headings_tables_and_code_and_extends_a_short_opening(app):
    reply = (
        "### 11:02\n\n```\nlog\n```\n\n**Verdict:** ok.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- ep 3: regrasp"
    )
    assert spoken_summary(reply) == "Verdict: ok. ep 3: regrasp"


def test_next_button_says_what_n_does_in_each_phase(app):
    window = _window()
    labels = {}
    for state in ("READY", "RECORDING", "RESET"):
        window.on_update({}, RecordStatus(state=state, recording=state == "RECORDING"))
        labels[state] = window._control_buttons[0][0].text()
    assert labels == {
        "READY": "Start demo collection  (n)",
        "RECORDING": "Save demo  (n)",
        "RESET": "Start demo collection  (n)",
    }


def test_review_text_resizes_within_its_range_and_keeps_paragraph_gaps(app):
    window = _window()  # keep it: dropping it deletes the dock's widgets under it
    dock = window.review_dock
    dock.entries = ["first paragraph\n\nsecond paragraph"]
    dock.larger.click()
    document = dock.view.document()
    assert document.defaultFont().pointSize() == REVIEW_FONT_PT + REVIEW_FONT_STEP
    # Regression: a pixel-sized font gave the markdown importer a point size of -1, and
    # every paragraph gap collapsed to 0.
    assert document.begin().blockFormat().bottomMargin() > 0

    low, high = REVIEW_FONT_RANGE
    for _ in range(high):
        dock.larger.click()
    assert dock._font_pt == high and not dock.larger.isEnabled()
    for _ in range(high):
        dock.smaller.click()
    assert dock._font_pt == low and not dock.smaller.isEnabled()


@pytest.mark.parametrize("ticked", [True, False])
def test_a_finished_review_is_read_aloud_only_when_ticked(app, ticked):
    window = _window(command="printf '**Verdict:** fine.'")
    dock = window.review_dock
    dock._speech_command = ["say"]
    dock.speech = MagicMock()
    dock.speech.state.return_value = QProcess.NotRunning
    dock.read_aloud.setEnabled(True)
    dock.read_aloud.setChecked(ticked)

    dock.start()
    deadline = time.monotonic() + 10
    while dock.process is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert dock.entries and "Verdict:" in dock.entries[0]
    if ticked:
        dock.speech.start.assert_called_once_with("say", ["--", "Verdict: fine."])
    else:
        dock.speech.start.assert_not_called()
