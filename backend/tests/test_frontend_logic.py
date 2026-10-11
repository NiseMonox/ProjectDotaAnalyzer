"""Regression tests of the DOM-free frontend modules (game.js, coach.js, i18n.js).

The frontend has no build step and no test runner of its own, so its pure
logic (ward states, alive intervals of a teamfight, chat-wheel lines, the
"generate again" state of a coach report, the labels of both languages) is
checked by a plain Node script next to this file; this test runs it, so
``pytest`` covers the frontend logic too. Skipped where Node.js is missing
(set ``NODE`` to its path when it is not on ``PATH``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("frontend_logic.test.js")


def test_frontend_logic():
    node = os.environ.get("NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    result = subprocess.run(
        [node, str(SCRIPT), str(ROOT / "frontend" / "static" / "js")],
        # Node writes UTF-8 to a pipe on every system (the labels are Chinese)
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
