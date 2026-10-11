"""Shared test setup: import path, engine readiness, SRT and CLI helpers."""

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def timestamp(ms: int) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: str, cues) -> str:
    """Write cues as [(start_ms, end_ms, text), ...] and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blocks = [
        f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(cues, 1)
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(blocks) + "\n")
    return path


def offline_engine_ready() -> bool:
    """True when an engine that works without the network can actually render.

    The end-to-end tests need one: edge_tts is skipped in CI, piper needs a voice
    model (scripts/download_voice.py fetches one), pyttsx3 needs SAPI or espeak.
    """
    from tts_engines import find_piper_model

    if find_piper_model():
        return True
    if importlib_available("pyttsx3"):
        try:
            import pyttsx3
            pyttsx3.init()
            return True
        except Exception:
            return False
    return False


def importlib_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def run_cli(args, cwd=None, timeout=1800):
    """Run the CLI in a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "subtitle_to_audio.py"), *args],
        cwd=cwd or ROOT, capture_output=True, text=True, timeout=timeout,
    )


needs_engine = pytest.mark.skipif(
    not offline_engine_ready(),
    reason="no offline TTS engine available (run scripts/download_voice.py for piper)",
)
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture
def srt(tmp_path):
    """Factory: srt(cues, name='x.srt') -> path."""
    def _write(cues, name="x.srt"):
        return write_srt(str(tmp_path / name), cues)
    return _write


@pytest.fixture
def cli(tmp_path):
    """Factory: cli(args) -> CompletedProcess, run from the project root."""
    def _run(args, timeout=1800):
        return run_cli(args, timeout=timeout)
    return _run


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end render tests")
