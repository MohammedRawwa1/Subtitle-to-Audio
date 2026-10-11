"""Tests for the objective voice measurements, and for the benchmark script itself.

These are the numbers that stand in for "does it sound right": pitch register, pitch
range (expressiveness), pacing over voiced audio only, and distortion.
"""

import json
import os
import subprocess
import sys

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import tts_engines as te
import voice_metrics as vm
from tests.conftest import ROOT, needs_engine, needs_ffmpeg, run_cli, write_srt


def tone(freq, ms):
    return Sine(freq).to_audio_segment(duration=ms)


# ------------------------------------------------------------------ pitch


def test_pitch_tracker_finds_a_pure_tone():
    metrics = vm.frame_pitch(tone(150, 1000))
    assert abs(metrics["median_f0_hz"] - 150) <= 4
    assert metrics["voiced_pct"] > 80
    assert metrics["semitones"] is not None and metrics["semitones"] < 1.0


def test_pitch_tracker_sees_a_wide_range():
    """A 110 -> 220 Hz sweep is one octave, so ~12 semitones of range."""
    metrics = vm.frame_pitch(tone(110, 600) + tone(220, 600))
    assert metrics["semitones"] >= 9.0
    assert 120 <= metrics["median_f0_hz"] <= 220


def test_silence_is_not_mistaken_for_speech():
    metrics = vm.frame_pitch(AudioSegment.silent(duration=3000) + tone(200, 1000))
    assert metrics["silent_pct"] > 60
    assert abs(metrics["median_f0_hz"] - 200) <= 6


def test_unvoiced_noise_reports_no_pitch():
    """Broadband noise has no pitch; the tracker must say so rather than invent one."""
    import random
    random.seed(1)
    noise = AudioSegment(
        data=bytes(random.randint(0, 255) for _ in range(44_100 * 2)),
        sample_width=2, frame_rate=44_100, channels=1,
    )
    metrics = vm.frame_pitch(noise)
    assert metrics["voiced_pct"] < 20 or metrics["semitones"] is None


# --------------------------------------------------------------- distortion


def test_clipping_needs_a_run_not_a_peak():
    """A clean tone touching full scale is not distorted; driving it is."""
    assert vm.clipped_percent(tone(300, 500)) == 0.0
    assert vm.clipped_percent(tone(300, 500).apply_gain(6)) > 10.0
    assert vm.clipped_percent(AudioSegment.silent(duration=200)) == 0.0


def test_rate_counts_only_the_voiced_part():
    """Two words a second is 120 wpm; leading silence must not change that."""
    speech = tone(200, 2000)
    quiet = AudioSegment.silent(duration=4000) + speech
    assert vm.voice_metrics(speech, words=4)["words_per_minute"] == pytest.approx(120, abs=12)
    assert vm.voice_metrics(quiet, words=4)["words_per_minute"] == pytest.approx(120, abs=12)
    assert vm.voice_metrics(quiet, words=4)["silent_pct"] > 50


def test_register_and_expressiveness_labels():
    assert vm.register_label(120) == "male-ish"
    assert vm.register_label(200) == "female-ish"
    assert vm.register_label(None) == "no voice detected"
    assert vm.expressiveness_label(1.0) == "flat"
    assert vm.expressiveness_label(3.0) == "limited"
    assert vm.expressiveness_label(7.0) == "expressive"


@needs_ffmpeg
def test_measure_file_adds_loudness(tmp_path):
    path = str(tmp_path / "tone.wav")
    tone(200, 1500).export(path, format="wav")
    metrics = vm.measure_file(path, words=3, elapsed_s=0.5)
    assert metrics["lufs"] is not None and -40 < metrics["lufs"] < 0
    assert metrics["true_peak_db"] is not None
    assert metrics["realtime_factor"] > 0
    assert abs(metrics["median_f0_hz"] - 200) <= 5


# --------------------------------------------------- the benchmark itself


@needs_engine
@needs_ffmpeg
@pytest.mark.slow
def test_benchmark_script_measures_every_piper_variant(tmp_path):
    if te.find_piper_model() is None:
        pytest.skip("needs a piper voice model (python scripts/download_voice.py)")

    report = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "benchmark_voices.py"),
         "--engines", "piper", "--repeat", "2", "--json", str(report)],
        capture_output=True, text=True, timeout=1800, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "piper" in proc.stdout

    rows = json.loads(report.read_text(encoding="utf-8"))["voices"]
    assert rows
    for row in rows:
        assert 60 <= row["median_f0_hz"] <= 400, row      # a voice, not noise
        assert row["semitones"] > 1.0, row                # speech moves in pitch
        assert row["clipped_pct"] == 0.0, row             # nothing was driven into the roof
        assert 60 <= row["words_per_minute"] <= 400, row
        assert row["lufs"] is not None, row

    deterministic = [row for row in rows if "[det]" in row["voice"]]
    assert deterministic, f"the deterministic variant should be measured: {rows}"
    assert deterministic[0]["repeat_spread_pct"] == 0.0


def _benchmark(tmp_path, *args, cwd=None, env=None):
    report = tmp_path / "bench.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "benchmark_voices.py"),
         "--json", str(report), *args],
        capture_output=True, text=True, timeout=900, cwd=cwd or ROOT, env=env,
    )
    return proc, report


def test_benchmark_script_rejects_an_engine_it_does_not_have(tmp_path):
    proc, report = _benchmark(tmp_path, "--engines", "not_an_engine")
    assert proc.returncode == 1
    assert "Unknown engine" in proc.stdout
    assert "Traceback" not in proc.stderr
    assert not report.exists()


def test_benchmark_script_refuses_to_measure_a_different_engine(tmp_path):
    """Asking for piper with no voice installed must say so, not measure the fallback.

    Run from a directory with no models/ and no $PIPER_MODEL, so piper is genuinely
    unavailable no matter what the machine has installed.
    """
    env = {k: v for k, v in os.environ.items() if k != "PIPER_MODEL"}
    proc, report = _benchmark(tmp_path, "--engines", "piper", cwd=str(tmp_path), env=env)
    assert proc.returncode == 1
    assert "No requested engine is usable" in proc.stdout
    assert not report.exists()


@needs_engine
@needs_ffmpeg
def test_a_real_render_measures_like_a_voice(tmp_path):
    """End to end: the delivered audio has a plausible register, movement and no distortion."""
    srt = write_srt(str(tmp_path / "v.srt"), [
        (0, 4000, "Doctor Reyes spent the morning recalibrating the array."),
    ])
    out = str(tmp_path / "v.wav")
    proc = run_cli(["-p", srt, "-f", "wav", "--deterministic"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert os.path.exists(out)

    metrics = vm.measure_file(out, words=7)
    assert 70 <= metrics["median_f0_hz"] <= 350, metrics
    assert metrics["semitones"] > 1.0, metrics
    assert metrics["clipped_pct"] == 0.0, metrics
    assert 100 <= metrics["words_per_minute"] <= 400, metrics
    # the export was normalized, so it should land near the -16 LUFS target
    assert -20.0 <= metrics["lufs"] <= -12.0, metrics
