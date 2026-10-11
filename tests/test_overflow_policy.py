"""End-to-end tests for --overflow, run against a tiny SRT with one impossible line.

The line is far too long for its window, so every policy has something to do:
  cut      -> trimmed to the window, reported as TRUNCATED, timeline preserved
  bleed    -> kept whole, runs past the cue
  compress -> pitch-preserving atempo pulls it into the window instead
"""

import json
import os

import pytest
from pydub import AudioSegment

import pipeline_core as pc
from tests.conftest import needs_engine, run_cli, write_srt

# Matches the tool's "an overrun this small is inaudible" threshold: after
# compressing, a line must not be left materially past its window.
MAX_INAUDIBLE_MS = 50

pytestmark = needs_engine

IMPOSSIBLE = (
    "The transmission was not a distress call at all but a complete manifest of every "
    "vessel in the system together with their registry numbers, crew complements, and "
    "the exact minute each one is scheduled to leave dock, which means somebody copied "
    "the whole fleet schedule onto a relay that was supposed to have been dark for years."
)


def write_overflow_srt(path):
    cues = [
        (0, 2_500, "Short and comfortable."),
        (2_600, 4_100, IMPOSSIBLE),        # ~290 chars in 1.5 s: cannot possibly fit
        (4_200, 7_000, "And then it was over."),
    ]

    write_srt(path, cues)
    return cues[-1][1]


def run(tmp_path, policy, extra=()):
    srt = str(tmp_path / f"overflow_{policy}.srt")
    total = write_overflow_srt(srt)
    report = str(tmp_path / f"report_{policy}.json")
    wav = str(tmp_path / f"overflow_{policy}.wav")

    proc = run_cli(["-p", srt, "-f", "wav", "--overflow", policy,
                    "--report", report, *extra])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(report, encoding="utf-8") as f:
        return json.load(f), total, wav, proc.stdout + proc.stderr


@pytest.mark.parametrize("policy", ["cut", "bleed", "compress"])
def test_every_policy_produces_audio(tmp_path, policy):
    report, total, wav, log = run(tmp_path, policy)
    assert os.path.exists(wav)
    assert len(AudioSegment.from_wav(wav)) > 0
    assert report["units"] == 3
    assert "Overflow: " + policy in log


def test_cut_trims_and_reports_the_impossible_line(tmp_path):
    report, total, wav, _ = run(tmp_path, "cut")
    assert report["cuts"]["count"] == 1
    assert report["cuts"]["total_ms"] > 1_000
    assert report["bleeds"]["count"] == 0
    # exact timeline: the cut policy never exceeds the subtitle timeline
    assert report["final_ms"] == total


def test_bleed_keeps_the_whole_line_and_overruns(tmp_path):
    report, total, wav, _ = run(tmp_path, "bleed")
    assert report["bleeds"]["count"] == 1
    assert report["bleeds"]["total_ms"] > 1_000
    assert report["cuts"]["count"] == 0
    # the whole line was kept, so it is longer than the window it had to fit
    bleeding = next(u for u in report["units_detail"] if u["bleed_ms"])
    assert bleeding["fit"] > 1.0
    # nothing was dropped and the mix still covers the whole timeline
    assert report["final_ms"] >= total
    assert len(AudioSegment.from_wav(wav)) >= total - 60


def test_compress_fits_the_line_with_atempo(tmp_path):
    report, total, wav, _ = run(tmp_path, "compress")
    assert report["compressed"] == 1
    assert report["cuts"]["count"] == 0          # nothing needed dropping
    assert report["final_ms"] == total
    unit = next(u for u in report["units_detail"] if any(t.startswith("COMPRESS") for t in u["tags"]))
    # it really fits now: the window is essentially filled and nothing overran it
    residue = round(unit["fit"] * unit["window_ms"]) - unit["window_ms"]
    assert unit["fit"] >= 0.9
    assert residue <= max(MAX_INAUDIBLE_MS, pc.CUT_EPSILON_MS)


def test_policies_are_reported_consistently(tmp_path):
    """Whichever policy runs, the timeline and unit count must not drift."""
    runs = [run(tmp_path, p) for p in ("cut", "bleed", "compress")]
    cut, bleed, compress = [r[0] for r in runs]
    total = runs[0][1]
    assert {r["units"] for r in (cut, bleed, compress)} == {3}
    # cut and compress both land inside the window; bleed is the one policy that
    # deliberately lets the speech run past it.
    assert cut["fit"]["max"] <= 1.01 and compress["fit"]["max"] <= 1.01
    assert bleed["fit"]["max"] > 1.0
    assert cut["final_ms"] == bleed["final_ms"] == compress["final_ms"] == total


def test_bad_overflow_value_is_rejected(tmp_path):
    srt = str(tmp_path / "x.srt")
    write_overflow_srt(srt)
    proc = run_cli(["-p", srt, "--overflow", "nonsense"], timeout=300)
    assert proc.returncode != 0
    assert "invalid choice" in (proc.stderr + proc.stdout).lower()


def test_music_bed_end_to_end(tmp_path):
    srt = str(tmp_path / "music.srt")
    total = write_overflow_srt(srt)
    music = str(tmp_path / "bed.wav")
    AudioSegment.silent(duration=total).export(music, format="wav")
    out = str(tmp_path / "music.m4a")
    proc = run_cli(["-p", srt, "-f", "m4a", "--music", music, "--music-volume", "0.2"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert os.path.exists(out) and os.path.getsize(out) > 0
