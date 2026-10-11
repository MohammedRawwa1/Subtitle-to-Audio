"""CLI contract tests: documented flags, exit codes, and the feature switches."""

import json
import os

import pytest

from tests.conftest import needs_engine, run_cli, write_srt

DOCUMENTED_FLAGS = [
    "--path", "--format", "--engine", "--voice", "--piper-model", "--gender", "--auto-voice",
    "--speaker-voices", "--auto-speakers", "--piper-noise", "--piper-noise-w", "--deterministic",
    "--edge-proxy", "--base-rate", "--max-rate", "--pyttsx3-rate",
    "--max-wpm", "--max-speed", "--no-stretch", "--stretch-tolerance", "--fit-attempts",
    "--tts-retries", "--overflow", "--max-compress", "--overlap", "--no-stitch",
    "--no-normalize", "--no-level",
    "--no-loudnorm", "--music", "--music-volume", "--lexicon", "--say", "--concurrency",
    "--cache-dir", "--no-cache", "--clear-cache", "--fallback-after", "--metrics", "--report",
    "--strict",
]


def test_help_lists_every_documented_flag():
    proc = run_cli(["--help"], timeout=120)
    assert proc.returncode == 0
    missing = [flag for flag in DOCUMENTED_FLAGS if flag not in proc.stdout]
    assert not missing, f"documented but missing from --help: {missing}"


def test_help_documents_the_choices():
    proc = run_cli(["--help"], timeout=120)
    assert "{cut,bleed,compress}" in proc.stdout
    assert "{mix,trim,shift}" in proc.stdout
    assert "{auto,edge_tts,piper,pyttsx3}" in proc.stdout


def test_missing_subtitle_file_exits_with_a_setup_error(tmp_path):
    proc = run_cli(["-p", str(tmp_path / "nope.srt")], timeout=300)
    assert proc.returncode == 1
    assert "No such subtitle file" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_unreadable_subtitles_exit_with_a_setup_error(tmp_path):
    bad = tmp_path / "bad.srt"
    bad.write_text("this file has no cues at all\n", encoding="utf-8")
    proc = run_cli(["-p", str(bad)], timeout=300)
    assert proc.returncode == 1
    assert "Cannot read subtitles" in proc.stdout


def test_bad_say_pair_is_rejected(tmp_path):
    proc = run_cli(["-p", str(tmp_path / "x.srt"), "--say", "no-equals-sign"], timeout=120)
    assert proc.returncode != 0
    assert "NAME=VALUE" in (proc.stdout + proc.stderr)


def test_bad_lexicon_file_is_rejected(tmp_path):
    lex = tmp_path / "lex.json"
    lex.write_text("[1, 2, 3]", encoding="utf-8")
    proc = run_cli(["-p", str(tmp_path / "x.srt"), "--lexicon", str(lex)], timeout=120)
    assert proc.returncode != 0
    assert "JSON object" in (proc.stdout + proc.stderr)


def test_clear_cache_reports_and_exits(tmp_path):
    proc = run_cli(["--clear-cache", "--cache-dir", str(tmp_path / "cache"), "-p", "unused.srt"],
                   timeout=120)
    assert proc.returncode == 0
    assert "cached render" in proc.stdout


# ------------------------------------------------------- feature switches (need an engine)

MARKUP_SRT = "1\n00:00:00,000 --> 00:00:02,500\n<i>Raw</i> &amp; 50% text\n"


@needs_engine
def test_text_is_normalized_by_default(tmp_path):
    srt = write_srt(str(tmp_path / "m.srt"), [(0, 2500, "<i>Raw</i> &amp; 50% text")])
    proc = run_cli(["-p", srt, "-f", "wav"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Raw and 50 percent text" in proc.stdout
    assert "text normalization: on" in proc.stdout


@needs_engine
def test_no_normalize_keeps_the_raw_text(tmp_path):
    srt = write_srt(str(tmp_path / "m.srt"), [(0, 2500, "<i>Raw</i> &amp; 50% text")])
    proc = run_cli(["-p", srt, "-f", "wav", "--no-normalize"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "<i>Raw</i> &amp; 50% text" in proc.stdout
    assert "text normalization: off" in proc.stdout


@needs_engine
def test_lexicon_and_say_overrides_change_the_spoken_text(tmp_path):
    srt = write_srt(str(tmp_path / "l.srt"), [(0, 3000, "The SQL server and nginx are up")])
    lex = tmp_path / "lex.json"
    lex.write_text(json.dumps({"SQL": "sequel"}), encoding="utf-8")

    proc = run_cli(["-p", srt, "-f", "wav", "--lexicon", str(lex), "--say", "nginx=engine x",
                    "--no-normalize"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "The sequel server and engine x are up" in proc.stdout
    assert "Pronunciation overrides applied: 2 entries" in proc.stdout


@needs_engine
def test_speaker_labels_are_metadata_not_speech(tmp_path):
    srt = write_srt(str(tmp_path / "s.srt"), [
        (0, 2500, "JENNY: the reactor is holding"),
        (2600, 5200, "MARCUS: for now, at least."),
    ])
    proc = run_cli(["-p", srt, "-f", "wav", "--auto-speakers",
                    "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 0, proc.stdout + proc.stderr

    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert set(report["speaker_voices"]) == {"JENNY", "MARCUS"}
    assert {u["speaker"] for u in report["units_detail"]} == {"JENNY", "MARCUS"}
    assert "SPEAKER(JENNY)" in proc.stdout and "SPEAKER(MARCUS)" in proc.stdout
    # the label is a tag, not something to read out loud
    assert '"the reactor is holding"' in proc.stdout
    assert '"for now, at least."' in proc.stdout

    # Distinct voices need an engine with more than one voice: edge_tts rotates a
    # pool, piper has a single model unless --speaker-voices names one per speaker.
    if "edge_tts" in report["engines"][:1]:
        assert len(set(report["speaker_voices"].values())) == 2


@needs_engine
def test_explicit_speaker_voices_are_used_and_unknown_names_are_reported(tmp_path):
    srt = write_srt(str(tmp_path / "s.srt"), [(0, 2500, "JENNY: the reactor is holding")])
    proc = run_cli(["-p", srt, "-f", "wav", "--speaker-voices", "JENNY=en-US-AriaNeural,GHOST=x",
                    "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["speaker_voices"] == {"JENNY": "en-US-AriaNeural"}
    assert "never seen in the subtitles" in proc.stdout and "GHOST" in proc.stdout


@needs_engine
def test_cache_makes_the_second_run_reuse_renders(tmp_path):
    srt = write_srt(str(tmp_path / "c.srt"), [(0, 2500, "Cache me once and reuse me.")])
    cache_dir = str(tmp_path / "cache")

    first = run_cli(["-p", srt, "-f", "wav", "--cache-dir", cache_dir, "--report",
                     str(tmp_path / "r1.json")])
    second = run_cli(["-p", srt, "-f", "wav", "--cache-dir", cache_dir, "--report",
                      str(tmp_path / "r2.json")])
    assert first.returncode == 0 and second.returncode == 0

    stats1 = json.loads((tmp_path / "r1.json").read_text(encoding="utf-8"))["cache"]
    stats2 = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))["cache"]
    assert stats1["hits"] == 0 and stats1["misses"] >= 1
    assert stats2["hits"] >= 1, f"cache did not serve the second run: {stats2}"


@needs_engine
def test_no_cache_never_populates_the_cache(tmp_path):
    srt = write_srt(str(tmp_path / "c.srt"), [(0, 2500, "Do not cache this line.")])
    cache_dir = tmp_path / "cache"
    proc = run_cli(["-p", srt, "-f", "wav", "--no-cache", "--cache-dir", str(cache_dir),
                    "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["cache"]["enabled"] is False
    assert not cache_dir.exists() or not os.listdir(cache_dir)


@needs_engine
def test_strict_gate_passes_a_clean_run(tmp_path):
    srt = write_srt(str(tmp_path / "ok.srt"), [
        (0, 3000, "A comfortable line that fits."),
        (3200, 7000, "Another comfortable line that fits."),
    ])
    proc = run_cli(["-p", srt, "-f", "wav", "--strict", "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["strict"]["passed"] is True


@needs_engine
def test_strict_gate_fails_an_impossible_line(tmp_path):
    # ~700 characters into a 3 second window: no rate or atempo can save it.
    impossible = " ".join(["Relaying"] * 120)
    srt = write_srt(str(tmp_path / "bad.srt"), [(0, 3_000, impossible)])
    proc = run_cli(["-p", srt, "-f", "wav", "--overflow", "cut", "--strict",
                    "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "STRICT GATE FAILED" in proc.stdout
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["strict"]["passed"] is False
    assert any("cut" in reason for reason in report["strict"]["reasons"])


@needs_engine
@pytest.mark.slow
def test_metrics_reports_alignment_honestly(tmp_path):
    srt = write_srt(str(tmp_path / "m.srt"), [
        (0, 3000, "First line for the alignment check."),
        (4000, 7500, "Second line, well separated."),
    ])
    proc = run_cli(["-p", srt, "-f", "wav", "--metrics", "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    alignment = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["alignment"]
    assert alignment["units_checked"] == 2
    assert alignment["onsets_matched"] + alignment["onsets_missing"] == 2
    assert alignment["search_window_ms"] == 250
    if alignment["onsets_matched"]:
        assert alignment["within_150ms_pct"] is not None
