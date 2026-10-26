"""CLI contract tests: documented flags, exit codes, and the feature switches."""

import json
import os

import pytest

import subtitle_to_audio as cli
from tests.conftest import needs_engine, needs_ffmpeg, run_cli, write_srt


def parser_flags():
    """Every long option the CLI actually defines, read from the parser itself."""
    flags = []
    for action in cli.build_parser()._actions:
        flags.extend(opt for opt in action.option_strings if opt.startswith("--"))
    return sorted(set(flags))

DOCUMENTED_FLAGS = [
    "--path", "--format", "--output-dir", "--engine", "--voice", "--piper-model", "--gender", "--auto-voice",
    "--speaker-voices", "--auto-speakers", "--piper-noise", "--piper-noise-w", "--deterministic",
    "--edge-proxy", "--base-rate", "--max-rate", "--pyttsx3-rate",
    "--max-wpm", "--max-speed", "--no-stretch", "--stretch-tolerance", "--fit-attempts",
    "--tts-retries", "--overflow", "--max-compress", "--overlap", "--no-stitch",
    "--no-normalize", "--no-level",
    "--no-loudnorm", "--music", "--music-volume", "--lexicon", "--say", "--concurrency",
    "--cache-dir", "--no-cache", "--clear-cache", "--fallback-after", "--metrics", "--report",
    "--strict", "--preview", "--voices", "--guide", "--help",
]

# The choices a user picks from, which --guide and --help must both spell out.
DOCUMENTED_CHOICES = [
    "{auto,edge_tts,piper,pyttsx3}", "{wav,mp3,m4a,ogg}", "{cut,bleed,compress}",
    "{mix,trim,shift}", "{male,female}",
]


def test_help_lists_every_documented_flag():
    proc = run_cli(["--help"], timeout=120)
    assert proc.returncode == 0
    missing = [flag for flag in DOCUMENTED_FLAGS if flag not in proc.stdout]
    assert not missing, f"documented but missing from --help: {missing}"


def test_the_documented_flag_list_matches_the_parser():
    """DOCUMENTED_FLAGS is only worth having if it cannot drift from the real CLI."""
    assert sorted(DOCUMENTED_FLAGS) == parser_flags()


def test_guide_covers_every_flag_the_parser_has():
    """The one-page guide is the promise that a user can find every option."""
    proc = run_cli(["--guide"], timeout=120)
    assert proc.returncode == 0
    missing = [flag for flag in parser_flags() if flag not in proc.stdout]
    assert not missing, f"the parser has options the guide never mentions: {missing}"
    assert "--help" in proc.stdout


def test_help_documents_the_choices():
    proc = run_cli(["--help"], timeout=120)
    for choice in DOCUMENTED_CHOICES:
        assert choice in proc.stdout, f"{choice} missing from --help"


def test_guide_is_the_one_page_cheat_sheet():
    """`--guide` is how a user discovers the options without reading the README."""
    proc = run_cli(["--guide"], timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    missing = [flag for flag in DOCUMENTED_FLAGS if flag not in proc.stdout]
    assert not missing, f"documented but missing from --guide: {missing}"
    for choice in DOCUMENTED_CHOICES:
        assert choice in proc.stdout, f"{choice} missing from --guide"
    assert "START HERE" in proc.stdout
    assert "download_voice.py" in proc.stdout


def test_guide_runs_without_a_subtitle_file_or_ffmpeg():
    """Discovering the options must not need an input file."""
    proc = run_cli(["--guide"], timeout=120)
    assert proc.returncode == 0
    assert "Traceback" not in proc.stderr
    assert "the following arguments are required" not in proc.stderr


def test_voices_lists_the_choices_and_exits(tmp_path):
    proc = run_cli(["--voices"], timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Engines" in proc.stdout
    for engine in ("edge_tts", "piper", "pyttsx3"):
        assert engine in proc.stdout
    assert "Add a voice" in proc.stdout
    assert not list(tmp_path.iterdir())        # listing writes nothing


def test_positional_path_is_the_same_as_the_flag(tmp_path):
    """`subtitle_to_audio.py movie.srt` and `-p movie.srt` must behave alike."""
    srt = write_srt(str(tmp_path / "p.srt"), [(0, 2200, "The simple form works.")])
    out = tmp_path / "p.wav"

    positional = run_cli([srt, "-f", "wav"], timeout=600)
    assert positional.returncode == 0, positional.stdout + positional.stderr
    assert out.exists()
    out.unlink()

    flagged = run_cli(["-p", srt, "-f", "wav"], timeout=600)
    assert flagged.returncode == 0, flagged.stdout + flagged.stderr
    assert out.exists()


@needs_engine
@needs_ffmpeg
def test_output_dir_receives_the_generated_audio(tmp_path):
    srt = write_srt(str(tmp_path / "input.srt"), [(0, 2500, "Put this audio in results.")])
    output_dir = tmp_path / "results" / "nested"

    proc = run_cli([srt, "-f", "wav", "--output-dir", str(output_dir)], timeout=600)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (output_dir / "input.wav").exists()
    assert not (tmp_path / "input.wav").exists()


def test_no_input_at_all_asks_for_one(tmp_path):
    proc = run_cli([], timeout=120)
    assert proc.returncode == 2                       # argparse usage error
    assert "give the subtitle file" in proc.stderr


@needs_engine
def test_non_latin_subtitles_survive_a_piped_run(tmp_path):
    """Regression: a redirected run must not die encoding non-Latin text.

    On Windows a redirected stdout used the console codepage, so the per-unit log
    raised UnicodeEncodeError the moment the line was Arabic (or Cyrillic, or CJK).
    run_cli pipes the output, which is exactly that case.
    """
    path = tmp_path / "ar.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\nمرحباً بكم في الدرس التاسع عشر.\n\n"
        "2\n00:00:03,200 --> 00:00:06,000\nسنتعرف اليوم على القوائم المنسدلة.\n",
        encoding="utf-8",
    )
    proc = run_cli([str(path), "--preview", "1", "--report", str(tmp_path / "r.json")],
                   timeout=600)
    assert "UnicodeEncodeError" not in proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "الدرس" in proc.stdout                    # logged, not mangled or dropped
    assert (tmp_path / "ar.preview.m4a").exists()


@needs_engine
def test_preview_speaks_only_the_first_cues(tmp_path):
    srt = write_srt(str(tmp_path / "long.srt"), [
        (0, 2500, "First line of the lecture."),
        (2700, 5200, "Second line, which the preview should also read."),
        (5400, 8000, "Third line."),
        (8200, 11000, "Fourth line, which the preview must skip."),
    ])
    report = str(tmp_path / "r.json")
    proc = run_cli([srt, "--preview", "2", "--report", report], timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Preview: speaking the first 2 cue(s)" in proc.stdout
    preview = tmp_path / "long.preview.m4a"
    assert preview.exists() and preview.stat().st_size > 0
    assert not (tmp_path / "long.m4a").exists()       # the full file is not written

    data = json.loads(open(report, encoding="utf-8").read())
    assert data["preview_cues"] == 2
    assert data["units"] == 2
    assert data["cues"] == 2
    assert data["final_ms"] == 5200                   # the timeline stops at the last previewed cue


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
def test_successful_delivery_removes_cache_before_the_next_run(tmp_path):
    srt = write_srt(str(tmp_path / "c.srt"), [(0, 2500, "Cache this during the run.")])
    cache_dir = str(tmp_path / "cache")

    first = run_cli(["-p", srt, "-f", "wav", "--cache-dir", cache_dir, "--report",
                     str(tmp_path / "r1.json")])
    assert first.returncode == 0, first.stdout + first.stderr
    assert not os.path.exists(cache_dir)

    second = run_cli(["-p", srt, "-f", "wav", "--cache-dir", cache_dir, "--report",
                      str(tmp_path / "r2.json")])
    assert second.returncode == 0, second.stdout + second.stderr

    stats1 = json.loads((tmp_path / "r1.json").read_text(encoding="utf-8"))["cache"]
    stats2 = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))["cache"]
    assert stats1["hits"] == 0 and stats1["misses"] >= 1
    assert stats2["hits"] == 0 and stats2["misses"] >= 1
    assert not os.path.exists(cache_dir)


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
