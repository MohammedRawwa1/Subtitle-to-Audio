"""Loader / normalization tests: formats, malformed repair, text expansion."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subtitle_io as S  # noqa: E402

BAD_SRT = (
    "\ufeff1\r\n"
    "00:00:01,000.123 --> 00:00:02,000.999\r\n"
    "<i>Hello</i> &amp; welcome\r\n"
    "\r\n"
    "2\r\n"
    "00:00:03,000 --> 00:00:02,500\r\n"
    "Pre\u00e7o \u20ac5, 50% & <b>bold</b>\r\n"
)

NO_MS_SRT = "1\n00:00:02 --> 00:00:04\nNo milliseconds here\n"

ARROW_SRT = "1\n00:00:01.500 -> 00:00:03.000\nDot separator and short arrow\n"

VTT = (
    "WEBVTT\n\n"
    "NOTE a comment block\n\n"
    "1\n"
    "00:00:01.000 --> 00:00:02.500 align:start position:0%\n"
    "Hello there\n\n"
    "2\n"
    "00:00:03.000 --> 00:00:05.000\n"
    "Second & line\n"
)

ASS = (
    "[Script Info]\nTitle: x\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    "Dialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,{\\an8}Hello\\Nworld\n"
    "Dialogue: 0,0:00:04.00,0:00:06.00,Default,,0,0,0,,Second line\n"
)

ARABIC_DIGITS_SRT = (
    "1\n"
    "\u0660\u0660:\u0660\u0660:\u0660\u0661,\u0660\u0660\u0660 --> 00:00:02,000\n"
    "Digits in the timestamp\n"
)


def write(tmp_path, name, body, encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(body.encode(encoding))
    return str(path)


# ----------------------------------------------------------------- timestamps

@pytest.mark.parametrize("token, expected", [
    ("00:00:01,000", 1000),
    ("00:00:01,000.123", 1000),          # extra precision dropped
    ("0:00:01.50", 1500),                # ASS centiseconds
    ("01:02.500", 62500),                # MM:SS, no hours
    ("00:00:02", 2000),                  # no milliseconds at all
    ("1:02:03,004", 3723004),
    ("\u0660\u0660:\u0660\u0660:\u0660\u0661,\u0660\u0660\u0660", 1000),  # Arabic-Indic digits
    ("nonsense", None),
])
def test_parse_timestamp_ms(token, expected):
    assert S.parse_timestamp_ms(token) == expected


# ------------------------------------------------------------------- formats

def test_srt_repairs(tmp_path):
    result = S.load_subtitles(write(tmp_path, "bad.srt", BAD_SRT))
    assert result.fmt == "srt"
    assert result.encoding == "utf-8-sig"
    assert len(result.cues) == 2
    assert [c.start_ms for c in result.cues] == [1000, 2500]      # cue 2 was reversed
    assert [c.end_ms for c in result.cues] == [2000, 3000]
    assert any("BOM" in r for r in result.repairs)
    assert any("reversed" in r for r in result.repairs)
    # CRLF is normalized silently: it is normal, not a defect worth reporting.
    assert not any("CRLF" in r for r in result.repairs)
    assert result.cues[0].text == "Hello and welcome"             # tags + entity
    assert "euros" in result.cues[1].text and "percent" in result.cues[1].text
    assert "<b>" not in result.cues[1].text


def test_srt_without_milliseconds(tmp_path):
    result = S.load_subtitles(write(tmp_path, "noms.srt", NO_MS_SRT))
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (2000, 4000)


def test_srt_dot_separator_and_short_arrow(tmp_path):
    result = S.load_subtitles(write(tmp_path, "arrow.srt", ARROW_SRT))
    assert len(result.cues) == 1 and result.cues[0].start_ms == 1500


def test_vtt_cue_settings_and_note_blocks(tmp_path):
    result = S.load_subtitles(write(tmp_path, "t.vtt", VTT))
    assert result.fmt == "vtt"
    assert len(result.cues) == 2
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 2500)
    assert result.cues[0].text == "Hello there"
    assert result.cues[1].text == "Second and line"   # NOTE block not turned into a cue


def test_ass_override_tags_and_escapes(tmp_path):
    result = S.load_subtitles(write(tmp_path, "t.ass", ASS))
    assert result.fmt == "ass"
    assert len(result.cues) == 2
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1500, 3000)
    assert result.cues[0].text == "Hello world"      # {\an8} stripped, \N became a space
    assert result.cues[1].text == "Second line"


def test_ass_detected_by_extension_for_ssa(tmp_path):
    result = S.load_subtitles(write(tmp_path, "t.ssa", ASS))
    assert result.fmt == "ass"


def test_arabic_indic_digits_in_timestamp(tmp_path):
    result = S.load_subtitles(write(tmp_path, "ar.srt", ARABIC_DIGITS_SRT))
    assert (result.cues[0].start_ms, result.cues[0].end_ms) == (1000, 2000)


def test_extension_less_file_is_sniffed(tmp_path):
    result = S.load_subtitles(write(tmp_path, "movie", VTT))
    assert result.fmt == "vtt" and len(result.cues) == 2


def test_last_cue_kept_without_trailing_blank_line(tmp_path):
    body = "1\n00:00:00,000 --> 00:00:02,000\nOnly one cue, no trailing newline"
    result = S.load_subtitles(write(tmp_path, "one.srt", body))
    assert len(result.cues) == 1


def test_glued_cues_without_blank_separator_are_recovered(tmp_path):
    body = (
        "1\n00:00:00,000 --> 00:00:01,000\nFirst\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSecond\n"
    )
    result = S.load_subtitles(write(tmp_path, "glued.srt", body))
    assert len(result.cues) == 2
    assert any("blank separator" in r for r in result.repairs)


def test_zero_length_cue_dropped(tmp_path):
    body = (
        "1\n00:00:00,000 --> 00:00:02,000\nKeep me\n\n"
        "2\n00:00:03,000 --> 00:00:03,000\nDrop me\n"
    )
    result = S.load_subtitles(write(tmp_path, "zero.srt", body))
    assert [c.text for c in result.cues] == ["Keep me"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.load_subtitles(str(tmp_path / "nope.srt"))


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError):
        S.load_subtitles(write(tmp_path, "empty.srt", "no cues in here"))


# --------------------------------------------------------------- normalization

@pytest.mark.parametrize("raw, expected", [
    ("<i>Italic</i>", "Italic"),
    ("{\\an8}Top", "Top"),
    ("a &amp; b", "a and b"),
    ("Line one<br>Line two", "Line one Line two"),
    ("Use a < b > c safely", "Use a < b > c safely"),      # not a real tag
    ("Trailing braces {} stay", "Trailing braces {} stay"),
    ("Em\u2014dash", "Em, dash"),
    ("Wait\u2026", "Wait,"),
    ("50% off", "50 percent off"),
    ("\u20ac5 and $10", "5 euros and 10 dollars"),
    ("#TitanForce", "hashtag TitanForce"),
    ("Mr. Smith", "Mister Smith"),
    ("We use oxygen, etc. every day", "we use oxygen, etcetera every day".capitalize()),
    ("11 + 4", "11 plus 4"),
    ("  spaced   out  ", "spaced out"),
    ("- A dash line", "A dash line"),
    ("", ""),
])
def test_normalize_text(raw, expected):
    assert S.normalize_text(raw) == expected


def test_normalize_text_expand_off_keeps_symbols():
    assert S.normalize_text("50% & <i>x</i>", expand=False) == "50% & x"


def test_ass_escapes_only_applied_for_ass():
    assert S.normalize_text("C:\\new", ass_escapes=False) == "C:\\new"
    assert S.normalize_text("C:\\new", ass_escapes=True) == "C: ew"


def test_normalize_disabled_keeps_raw_text(tmp_path):
    body = "1\n00:00:00,000 --> 00:00:02,000\n<i>raw &amp; text</i>\n"
    result = S.load_subtitles(write(tmp_path, "raw.srt", body), normalize=False)
    assert result.cues[0].text == "<i>raw &amp; text</i>"
