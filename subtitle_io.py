"""Tolerant subtitle loading, repair and text normalization.

Supported inputs
----------------
* ``.srt``            -- SubRip
* ``.vtt``            -- WebVTT (cue settings, NOTE/STYLE/REGION blocks)
* ``.ass`` / ``.ssa`` -- Advanced SubStation Alpha
* anything else       -- delegated to ``pysubparser`` (``.sub``, ``.txt``)

Everything that had to be repaired while reading is collected in
``LoadResult.repairs`` so a run can report it instead of dying on a single
malformed timestamp (the old separate ``fix.py`` behaviour, folded in here).

Repairs covered:
  * BOM, CRLF, non-UTF-8 encodings, Arabic-Indic digits in timestamps
  * extra fractional precision (``00:00:01,000.123``)
  * ``.`` used where ``,`` belongs, and timestamps with no milliseconds
  * ``->`` / ``\u2192`` arrows, swapped start/end, ``H:MM:SS`` vs ``MM:SS``
  * cue blocks with a missing/odd index, stray blank lines inside a block
  * WebVTT cue settings after the end timestamp, header/NOTE/STYLE blocks
  * ASS/SSA override tags and ``\\N`` / ``\\h`` escapes (in text normalization)
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass, field

__all__ = [
    "Cue",
    "LoadResult",
    "load_subtitles",
    "normalize_text",
    "parse_timestamp_ms",
    "detect_format",
]

# ---------------------------------------------------------------- timestamps

# Arabic-Indic (U+0660..0669) and Extended Arabic-Indic (U+06F0..06F9) digits.
_DIGIT_TRANS = {
    **{0x0660 + i: 0x30 + i for i in range(10)},
    **{0x06F0 + i: 0x30 + i for i in range(10)},
}

_ARROW = re.compile(r"\s*(?:-->|->|\u2192|\u2013>|\u2014>)\s*")
_ARROW_HINT = re.compile(r"-->|->|\u2192")

# HH:MM:SS,mmm  |  MM:SS.mmm  |  H:MM:SS.cc (ASS centiseconds)  |  extra precision
_TS = re.compile(
    r"(?:(?P<h>\d{1,3}):)?(?P<m>\d{1,3}):(?P<s>\d{1,2})"
    r"(?:(?P<sep>[,.])(?P<frac>\d{1,6}))?"
)


def parse_timestamp_ms(token: str) -> int | None:
    """Parse one timestamp token to milliseconds, or None if there is none.

    Tolerates a missing hour field, a comma or a dot separator, a missing
    millisecond field, ASS centiseconds (``0:00:01.50`` == 1500 ms) and extra
    fractional precision (anything past 3 digits is dropped).
    """
    if not token:
        return None
    token = token.translate(_DIGIT_TRANS)
    m = _TS.search(token)
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m"))
    seconds = int(m.group("s"))
    frac = m.group("frac") or "0"
    millis = int((frac + "000")[:3])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _parse_timestamp_pair(line: str) -> tuple[int | None, int | None]:
    """Split a subtitle timing line into (start_ms, end_ms)."""
    line = line.translate(_DIGIT_TRANS)
    parts = _ARROW.split(line, maxsplit=1)
    if len(parts) != 2:
        return None, None
    start = parse_timestamp_ms(parts[0])
    end = parse_timestamp_ms(parts[1])
    return start, end


def _looks_like_timing(line: str) -> bool:
    return bool(_ARROW_HINT.search(line)) and bool(_TS.search(line.translate(_DIGIT_TRANS)))


# ------------------------------------------------------------------- formats

def detect_format(path: str, text: str = "") -> str:
    """Return one of 'srt' | 'vtt' | 'ass' | 'sub' | 'txt'."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "ssa":
        ext = "ass"
    if ext in ("srt", "vtt", "ass", "sub", "txt"):
        return ext
    head = text[:2048].lstrip("\ufeff \t\r\n")
    if head.upper().startswith("WEBVTT"):
        return "vtt"
    if re.search(r"^\[(?:Events|Script Info)\]", head, re.I | re.M) or "Dialogue:" in head:
        return "ass"
    return "srt"


# ------------------------------------------------------------------- parsing

@dataclass
class Cue:
    """One subtitle cue with the text already normalized for speech."""

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class LoadResult:
    cues: list[Cue] = field(default_factory=list)
    fmt: str = "srt"
    encoding: str = "utf-8"
    repairs: list[str] = field(default_factory=list)
    path: str = ""

    @property
    def total_duration_ms(self) -> int:
        return self.cues[-1].end_ms if self.cues else 0


def _read_text(path: str, repairs: list[str]) -> tuple[str, str]:
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        repairs.append("stripped UTF-8 BOM")
        return raw.decode("utf-8-sig"), "utf-8-sig"
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc != "utf-8":
                repairs.append(f"decoded as {enc} (not valid UTF-8)")
            return text, enc
        except UnicodeDecodeError:
            continue
    repairs.append("decoded with replacement characters")
    return raw.decode("utf-8", errors="replace"), "utf-8/replace"


def _normalize_newlines(text: str) -> str:
    # CRLF is not a defect -- most SRTs written on Windows have it -- so this is
    # normalized silently rather than reported as a repair.
    text = text.replace("\ufeff", "")
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


_INDEX_LINE = re.compile(r"\d{1,7}")


def _parse_srt(text: str, repairs: list[str]) -> list[tuple[int, int, list[str]]]:
    """Split a SubRip/WebVTT body into raw (start_ms, end_ms, text_lines).

    Scanned line by line rather than by blank-line blocks, so a file that is
    missing its blank separator (or its final newline) still parses instead of
    swallowing the next cue.
    """
    raw: list[list] = []
    current: list | None = None

    for line in text.split("\n"):
        if _looks_like_timing(line):
            # The index line of this cue may be glued to the previous cue's text.
            if current is not None and current[2]:
                if _INDEX_LINE.fullmatch(current[2][-1].strip()):
                    current[2].pop()
                    repairs.append("recovered an index line glued to the previous cue")
                if current[2]:
                    repairs.append("recovered a cue that was missing its blank separator")
            start, end = _parse_timestamp_pair(line)
            if start is None or end is None:
                repairs.append(f"dropped an unreadable timing line: {line[:60]!r}")
                current = None
                continue
            current = [start, end, []]
            raw.append(current)
            continue

        if line.strip() == "":
            current = None          # blank line ends the cue (and hides the next index)
            continue
        if current is None:
            continue                # index line / cue identifier before its timing line
        current[2].append(line)

    return [(int(r[0]), int(r[1]), r[2]) for r in raw]


def _parse_vtt(text: str, repairs: list[str]) -> list[tuple[int, int, list[str]]]:
    if text.lstrip().upper().startswith("WEBVTT"):
        text = re.sub(r"^[ \t]*WEBVTT[^\n]*\n", "", text, count=1, flags=re.I)
    kept: list[str] = []
    skipping = False
    for line in text.split("\n"):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith(("NOTE", "STYLE", "REGION")):
            skipping = True
            continue
        if stripped == "":
            skipping = False
            kept.append(line)
            continue
        if not skipping:
            kept.append(line)
    return _parse_srt("\n".join(kept), repairs)


_ASS_PREFIX = re.compile(r"^\s*dialogue\s*:", re.I)


def _parse_ass(text: str, repairs: list[str]) -> list[tuple[int, int, list[str]]]:
    raw: list[tuple[int, int, list[str]]] = []
    for line in text.split("\n"):
        if not _ASS_PREFIX.match(line):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            repairs.append("dropped a malformed ASS Dialogue line")
            continue
        start = parse_timestamp_ms(parts[1])
        end = parse_timestamp_ms(parts[2])
        if start is None or end is None:
            repairs.append(f"dropped an unreadable ASS timing: {parts[:3]!r}")
            continue
        body = parts[9].replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        raw.append((start, end, [body]))
    return raw


# --------------------------------------------------------------- text cleanup

# Only real subtitle tags are stripped, so text like "a < b > c" survives.
_TAG = re.compile(
    r"</?(?:i|b|u|s|em|strong|font|ruby|rt|rp|br|p|c|v|lang|span)\b[^>]*>",
    re.I,
)
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.I)
_BRACE = re.compile(r"\{[^}]+\}")          # ASS override blocks: {\an8}, {\i1}, {Y:i}
_WS = re.compile(r"[ \t\u00a0\u202f]+")

_ABBREV = (
    (re.compile(r"\bMr\.", re.I), "Mister"),
    (re.compile(r"\bMrs\.", re.I), "Missus"),
    (re.compile(r"\bMs\.", re.I), "Miss"),
    (re.compile(r"\bDr\.", re.I), "Doctor"),
    (re.compile(r"\bProf\.", re.I), "Professor"),
    (re.compile(r"\bSt\.", re.I), "Saint"),
    (re.compile(r"\bvs\.?\b", re.I), "versus"),
    (re.compile(r"\betc\.", re.I), "etcetera"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\bNo\.\s*(?=\d)", re.I), "number "),
)
_CURRENCY = (("$", "dollars"), ("\u20ac", "euros"), ("\u00a3", "pounds"), ("\u00a5", "yen"))


def _expand_text(text: str) -> str:
    """Text normalization: the things TTS engines read badly.

    Conservative on purpose -- digits are left alone because modern engines read
    them well, while symbols, dashes and hashtags are not.
    """
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dash and ellipsis are pauses, not words.
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    text = re.sub(r"\s*\u2026\s*", ", ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"#(\w[\w-]*)", r"hashtag \1", text)
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d)\s*\+\s*(?=\d)", r"\1 plus ", text)
    text = re.sub(r"\bC\+\+", "C plus plus", text)
    for symbol, word in _CURRENCY:
        text = re.sub(
            re.escape(symbol) + r"\s*(\d[\d.,]*)",
            lambda m, w=word: f"{m.group(1)} {w}",
            text,
        )
    for pattern, replacement in _ABBREV:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r"^[-\u2013\u2014]\s+", "", text)   # leading dialogue dash
    # Tidy punctuation spacing without touching paths/URLs: no space is added
    # before a backslash or a slash, so "C:\new" and "http://x" stay intact.
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([,.;!?])(?=[^\s\d\\/])", r"\1 ", text)
    return text


def apply_lexicon(text: str, lexicon: dict | None) -> str:
    """Apply pronunciation overrides: whole words, case-insensitive, longest first.

    Used for names and technical terms the engine would get wrong ("SQL" ->
    "sequel", "nginx" -> "engine x"). Runs after normalization so it wins.
    """
    if not text or not lexicon:
        return text
    sources = sorted((source for source in lexicon if source), key=len, reverse=True)
    if not sources:
        return text
    # One pass, so a replacement is never re-scanned by a later rule ("New York"
    # must not then be hit by a "New" rule) and the longest match wins.
    pattern = re.compile(
        "|".join(rf"(?<![\w-]){re.escape(source)}(?![\w-])" for source in sources),
        re.I,
    )
    replacements = {source.lower(): lexicon[source] for source in sources}
    return pattern.sub(lambda m: replacements[m.group(0).lower()], text)


def normalize_text(text: str, *, expand: bool = True, ass_escapes: bool = False) -> str:
    """Turn raw subtitle text into something worth speaking.

    Strips markup/override tags and HTML entities, flattens line breaks, then
    (unless ``expand=False``) expands the symbols and abbreviations TTS engines
    mispronounce.
    """
    if not text:
        return ""
    out = _BRACE.sub(" ", text)
    out = _BR_RE.sub(" ", out)
    out = _TAG.sub("", out)
    if ass_escapes:
        out = out.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    out = html.unescape(out)
    if expand:
        out = _expand_text(out)
    out = _WS.sub(" ", out)
    out = re.sub(r"\s*\n\s*", " ", out)
    return out.strip()


# --------------------------------------------------------------------- public

def load_subtitles(path: str, *, normalize: bool = True, expand: bool = True) -> LoadResult:
    """Load a subtitle file, repairing what can be repaired.

    Raises FileNotFoundError / ValueError with a readable message when the file
    is missing or yields no usable cues.
    """
    repairs: list[str] = []
    text, encoding = _read_text(path, repairs)
    text = _normalize_newlines(text)
    fmt = detect_format(path, text)

    if fmt == "srt":
        raw = _parse_srt(text, repairs)
    elif fmt == "vtt":
        raw = _parse_vtt(text, repairs)
    elif fmt == "ass":
        raw = _parse_ass(text, repairs)
    else:
        raw = _parse_via_pysubparser(path, fmt, repairs)

    ass_escapes = fmt == "ass"
    cues: list[Cue] = []
    for start, end, body in raw:
        joined = " ".join(body)
        if start > end:
            repairs.append(f"swapped a reversed cue ({start} > {end})")
            start, end = end, start
        if end <= start:
            repairs.append("dropped a cue with zero length")
            continue
        cue_text = normalize_text(joined, expand=expand, ass_escapes=ass_escapes) if normalize \
            else joined.strip()
        if not cue_text:
            repairs.append("dropped a cue with no text")
            continue
        cues.append(Cue(index=len(cues) + 1, start_ms=int(start), end_ms=int(end), text=cue_text))

    if not cues:
        raise ValueError(f"no usable subtitle cues found in {path}")

    return LoadResult(cues=cues, fmt=fmt, encoding=encoding, repairs=repairs, path=path)


def _parse_via_pysubparser(path: str, fmt: str, repairs: list[str]) -> list[tuple[int, int, list[str]]]:
    """Fallback path for formats we do not parse ourselves (.sub, .txt)."""
    try:
        from pysubparser import parser as _parser
    except Exception as e:  # pragma: no cover - dependency is pinned
        raise ValueError(f"cannot read .{fmt} files: {e}") from e
    try:
        parsed = list(_parser.parse(path))
    except Exception as e:
        raise ValueError(f"cannot read {path} as .{fmt}: {e}") from e
    repairs.append(f"read .{fmt} via pysubparser")
    out: list[tuple[int, int, list[str]]] = []
    for sub in parsed:
        def _ms(t):
            return ((t.hour * 60 + t.minute) * 60 + t.second) * 1000 + t.microsecond // 1000
        body = getattr(sub, "lines", None) or [getattr(sub, "text", "") or ""]
        out.append((_ms(sub.start), _ms(sub.end), list(body)))
    return out
