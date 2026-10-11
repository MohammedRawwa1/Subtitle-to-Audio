"""Pure pipeline logic: rate fitting, cue stitching, overlap and overflow policy.

Nothing here touches audio files, TTS engines or ffmpeg -- it is all arithmetic and
text rules, which is what makes it cheap to test exhaustively.
"""

from __future__ import annotations

import math
import re

__all__ = [
    "CUT_EPSILON_MS",
    "STITCH_MAX_GAP_MS",
    "STITCH_MAX_CUES",
    "STITCH_MAX_WINDOW_MS",
    "RAPID_CHAIN_GAP_MS",
    "OVERFLOWS",
    "OVERLAPS",
    "RenderUnit",
    "apply_overlap_policy",
    "build_units",
    "decide_overflow",
    "format_timestamp",
    "solve_rate",
    "curve_k",
]

# A window overrun this small (a few ms of atempo rounding) is inaudible: leave it
# and do not report the line as truncated.
CUT_EPSILON_MS = 40

# Cue stitching: consecutive cues that read as one sentence are spoken as one phrase.
STITCH_MAX_GAP_MS = 600
STITCH_MAX_CUES = 5
STITCH_MAX_WINDOW_MS = 15_000
RAPID_CHAIN_GAP_MS = 250

OVERFLOWS = ("cut", "bleed", "compress")
OVERLAPS = ("mix", "trim", "shift")

_SENTENCE_END = re.compile(r"[.!?][\"')\u201d]?\s*$")
_SPEAKER_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 .'\-]{0,24}?)\s*:\s+(?=\S)")


def format_timestamp(ms: float) -> str:
    """Milliseconds -> the usual HH:MM:SS,mmm subtitle stamp."""
    total = int(ms)
    h, rem = divmod(total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_ = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


# ---------------------------------------------------------------- rate fitting

def curve_k(rate_lo: float, dur_lo: float, rate_hi: float, dur_hi: float, default: float = 2.0) -> float:
    """Fit duration ~ speed^-k from two measured points.

    Both edge_tts and piper compress faster than 1/speed, so a straight ratio step
    overshoots and leaves an unnaturally fast line. Measuring k from real renders
    lets the next step land on the window.
    """
    try:
        if rate_lo <= 0 or rate_hi <= 0 or rate_lo == rate_hi or dur_lo <= 0 or dur_hi <= 0:
            return default
        k = math.log(dur_lo / dur_hi) / math.log(rate_hi / rate_lo)
        if 0.2 <= k <= 15.0:
            return k
    except Exception:
        pass
    return default


def solve_rate(points, window_ms: float, tolerance: float, max_speed: float, default_k: float = 2.0):
    """From measured (speed, duration_ms) points, return the next speed to try, or None.

    The goal is the SLOWEST speed whose speech still fits `window_ms`:
      * nothing fits yet  -> step faster, extrapolating the measured curve;
      * one fits but wastes a lot of the window -> step back toward the window
        using an exact extrapolation between the overlong and the fitting point;
      * otherwise -> stop, the current take is the answer.
    """
    if window_ms <= 0 or not points:
        return None
    fit_thresh = window_ms * (1.0 + max(0.0, tolerance))
    fitting = [(r, d) for r, d in points if d <= fit_thresh]
    overlong = [(r, d) for r, d in points if d > fit_thresh]

    if fitting:
        hi_r, hi_d = min(fitting, key=lambda p: p[0])
        # Only hunt for a slower fit when the current one wastes a lot of the window
        # (renders are not free).
        if hi_d <= 0.90 * window_ms and overlong:
            lo_r, lo_d = max(overlong, key=lambda p: p[0])
            if lo_r < hi_r:
                k = curve_k(lo_r, lo_d, hi_r, hi_d, default_k)
                need = lo_r * (lo_d / window_ms) ** (1.0 / k)
                margin = max(1e-6, (hi_r - lo_r) * 0.02)
                cand = min(max(need, lo_r + margin), hi_r - margin)
                if lo_r < cand < hi_r:
                    return cand
        return None

    r_last, d_last = max(points, key=lambda p: p[0])
    if d_last <= 0:
        return None
    overrun = d_last / window_ms
    pts = sorted(points, key=lambda p: p[0])
    k = curve_k(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], default_k) if len(pts) >= 2 else default_k
    need = min(r_last * overrun ** (1.0 / k), max_speed)
    return need if need > r_last * (1.0 + 1e-6) else None


# ---------------------------------------------------------------- cue stitching

def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def _join_cue_text(left: str, right: str) -> str:
    """Join two halves of one sentence, healing a word broken by a hyphen."""
    left = left.rstrip()
    right = right.lstrip()
    if left.endswith("-") and right[:1].islower():
        return left[:-1] + right
    return f"{left} {right}".strip()


def speaker_of(text: str, known: set | None = None, auto: bool = False) -> str:
    """Return the speaker label of a cue, or '' if it is just prose.

    A leading "NAME:" is only treated as a speaker when it is a known name, looks
    like a name tag (all caps), or the caller explicitly opted into auto speakers --
    otherwise "Note: this matters" would lose its first word.
    """
    match = _SPEAKER_LABEL.match(text or "")
    if not match:
        return ""
    label = match.group(1).strip()
    if known and label.upper() in known:
        return label.upper()
    if auto and (label.isupper() or len(label.split()) <= 3):
        return label.upper()
    return ""


def strip_speaker_label(text: str, label: str) -> str:
    if not label:
        return text
    match = _SPEAKER_LABEL.match(text or "")
    if not match or match.group(1).strip().upper() != label.upper():
        return text
    return text[match.end():].strip()


class RenderUnit:
    """One spoken utterance: a single cue, or a stitched run of continuation cues."""

    __slots__ = ("cues", "start_ms", "end_ms", "text", "speak", "speaker", "place_ms", "limit_ms")

    def __init__(self, cues, start_ms, end_ms, text, speaker: str = ""):
        self.cues = cues
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text
        self.speaker = speaker
        self.speak = text          # what actually gets spoken (may drop a "NAME:" tag)
        self.place_ms = start_ms
        self.limit_ms = None       # max audio length once the overlap policy is known

    @property
    def window_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def stitched(self) -> bool:
        return len(self.cues) > 1


def _can_stitch(unit: RenderUnit, cue) -> bool:
    """True when `cue` continues the sentence started by the current unit."""
    last = unit.cues[-1]
    if len(unit.cues) >= STITCH_MAX_CUES:
        return False
    gap = cue.start_ms - last.end_ms
    if gap < 0 or gap > STITCH_MAX_GAP_MS:
        return False
    if cue.end_ms - unit.start_ms > STITCH_MAX_WINDOW_MS:
        return False
    if _ends_sentence(last.text) or unit.speaker != speaker_of(cue.text):
        return False
    if _SPEAKER_LABEL.match(cue.text):
        return False   # a name tag here means a new speaker starts, never merge
    # A lowercase continuation is always a stitch; capitalised text only chains when
    # the cues are truly back-to-back (rapid-fire bursts).
    return cue.text[:1].islower() or gap <= RAPID_CHAIN_GAP_MS


def build_units(cues, stitch: bool = True, known_speakers: set | None = None, auto_speakers: bool = False):
    """Group cues into render units (one unit = one spoken utterance)."""
    units: list[RenderUnit] = []
    for cue in cues:
        speaker = speaker_of(cue.text, known_speakers, auto_speakers)
        if units and stitch and _can_stitch(units[-1], cue):
            unit = units[-1]
            unit.text = _join_cue_text(unit.text, cue.text)
            unit.end_ms = max(unit.end_ms, cue.end_ms)
            unit.cues.append(cue)
            continue
        units.append(RenderUnit(cues=[cue], start_ms=cue.start_ms, end_ms=cue.end_ms,
                                text=cue.text, speaker=speaker))
    for unit in units:
        unit.speak = strip_speaker_label(unit.text, unit.speaker)
    return units


def apply_overlap_policy(units, policy: str, total_duration_ms: int = 0):
    """Resolve overlapping units. Returns (units, tags) where tags maps index -> note."""
    tags = {}
    if policy == "shift":
        cursor = 0
        for i, unit in enumerate(units):
            if unit.start_ms < cursor:
                shift = cursor - unit.start_ms
                window_ms = unit.end_ms - unit.start_ms   # read before moving the start
                unit.start_ms = cursor
                unit.end_ms = unit.start_ms + window_ms
                tags[i] = f"SHIFT(+{shift}ms)"
            cursor = max(cursor, unit.end_ms)
    elif policy == "trim":
        for i, unit in enumerate(units):
            nxt = units[i + 1] if i + 1 < len(units) else None
            if nxt is not None and nxt.start_ms < unit.end_ms:
                available = nxt.start_ms - unit.start_ms
                if available > 0:
                    unit.limit_ms = available
                    tags[i] = "OVERLAP-TRIM"
    return units, tags


def decide_overflow(overrun_ms: float, policy: str) -> str:
    """What to do with speech that still overran its window.

    'keep'   -> the line fits (within the inaudible epsilon)
    'bleed'  -> keep the whole line and let it run past the cue
    'compress' -> pull it in with pitch-preserving atempo
    'cut'    -> trim it to the window and report the loss
    """
    if overrun_ms <= CUT_EPSILON_MS:
        return "keep"
    return policy if policy in ("bleed", "compress") else "cut"
