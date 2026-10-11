"""The hostile 5-minute SRT, end to end.

Covers rapid-fire tight windows, long lines in short windows, short lines in very
long windows (must NOT be slowed), an impossible line (must be cut and reported),
overlapping cues, a sentence split across cues (must be stitched into one phrase),
unicode/punctuation, and a busy tail. Verifies the timeline, the cue starts against
real detected speech, the log and the JSON report.

The generated SRT is compared against the committed fixture, so the deliverable in
tests/fixtures/ can never silently drift from the generator.
"""

import json
import os
import re

import pytest
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import pipeline_core as pc
from subtitle_io import load_subtitles
from tests.conftest import FIXTURES, needs_engine, needs_ffmpeg, run_cli, write_srt

FIXTURE_SRT = os.path.join(FIXTURES, "hard_5min.srt")
TOTAL_MS = 300_000  # exactly 5 minutes
FIT_TOLERANCE = 0.05  # the tool's --stretch-tolerance default

RAPID = [
    "Shields up.", "Come about, now.", "He's closing fast.", "Report status.",
    "No answer yet.", "We're hit!", "Hold steady.", "Fire at will.",
    "It's a trap.", "Fall back now.", "Systems online.", "Copy that.",
    "Stand by.", "Go, go, go.", "They're gone.", "Clear the deck.",
]
MEDIUM = [
    "The diagnostics keep reporting the same intermittent fault in the port stabiliser.",
    "Long-range sensors show a debris field where the relay station used to be, Captain.",
    "If we re-route power through the auxiliary conduits we might buy ourselves ten minutes.",
    "Their weapons are charged, but they have not locked on and they are holding position.",
    "Whatever happened down there, the logs were wiped before anyone thought to look.",
    "I want a full inventory of every escape pod and a head count within the hour.",
    "The hull temperature is climbing faster than the cooling loop can compensate for.",
    "We can take the long way around the nebula, but it adds four days and a lot of fuel.",
    "Tell the crew to seal the lower decks; I will not risk anyone on a guess.",
    "Comms are back, though the signal keeps breaking up whenever we turn the array.",
    "It is quiet out here, which is exactly the part that worries me the most.",
    "Run the simulation again, and this time do not optimise for a happy ending.",
]
LONGWIN = ["We made it.", "Quiet.", "Not yet.", "Almost there.", "Listen.", "Wait."]
HEAVY = [
    "Every single time we push the reactor past eighty percent the secondary manifold "
    "fractures in a way the manual swears is impossible, and I am tired of guessing why.",
    "The council will demand a full accounting of what we did at the colony, and I would "
    "rather hand them a clean record than an explanation nobody in this room believes.",
    "There is a difference between a retreat and a rout, and the only thing separating "
    "them is whether the people behind you make it back to the ships in time.",
    "I have read the telemetry twice, and the numbers do not describe a malfunction; they "
    "describe somebody deliberately venting the atmosphere from the observation deck.",
]
IMPOSSIBLE = (
    "The transmission was not a distress call at all but a complete manifest of every "
    "vessel in the system together with their registry numbers, crew complements, current "
    "headings, weapon loadouts, and the exact minute each one is scheduled to leave dock, "
    "which means somebody on the inside copied the whole fleet schedule onto a relay that "
    "was supposed to have been dark for six years and then chose this precise moment to "
    "send it straight to us knowing we would have no time to verify a single word of it."
)
STITCH = [
    "I keep telling myself that the door at the end of this corridor",
    "is just a door, and that whatever is standing behind it",
    "has been waiting for us to stop being afraid of it.",
]
UNICODE = [
    "C'est d\u00e9j\u00e0 fini \u2014 n'est-ce pas ?",
    "Pre\u00e7o, sa\u00fade, cora\u00e7\u00e3o\u2026 tudo bem.",
    "\u00abGuillemets\u00bb \u2014 em\u2014dash and an ellipsis\u2026",
    "N\u00fameros: 42, 7 %, e #hashtags.",
]
FILLER = [
    "Engineering reports the starboard coupling is holding, for now, at ninety-one percent.",
    "Navigation wants a decision before the next shift change, one way or the other.",
    "The mess hall has been converted to a triage bay, which tells you how the day is going.",
    "We are running on reserves, and the reserves are running on optimism and cold coffee.",
    "Somebody has been siphoning rations from the aft locker, and I intend to find out who.",
    "The captain's log ends mid-sentence, which is not the part that frightens me.",
    "If the storm holds we can make the crossing before the last of the fuel burns off.",
    "Every panel in this corridor is dead, and yet the door at the end still has power.",
    "I have been staring at this schematic for six hours and it still makes no sense.",
    "The refugees are aboard, accounted for, and asking more questions than we can answer.",
    "They will expect a report by morning, so somebody had better invent the truth tonight.",
    "Whatever you decide, decide it now, because the window closes in about four minutes.",
]


def build():
    """Return [(start_ms, end_ms, text, section), ...] filling exactly 5 minutes."""
    segs = []
    t = 2_000

    for text in RAPID:                                  # tight, back-to-back
        segs.append((t, t + 1_100, text, "rapid"))
        t += 1_100

    t += 300
    for i, text in enumerate(MEDIUM):                   # natural, small gaps
        width = 2_400 if i % 2 else 2_800
        segs.append((t, t + width, text, "medium"))
        t += width + 300

    t += 400
    for text in LONGWIN:                                # short text, huge window
        segs.append((t, t + 8_000, text, "longwin"))
        t += 8_400

    t += 300
    for text in HEAVY:                                  # very long in a tight window
        segs.append((t, t + 1_700, text, "heavy"))
        t += 2_000

    t += 300
    segs.append((t, t + 1_500, IMPOSSIBLE, "impossible"))
    t += 1_800

    t += 300
    for text in UNICODE:
        segs.append((t, t + 2_400, text, "unicode"))
        t += 2_700

    t += 300
    segs.append((t, t + 3_000, "Two contacts, bearing zero three five.", "overlap"))  # A
    segs.append((t + 800, t + 3_800, "They're not answering our hails.", "overlap"))  # B over A

    t += 4_200                                          # one sentence across three cues
    for text in STITCH:
        segs.append((t, t + 2_200, text, "stitch"))
        t += 2_400

    t += 3_900
    i = 0
    while t + 4_200 + 1_500 < TOTAL_MS:                 # busy tail out to 5 minutes
        segs.append((t, t + 4_200, FILLER[i % len(FILLER)], "filler"))
        t += 4_600
        i += 1
    t += 300
    segs.append((max(t, TOTAL_MS - 2_500), TOTAL_MS, "That is the end of the recording.", "final"))
    return segs


RE_LOG = re.compile(
    r"^\s*\[(\d+)/(\d+)\]\s+\S+\s+->\s+\S+\s+\|\s+(\S+)\s+(\S+)\s+\|\s+fit\s+([\d.]+)x(.*?)\|\s+\"(.*)\"\s*$"
)


def parse_log(log):
    rows = []
    for line in log.splitlines():
        match = RE_LOG.match(line)
        if match:
            rows.append({
                "i": int(match.group(1)),
                "engine": match.group(3),
                "rate": match.group(4),
                "fit": float(match.group(5)),
                "tags": match.group(6).strip(),
            })
    return rows


def check(segs, report, log):
    """Verify the run through the report and the log; returns (results, ok)."""
    rows = parse_log(log)
    results, ok = [], True

    def rec(name, cond, extra=""):
        nonlocal ok
        results.append((name, cond, extra))
        if not cond:
            ok = False

    units = report["units_detail"]
    sections = {u["unit"]: {segs[i - 1][3] for i in u["cues"]} for u in units}
    covered = [c for u in units for c in u["cues"]]

    rec("log lines match units", len(rows) == report["units"],
        f"log={len(rows)} report={report['units']}")
    rec("every cue rendered exactly once", sorted(covered) == list(range(1, len(segs) + 1)),
        f"covered={len(covered)} of {len(segs)}")

    stitched = [u for u in units if u["stitched"]]
    rec("sentence split across cues is stitched",
        len(stitched) == 1 and len(stitched[0]["cues"]) == 3,
        f"stitched={[u['cues'] for u in stitched]}")
    rec("stitched unit is tagged",
        any(any(t.startswith("STITCH(") for t in u["tags"]) for u in stitched))
    rec("stitched unit keeps the full group window",
        all(u["window_ms"] >= 6_000 for u in stitched))

    cuts = dict(report["cuts"]["units"])
    rec("no sub-epsilon phantom truncations", all(c >= 40 for c in cuts.values()), f"cuts={cuts}")
    rec("long-window lines never accelerated",
        not [u for u in units if "longwin" in sections[u["unit"]]
             and ({"rate-fit", "safety-atempo"} & set(u["tags"]))])
    # The contract is the inaudible epsilon, not a pixel-perfect fit: a line may end up
    # to CUT_EPSILON_MS past its window and only what is left beyond that is trimmed.
    # Tighter than that measures the engine rather than the code -- piper is stochastic
    # (the same text at the same length_scale gave 89_856, 92_416 and 94_976 samples on
    # three consecutive runs), so its line durations move by a few ms per run.
    def overrun_ms(u):
        return round(u["fit"] * u["window_ms"]) - u["window_ms"]

    rec("no line ends audibly past its window",
        all(overrun_ms(u) <= pc.CUT_EPSILON_MS + 1 for u in units),
        f"max overrun={max(overrun_ms(u) for u in units)} ms")
    rec("fitter engages on tight/overrunning cues",
        report["rate_fitted"] + report["safety_atempo"] >= 5,
        f"{report['rate_fitted']} + {report['safety_atempo']}")
    # A normal cue may only be trimmed by what the fitter itself accepted as close
    # enough for that window; anything beyond that is a real chop and belongs to the
    # designed-hard cues (a very long line in a very short one).
    hard = {"impossible", "heavy"}
    accepted_ms = {u["unit"]: max(pc.CUT_EPSILON_MS, FIT_TOLERANCE * u["window_ms"])
                   for u in units}
    off_target = [i for i, c in cuts.items()
                  if not sections[i] <= hard and c > accepted_ms[i] + 1]
    rec("material cuts only on designed-hard cues", not off_target, f"off-target units={off_target}")
    rec("impossible cue gets a large material cut",
        any("impossible" in sections[i] and c > 400 for i, c in cuts.items()), f"cuts={cuts}")
    rec("no line bled past its cue under the default policy", report["bleeds"]["count"] == 0)

    rec("total duration == timeline", abs(report["final_ms"] - TOTAL_MS) <= 60,
        f"{report['final_ms']} ms")
    alignment = report.get("alignment")
    rec("alignment measured", bool(alignment))
    if alignment:
        rec("alignment >= 90% within 150 ms", alignment["within_150ms_pct"] >= 90.0,
            f"{alignment}")
        rec("alignment matched an onset for most units",
            alignment["onsets_matched"] >= 0.8 * alignment["units_checked"],
            f"{alignment}")
    return results, ok


@needs_engine
@needs_ffmpeg
@pytest.mark.slow
def test_hard_five_minute_srt(tmp_path):
    segs = build()
    srt = write_srt(str(tmp_path / "hard_5min.srt"), [(s, e, t) for s, e, t, _ in segs])
    wav = str(tmp_path / "hard_5min.wav")
    report_path = str(tmp_path / "report.json")

    proc = run_cli(["-p", srt, "-f", "wav", "--metrics", "--report", report_path], timeout=3600)
    log = proc.stdout + proc.stderr

    assert proc.returncode == 0, log[-3000:]
    report = json.loads(open(report_path, encoding="utf-8").read())

    results, ok = check(segs, report, log)

    # Independent confirmation straight from the rendered file.
    audio = AudioSegment.from_wav(wav)
    results.append(("rendered WAV duration == timeline", abs(len(audio) - TOTAL_MS) <= 60,
                    f"dur={len(audio)}"))
    onsets = [r[0] for r in detect_nonsilent(audio, min_silence_len=250, silence_thresh=-45)]
    probes = [s for s, _e, _t, tag in segs if tag in ("filler", "medium", "longwin")]
    missed = [s for s in probes if not any(abs(o - s) <= 200 for o in onsets)]
    results.append(("cue starts align with speech", not missed,
                    f"missed={len(missed)}/{len(probes)} {missed[:5]}"))

    ok = ok and all(passed for _n, passed, _e in results)
    summary = "\n".join(("PASS " if p else "FAIL ") + n + (f" :: {e}" if e and not p else "")
                        for n, p, e in results)
    assert ok, "hard 5-minute SRT failed:\n" + summary


def test_committed_fixture_matches_the_generator(tmp_path):
    """The fixture is the deliverable; keep it in step with the generator."""
    segs = build()
    generated = write_srt(str(tmp_path / "hard_5min.srt"), [(s, e, t) for s, e, t, _ in segs])
    assert os.path.exists(FIXTURE_SRT), (
        f"missing {FIXTURE_SRT}; regenerate it from the generator in this module")
    with open(generated, encoding="utf-8") as f:
        new = f.read()
    with open(FIXTURE_SRT, encoding="utf-8") as f:
        committed = f.read()
    assert new == committed, (
        "tests/fixtures/hard_5min.srt is out of date; regenerate it from build() in this module")
    assert load_subtitles(FIXTURE_SRT).cues[-1].end_ms == TOTAL_MS
