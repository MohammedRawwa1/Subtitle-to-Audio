"""Measure every available voice instead of guessing which one sounds better.

Renders one fixed paragraph at each engine's natural rate, then reports objective
proxies for the things listeners notice -- register (median pitch), expressiveness
(pitch range in semitones), pacing (words per minute over voiced audio only),
distortion (clipping, true peak), delivery (LUFS) and whether the engine even repeats
itself. Piper is measured twice: with its default rhythm noise, and with the duration
noise off, which is the switch that makes line lengths repeatable.

    python scripts/benchmark_voices.py
    python scripts/benchmark_voices.py --engines piper --json bench.json

Listening is still the only test of taste; this tells you which voices are plausible,
which are monotone, and what reproducibility costs in expressiveness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydub import AudioSegment  # noqa: E402

from audio_post import trim_trailing_silence  # noqa: E402
from tts_engines import (  # noqa: E402
    ENGINE_CLASSES,
    PIPER_DETERMINISTIC_NOISE_W,
    build_chain,
    find_piper_model,
)
from voice_metrics import measure_loudness_file, voice_metrics  # noqa: E402

PARAGRAPH = (
    "Station log, day one hundred and twelve. "
    "The relay is still silent, and the crew has stopped asking why. "
    "Doctor Reyes spent the morning recalibrating the array; by noon, "
    "forty percent of the panels were reporting power again. "
    "If the signal comes back tonight, we will know it was worth the wait."
)
WORDS = len(PARAGRAPH.split())


async def render(engine, voice: str, out_path: str) -> float:
    """Render the paragraph at the engine's natural rate; returns seconds taken."""
    start = time.perf_counter()
    ok = await engine.render_async(PARAGRAPH, voice, engine.base_speed, out_path)
    elapsed = time.perf_counter() - start
    if not ok or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"{engine.name} produced no audio")
    return elapsed


async def measure(engine, voice: str, tmp: str, label: str, name: str = None,
                  repeats: int = 3) -> dict:
    """    Render a few times and measure: the spread in raw length is the reproducibility."""
    # Warm up first: loading the 63 MB voice would otherwise be charged to whichever
    # variant happens to run first and make the realtime column incomparable.
    await render(engine, voice, os.path.join(tmp, f"{label}_warm.wav"))

    paths, elapsed = [], None
    for pass_index in range(max(2, repeats)):
        path = os.path.join(tmp, f"{label}_{pass_index}.wav")
        took = await render(engine, voice, path)
        elapsed = took if elapsed is None else elapsed
        paths.append(path)

    # Compare the raw renders: trimming to a silence boundary would hide the drift.
    lengths = [len(AudioSegment.from_file(p)) for p in paths]
    mean = sum(lengths) / len(lengths)
    spread = (max(lengths) - min(lengths)) / mean * 100.0 if mean else 0.0

    audio = trim_trailing_silence(AudioSegment.from_file(paths[0]))
    metrics = voice_metrics(audio, words=WORDS, elapsed_s=elapsed)
    metrics["ms_per_char"] = round(len(audio) / max(1, len(PARAGRAPH)), 1)
    metrics["repeatable"] = len(set(lengths)) == 1
    metrics["repeat_spread_pct"] = round(spread, 2)
    metrics["raw_lengths_ms"] = lengths
    metrics["engine"] = engine.name
    metrics["voice"] = name or (os.path.basename(voice) if voice else "(engine default)")
    loud = measure_loudness_file(paths[0])
    if loud:
        metrics.update(loud)
    return metrics


def variants(chain):
    """What to measure: every installed piper voice, plus piper without rhythm noise."""
    from tts_engines import PiperEngine, find_piper_models

    for engine in chain:
        if engine.name == "piper":
            for model in find_piper_models() or [engine.model]:
                spoken = os.path.basename(model).removesuffix(".onnx")
                yield spoken, PiperEngine(model=model), spoken
            if engine.noise_w is None:
                spoken = f"{os.path.basename(engine.model or '').removesuffix('.onnx')} [det]"
                yield spoken, PiperEngine(model=engine.model,
                                          noise_w=PIPER_DETERMINISTIC_NOISE_W), spoken
        else:
            yield engine.name, engine, None


async def main_async(args) -> int:
    chain = build_chain(args.engines[0] if len(args.engines or []) == 1 else "auto",
                        piper_model=find_piper_model(), verbose=False)
    if args.engines:
        wanted = {name.strip() for name in args.engines}
        chain = [e for e in chain if e.name in wanted]
        if not chain:
            # Measuring some other engine would answer a question nobody asked.
            print(f"No requested engine is usable: {', '.join(sorted(wanted))} "
                  f"(install it, and for piper also a voice model)")
            return 1

    rows = []
    with tempfile.TemporaryDirectory(prefix="voice_bench_") as tmp:
        for label, engine, spoken in variants(chain):
            voice = engine.resolve_voice(None, args.gender)
            try:
                rows.append(await measure(engine, voice, tmp, label.replace(" ", "_"),
                                          spoken, repeats=args.repeat))
            except Exception as e:
                print(f"  {label}: skipped ({type(e).__name__}: {str(e)[:90]})")

    if not rows:
        print("No voice could be measured (install piper-tts with a voice model, or edge-tts).")
        return 1

    header = ("engine", "voice", "rtf", "ms/char", "wpm", "LUFS", "peak dBFS",
              "clip %", "F0 med", "register", "F0 range", "expressive", "length spread")
    print(f"\nNatural-rate measurements of one {WORDS}-word paragraph:\n")
    print("{:9s} {:27s} {:>6s} {:>7s} {:>6s} {:>7s} {:>6s} {:>6s} {:>8s} {:>10s} {:>9s} {:>11s} {:>7s}".format(*header))
    for row in rows:
        print("{:9s} {:27s} {:>6s} {:>7s} {:>6s} {:>7s} {:>6s} {:>6s} {:>8s} {:>10s} {:>9s} {:>11s} {:>7s}".format(
            row["engine"],
            row["voice"][:27],
            f"{row.get('realtime_factor', 0):.1f}x",
            f"{row.get('ms_per_char', 0):.1f}",
            f"{row.get('words_per_minute') or 0:.0f}",
            f"{row.get('lufs', float('nan')):.1f}" if row.get("lufs") is not None else "-",
            f"{row.get('true_peak_db', float('nan')):.1f}" if row.get("true_peak_db") is not None else "-",
            f"{row.get('clipped_pct', 0):.2f}",
            f"{row.get('median_f0_hz') or 0:.0f} Hz",
            row.get("register", ""),
            f"{row.get('semitones') or 0:.1f} st",
            row.get("expressiveness", ""),
            f"{row.get('repeat_spread_pct', 0):.2f}%",
        ))

    print("\nRead it as: 150-255 Hz reads female-ish, 80-155 Hz male-ish; under ~2 st of")
    print("pitch range is monotone, 4+ st is normal conversational reading; clip % should")
    print("be 0.00; a length spread of 0.00% means the engine repeats itself exactly")
    print("(piper needs --deterministic for that). LUFS is the render before normalization.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"paragraph_words": WORDS, "voices": rows}, f, indent=2)
        print(f"\nWrote {args.json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="*",
                        help="only these engines (edge_tts, piper, pyttsx3)")
    parser.add_argument("--gender", default="female", choices=["male", "female"],
                        help="voice hint for engines that pick their own (default: female)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="renders per voice, to measure how repeatable it is (default: 3)")
    parser.add_argument("--json", help="also write the measurements here")
    args = parser.parse_args(argv)
    if args.engines:
        unknown = sorted(set(args.engines) - set(ENGINE_CLASSES))
        if unknown:
            print(f"Unknown engine(s): {', '.join(unknown)}; choose from "
                  f"{', '.join(sorted(ENGINE_CLASSES))}")
            return 1
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
