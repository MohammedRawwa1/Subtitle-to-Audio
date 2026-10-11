"""Command line entry point: subtitles in, synchronized voice-over out.

The work lives in focused modules so each piece can be tested on its own:

    subtitle_io.py    reading/repairing subtitles, text normalization, lexicon
    pipeline_core.py  pure logic: rate fitting, stitching, overlap, overflow
    tts_engines.py    edge_tts / piper / pyttsx3 behind one interface + cache
    audio_post.py     atempo, levelling, two-pass EBU R128, music bed, export
    pipeline.py       orchestration and reporting
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import traceback

from audio_post import FORMATS
from pipeline import generate_audio
from pipeline_core import OVERFLOWS, OVERLAPS
from tts_engines import (
    EDGE_VOICE_DEFAULT,
    ENGINE_CLASSES,
    RenderCache,
    find_piper_model,
)

EXIT_OK = 0
EXIT_SETUP_ERROR = 1
EXIT_STRICT_FAILED = 3
EXIT_INTERRUPTED = 130


def _parse_pairs(values, flag: str) -> dict:
    """Parse repeated or comma-separated NAME=VALUE options into a dict."""
    pairs = {}
    for raw in values or []:
        for chunk in str(raw).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise SystemExit(f"{flag}: expected NAME=VALUE, got {chunk!r}")
            name, value = chunk.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name or not value:
                raise SystemExit(f"{flag}: expected NAME=VALUE, got {chunk!r}")
            pairs[name.upper()] = value
    return pairs


def _load_lexicon_file(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"--lexicon: cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit("--lexicon: expected a JSON object of {\"word\": \"replacement\"}")
    return {str(k): str(v) for k, v in data.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synchronized audio from subtitle files. Each line is spoken at a "
            "rate chosen to fit its exact time window, using an online neural voice "
            "(edge_tts), an offline neural voice (piper) or an offline SAPI voice."
        )
    )
    parser.add_argument("-p", "--path", required=True,
                        help="subtitle file path (.srt, .vtt, .ass, .ssa, .sub, .txt)")
    parser.add_argument("-f", "--format", choices=list(FORMATS), default="m4a",
                        help="output audio format (default: m4a)")

    voices = parser.add_argument_group("voices")
    voices.add_argument("--engine", choices=["auto", *ENGINE_CLASSES], default="auto",
                        help="TTS engine; auto tries edge_tts, then piper, then pyttsx3")
    voices.add_argument("--voice", default=EDGE_VOICE_DEFAULT,
                        help=f"edge_tts voice ShortName (default: {EDGE_VOICE_DEFAULT})")
    voices.add_argument("--piper-model",
                        help="path to a piper .onnx voice (default: $PIPER_MODEL or models/*.onnx)")
    voices.add_argument("--piper-noise", type=float,
                        help="piper expressiveness: higher is livelier and less stable "
                             "(model default is usually 0.667)")
    voices.add_argument("--piper-noise-w", type=float,
                        help="piper timing noise: higher varies the rhythm more "
                             "(model default is usually 0.8)")
    voices.add_argument("--edge-proxy",
                        help="proxy URL for the online engine, for networks Microsoft "
                             "refuses (else $EDGE_TTS_PROXY / $HTTPS_PROXY)")
    voices.add_argument("--deterministic", action="store_true",
                        help="turn piper's duration noise off so re-runs come out the same "
                             "(add --piper-noise 0 for byte-identical audio)")
    voices.add_argument("--gender", choices=["male", "female"], default="female",
                        help="gender hint for the fallback voice (default: female)")
    voices.add_argument("--auto-voice", action="store_true",
                        help="pick the voice from the subtitle text instead of --voice")
    voices.add_argument("--speaker-voices", action="append", metavar="NAME=VOICE",
                        help="per-speaker voice, repeatable or comma separated "
                             "(edge ShortName or piper .onnx path)")
    voices.add_argument("--auto-speakers", action="store_true",
                        help="detect \"NAME:\" tags and rotate voices between speakers")

    speaking = parser.add_argument_group("speaking rate")
    speaking.add_argument("--base-rate", default="+0%",
                          help="natural starting rate; fitting never goes below it (default: +0%%)")
    speaking.add_argument("--max-rate", type=int, default=30,
                          help="fastest rate the fitter may use, in percent (default: 30)")
    speaking.add_argument("--pyttsx3-rate", type=int, default=165,
                          help="pyttsx3 base rate in words per minute (default: 165)")
    speaking.add_argument("--max-wpm", type=int, default=280,
                          help="pyttsx3 ceiling in words per minute (default: 280)")
    speaking.add_argument("--max-speed", type=float, default=1.25,
                          help="cap on the last-resort safety atempo (1.0 = none, default: 1.25)")
    speaking.add_argument("--no-stretch", action="store_true",
                          help="disable the safety atempo entirely")
    speaking.add_argument("--stretch-tolerance", type=float, default=0.05,
                          help="a line within this fraction of its window counts as fitting (default: 0.05)")
    speaking.add_argument("--fit-attempts", type=int, default=3,
                          help="renders per line while fitting the rate (default: 3)")
    speaking.add_argument("--tts-retries", type=int, default=2,
                          help="attempts per line for an online engine (default: 2)")

    timeline = parser.add_argument_group("timeline")
    timeline.add_argument("--overflow", choices=list(OVERFLOWS), default="cut",
                          help="a line that still overruns: cut it, let it bleed past the cue, "
                               "or compress it to fit (default: cut)")
    timeline.add_argument("--max-compress", type=float, default=8.0,
                          help="speed-up ceiling for --overflow compress before it trims "
                               "(default: 8.0)")
    timeline.add_argument("--overlap", choices=list(OVERLAPS), default="mix",
                          help="overlapping cues: mix, trim the earlier one, or shift the later one")
    timeline.add_argument("--no-stitch", action="store_true",
                          help="do not merge sentences split across consecutive cues")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--no-normalize", action="store_true",
                       help="keep the raw subtitle text (no tag stripping or expansion)")
    audio.add_argument("--no-level", action="store_true",
                       help="skip per-line loudness levelling")
    audio.add_argument("--no-loudnorm", action="store_true",
                       help="skip the two-pass EBU R128 pass on the export")
    audio.add_argument("--music", help="background music file, looped and ducked under the voice")
    audio.add_argument("--music-volume", type=float, default=0.25,
                       help="music level before ducking, 0.0-1.0 (default: 0.25)")
    audio.add_argument("--lexicon", help="JSON file of pronunciation overrides {\"word\": \"replacement\"}")
    audio.add_argument("--say", action="append", metavar="WORD=REPLACEMENT",
                       help="pronunciation override, repeatable or comma separated")

    speed = parser.add_argument_group("performance")
    speed.add_argument("--concurrency", type=int,
                       help="lines to render in parallel (default: auto - 4 for a network "
                            "engine, 1 for CPU-bound engines like piper)")
    speed.add_argument("--cache-dir", help="render cache directory (default: ./.tts_cache)")
    speed.add_argument("--no-cache", action="store_true", help="disable the render cache")
    speed.add_argument("--clear-cache", action="store_true",
                       help="empty the render cache and exit")
    speed.add_argument("--fallback-after", type=int, default=1,
                       help="failures before an engine is dropped for the rest of the run (default: 1)")

    report = parser.add_argument_group("reporting")
    report.add_argument("--metrics", action="store_true",
                        help="measure speech-onset alignment in the final audio")
    report.add_argument("--report", dest="report_path", help="write the JSON quality report here")
    report.add_argument(
        "--strict", action="store_true",
        help=(f"exit {EXIT_STRICT_FAILED} if the run degraded (cuts, bleeds, engine "
              "fallback, wrong duration or poor alignment)"),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.clear_cache:
        removed = RenderCache(args.cache_dir).clear()
        print(f"Removed {removed} cached render(s)")
        return EXIT_OK

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is required (time compression, loudness, m4a export).")
        print("  Windows: winget install ffmpeg   |   macOS: brew install ffmpeg   |   "
              "Linux: sudo apt install ffmpeg")
        return EXIT_SETUP_ERROR

    if args.music and not os.path.exists(args.music):
        # Say so now, rather than letting ffmpeg discover it after every line is spoken.
        print(f"Warning: music file not found: {args.music}; continuing without a bed")
        args.music = None

    lexicon = {}
    if args.lexicon:
        lexicon.update(_load_lexicon_file(args.lexicon))
    lexicon.update(_parse_pairs(args.say, "--say"))
    speaker_voices = _parse_pairs(args.speaker_voices, "--speaker-voices")

    try:
        report = asyncio.run(_run(args, lexicon, speaker_voices))
    except KeyboardInterrupt:
        print("\nInterrupted; the output file was not written.")
        return EXIT_INTERRUPTED
    except Exception:
        # A CLI should end with a message and a code, not a traceback the user has to
        # scroll. The traceback still goes to stderr so a real bug stays reportable.
        traceback.print_exc()
        print("Error: the run failed unexpectedly (see the traceback above)", file=sys.stderr)
        return EXIT_SETUP_ERROR

    if report is None:
        return EXIT_SETUP_ERROR
    if not report.get("export_ok", True):
        return EXIT_SETUP_ERROR
    if args.strict and not report["strict"]["passed"]:
        print("STRICT GATE FAILED: " + "; ".join(report["strict"]["reasons"]))
        return EXIT_STRICT_FAILED
    return EXIT_OK


async def _run(args, lexicon, speaker_voices):
    """Build the call from the parsed arguments and hand it to the pipeline."""
    return await generate_audio(
        path=args.path,
        voice=None if args.auto_voice else args.voice,
        gender=args.gender,
        output_format=args.format,
        base_rate_percent=args.base_rate,
        max_rate_percent=args.max_rate,
        pyttsx3_rate_wpm=args.pyttsx3_rate,
        max_pyttsx3_wpm=args.max_wpm,
        max_speed=args.max_speed,
        allow_stretch=not args.no_stretch,
        tolerance=args.stretch_tolerance,
        fit_attempts=args.fit_attempts,
        engine=args.engine,
        engine_retries=args.tts_retries,
        piper_model=find_piper_model(args.piper_model),
        edge_proxy=args.edge_proxy,
        piper_noise=args.piper_noise,
        piper_noise_w=args.piper_noise_w,
        deterministic=args.deterministic,
        cache=not args.no_cache,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
        fallback_after=args.fallback_after,
        overflow=args.overflow,
        max_compress=args.max_compress,
        stitch=not args.no_stitch,
        overlap=args.overlap,
        normalize=not args.no_normalize,
        level=not args.no_level,
        loudnorm=not args.no_loudnorm,
        music=args.music,
        music_volume=args.music_volume,
        metrics=args.metrics,
        report_path=args.report_path,
        lexicon=lexicon or None,
        speaker_voices=speaker_voices or None,
        auto_speakers=args.auto_speakers,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())
