"""Pipeline orchestration.

Reads subtitles, groups them into render units, renders every unit through the TTS
engine chain (concurrently, with a persistent cache), places the results on the
subtitle timeline, exports the mix and reports what happened.

Rendering is concurrent; *placement and logging are strictly sequential*, so the
output timeline and the log never depend on which render happened to finish first.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from audio_post import (
    ATEMPO_MAX_RATIO,
    FORMATS,
    LOUDNESS_TARGET,
    compress_to_fit,
    export_audio,
    is_silent,
    level_segment,
    stretch_audio,
    trim_trailing_silence,
)
from pipeline_core import (
    CUT_EPSILON_MS,
    apply_overlap_policy,
    build_units,
    decide_overflow,
    format_timestamp,
    solve_rate,
)
from subtitle_io import apply_lexicon, load_subtitles
from tts_engines import (
    EDGE_VOICE_DEFAULT,
    PIPER_DETERMINISTIC_NOISE_W,
    RenderCache,
    build_chain,
    describe_engines,
    detect_language,
    filter_reachable,
    find_piper_models,
    first_guess_speed,
    load_rendered,
    parse_rate_pct,
    pick_auto_voice,
)

__all__ = ["generate_audio", "measure_alignment", "AUTO_SPEAKER_POOL", "POSITION_TOLERANCE_MS"]

# Voices to rotate through when speaker labels are detected and no explicit map is
# given. piper is one model per file, so its pool comes from --speaker-voices.
AUTO_SPEAKER_POOL = {
    "edge_tts": ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-BrianNeural"],
    "piper": [],
    "pyttsx3": [],
}

POSITION_TOLERANCE_MS = 150


def measure_alignment(output: AudioSegment, records, search_ms: int = 250,
                      tolerance_ms: int = POSITION_TOLERANCE_MS) -> dict:
    """Measure how well speech lands on the cue starts, honestly.

    Each unit looks for a speech onset within +/- `search_ms` of its start. Units
    with no onset nearby are counted as *missing* and excluded from the offset
    statistics rather than being matched to a distant neighbour, which is what made
    the old number flattering and useless.
    """
    onsets = [region[0] for region in detect_nonsilent(output, min_silence_len=200, silence_thresh=-45)]
    offsets, missing = [], 0
    for record in records:
        start = record["start_ms"]
        near = [onset for onset in onsets if abs(onset - start) <= search_ms]
        if near:
            offsets.append(min(near, key=lambda onset: abs(onset - start)) - start)
        else:
            missing += 1

    result = {
        "units_checked": len(records),
        "onsets_matched": len(offsets),
        "onsets_missing": missing,
        "search_window_ms": search_ms,
        "tolerance_ms": tolerance_ms,
        "within_150ms_pct": None,
        "median_abs_offset_ms": None,
        "mean_abs_offset_ms": None,
        "max_abs_offset_ms": None,
    }
    if offsets:
        abs_offsets = sorted(abs(o) for o in offsets)
        result.update({
            "within_150ms_pct": round(
                100.0 * sum(1 for o in abs_offsets if o <= tolerance_ms) / len(abs_offsets), 1),
            "median_abs_offset_ms": abs_offsets[len(abs_offsets) // 2],
            "mean_abs_offset_ms": round(sum(abs_offsets) / len(abs_offsets), 1),
            "max_abs_offset_ms": max(abs_offsets),
        })
    return result


def _speaker_voice_map(units, engine_name: str, default_voice: str, speaker_voices: dict,
                       auto_speakers: bool, piper_models=()) -> dict:
    """Assign a voice to every speaker label, in order of first appearance."""
    order = []
    for unit in units:
        if unit.speaker and unit.speaker not in order:
            order.append(unit.speaker)
    if not order:
        return {}

    pool = list(AUTO_SPEAKER_POOL.get(engine_name) or [])
    if engine_name == "piper":
        # Every installed voice model is a distinct offline voice.
        pool = list(piper_models)
    elif engine_name == "pyttsx3":
        pool = ["__gender__"]        # pyttsx3 picks its voice from the gender hint
    pool = pool or [default_voice or None]

    mapping = {}
    for index, name in enumerate(order):
        if speaker_voices and name in speaker_voices:
            mapping[name] = speaker_voices[name]
        elif auto_speakers:
            mapping[name] = pool[index % len(pool)]
        else:
            mapping[name] = default_voice or None
    if auto_speakers and engine_name == "piper" and len(piper_models) < 2 and not speaker_voices:
        print("  note: only one piper voice model is installed, so every speaker shares it; "
              "add more models (scripts/download_voice.py en_US-ryan-medium) or pass "
              "--speaker-voices NAME=model.onnx")
    return mapping


async def _fit_with_engine(engine, unit, voice: str, *, tolerance: float, fit_attempts: int,
                           cache: RenderCache, tmpdir: str, index: int):
    """Render one unit with one engine, fitting the speaking rate to its window.

    Returns a dict with the chosen audio, or None if every attempt failed. Cache
    hits still measure the real audio, so a cached run is identical to a fresh one.
    """
    text = unit.speak or unit.text
    window_ms = unit.window_ms
    if not text or window_ms <= 0:
        return None

    speed = first_guess_speed(engine, text, window_ms)
    records = []
    cached_hit = False

    variant = engine.cache_variant()
    for attempt in range(max(1, int(fit_attempts))):
        key = cache.key(engine.name, voice, speed, text, variant)
        suffix = ".mp3" if engine.name == "edge_tts" else ".wav"
        path = cache.get(key)
        if path is not None:
            cached_hit = True
        else:
            cache.record_miss()
            out_path = os.path.join(tmpdir, f"u{index:05d}_a{attempt}{suffix}")
            if not await engine.render_async(text, voice, speed, out_path):
                return None
            path = cache.put(key, out_path)

        try:
            audio = trim_trailing_silence(load_rendered(path))
        except Exception as e:
            if not cached_hit:
                print(f"  failed to read rendered audio: {e}")
                return None
            # A corrupt cache entry must never fail a line: drop it and render for real.
            print(f"  cached render unreadable ({e}); rendering it again")
            cache.invalidate(key)
            cache.record_miss()
            out_path = os.path.join(tmpdir, f"u{index:05d}_a{attempt}_refetch{suffix}")
            if not await engine.render_async(text, voice, speed, out_path):
                return None
            path = cache.put(key, out_path)
            cached_hit = False
            try:
                audio = trim_trailing_silence(load_rendered(path))
            except Exception as e2:
                print(f"  failed to read rendered audio: {e2}")
                return None

        records.append((speed, len(audio), audio))
        nxt = solve_rate([(r[0], r[1]) for r in records], window_ms, tolerance, engine.max_speed)
        if nxt is None:
            break
        speed = nxt

    fit_threshold = window_ms * (1.0 + max(0.0, tolerance))
    fitting = [r for r in records if r[1] <= fit_threshold]
    chosen = min(fitting, key=lambda r: r[0]) if fitting else max(records, key=lambda r: r[0])
    return {
        "audio": chosen[2],
        "speed": chosen[0],
        "renders": len(records),
        "engine": engine.name,
        "voice": voice,
        "cached": cached_hit,
    }


async def generate_audio(
    path: str,
    *,
    preview: int = None,
    voice: str = None,
    gender: str = "female",
    output_format: str = "m4a",
    base_rate_percent: str = "+0%",
    max_rate_percent: int = 30,
    pyttsx3_rate_wpm: int = 165,
    max_pyttsx3_wpm: int = 280,
    max_speed: float = 1.25,
    allow_stretch: bool = True,
    tolerance: float = 0.05,
    fit_attempts: int = 3,
    engine: str = "auto",
    engine_retries: int = 2,
    piper_model: str = None,
    edge_proxy: str = None,
    piper_noise: float = None,
    piper_noise_w: float = None,
    deterministic: bool = False,
    cache: bool = True,
    cache_dir: str = None,
    concurrency: int = None,
    fallback_after: int = 1,
    overflow: str = "cut",
    max_compress: float = ATEMPO_MAX_RATIO,
    stitch: bool = True,
    overlap: str = "mix",
    normalize: bool = True,
    level: bool = True,
    loudnorm: bool = True,
    music: str = None,
    music_volume: float = 0.25,
    metrics: bool = False,
    report_path: str = None,
    lexicon: dict = None,
    speaker_voices: dict = None,
    auto_speakers: bool = False,
    strict: bool = False,
) -> dict | None:
    """Render `path` to audio in `output_format`; returns the quality report."""
    base_rate_pct = parse_rate_pct(base_rate_percent)
    base_speed = max(0.1, 1.0 + base_rate_pct / 100.0)
    speed_ceiling = max(base_speed, 1.0 + max(max_rate_percent, base_rate_pct) / 100.0)

    print(f"Generating audio for {path}")
    print(
        f"Rate fitting: base {base_rate_percent}  |  "
        f"safety atempo: {'off' if not allow_stretch else f'max {max_speed:.2f}x'}"
    )
    print(
        f"Overflow: {overflow}  |  overlap: {overlap}  |  "
        f"stitching: {'on' if stitch else 'off'}  |  "
        f"text normalization: {'on' if normalize else 'off'}  |  "
        f"leveling: {'on' if level else 'off'}"
    )
    print(
        f"Loudness: {'EBU R128 two-pass' if loudnorm else 'off'}"
        + (f"  |  music bed: {music} @ {music_volume:.2f}" if music else "")
    )
    # --deterministic turns piper's duration noise off, which is the only thing that
    # makes a re-run of a stochastic engine come out the same. It costs the rhythm
    # variation that noise provides, so it is a choice, not a default.
    if deterministic and piper_noise_w is None:
        piper_noise_w = PIPER_DETERMINISTIC_NOISE_W
    byte_identical = piper_noise_w == PIPER_DETERMINISTIC_NOISE_W and piper_noise == 0.0
    print("Reproducibility: " + (
        "byte-identical (piper noise_w 0, noise 0)" if byte_identical
        else "repeatable timing (piper noise_w 0)" if piper_noise_w == PIPER_DETERMINISTIC_NOISE_W
        else "off (piper's rhythm noise is on, so line lengths vary by a few percent)"))

    try:
        loaded = load_subtitles(path, normalize=normalize)
    except FileNotFoundError:
        print(f"No such subtitle file: {path}")
        return None
    except ValueError as e:
        print(f"Cannot read subtitles: {e}")
        return None
    except OSError as e:
        # Permission denied, a directory, a share that went away: still a clean exit.
        print(f"Cannot read {path}: {e}")
        return None

    preview_cues = None
    if preview:
        preview_cues = max(1, int(preview))
        loaded.cues = loaded.cues[:preview_cues]
        print(f"Preview: speaking the first {len(loaded.cues)} cue(s) so this can be checked "
              f"in seconds")

    if lexicon:
        for cue in loaded.cues:
            cue.text = apply_lexicon(cue.text, lexicon)
        print(f"Pronunciation overrides applied: {len(lexicon)} entries")

    total_duration_ms = loaded.total_duration_ms
    print(f"Found {len(loaded.cues)} subtitle cues read as .{loaded.fmt} ({loaded.encoding})")
    for note in loaded.repairs:
        print(f"  repaired: {note}")

    # Pick the offline voice by the subtitles' writing system, so a voice installed for
    # one language cannot silently read another language's file.
    lang_hint = detect_language(" ".join(cue.text for cue in loaded.cues))
    piper_models = find_piper_models(piper_model, prefer_lang=lang_hint)
    default_piper = piper_models[0] if piper_models else piper_model
    if default_piper and len(piper_models) > 1:
        why = f"matches the {lang_hint} text" if lang_hint else "Latin text prefers English"
        print(f"Piper voice: {os.path.basename(default_piper)} ({why}; "
              f"{len(piper_models)} installed)")
    try:
        chain = build_chain(
            engine,
            base_speed=base_speed,
            max_speed=speed_ceiling,
            edge_retries=engine_retries,
            piper_model=default_piper,
            base_wpm=pyttsx3_rate_wpm,
            max_wpm=max_pyttsx3_wpm,
            edge_proxy=edge_proxy,
            piper_noise=piper_noise,
            piper_noise_w=piper_noise_w,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Cannot start a TTS engine: {e}")
        return None
    print(f"Engines, best first: {describe_engines(chain)}")

    # An imported online engine is not necessarily a reachable one: prove it once
    # before the run, so a blocked endpoint costs nothing and is not mistaken for
    # a quality regression later.
    probe_dir = tempfile.mkdtemp(prefix="subtitle_tts_probe_")
    try:
        chain = await filter_reachable(chain, probe_dir)
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    print(f"Engines in use: {describe_engines(chain)}")
    print("Rate ceilings: " + ", ".join(
        f"{engine.name} {engine.max_speed:.2f}x" for engine in chain)
        + f"  (--max-rate {max_rate_percent:+d}% applies to edge_tts)")

    # Parallelism only pays off for engines that wait on the network. piper is CPU
    # bound and onnxruntime already uses every core (measured: 4 and 8 concurrent
    # renders are no faster, and 8 is slower), so it defaults to serial.
    if concurrency:
        effective_concurrency = max(1, int(concurrency))
    else:
        effective_concurrency = 4 if chain[0].needs_network else 1
    print(f"Rendering: {effective_concurrency} line(s) in parallel")

    if voice is None:
        if chain and chain[0].name == "edge_tts":
            joined = " ".join(cue.text for cue in loaded.cues)
            voice = pick_auto_voice(joined, gender)
            print(f"Auto-picked edge_tts voice: {voice}")
        else:
            voice = EDGE_VOICE_DEFAULT

    known_speakers = {name.upper() for name in speaker_voices} if speaker_voices else None
    units = build_units(loaded.cues, stitch=stitch, known_speakers=known_speakers,
                        auto_speakers=auto_speakers)
    units, overlap_tags = apply_overlap_policy(units, overlap, total_duration_ms)
    if not units:
        print("Nothing to speak: these subtitles produced no render units")
        return None
    stitched_groups = sum(1 for u in units if u.stitched)
    if stitched_groups:
        print(f"Stitched {stitched_groups} multi-cue sentence(s) into a single phrase")
    print(f"Render units: {len(units)}")

    voices = _speaker_voice_map(units, chain[0].name, voice, speaker_voices, auto_speakers,
                                piper_models=piper_models or ())
    piper_model = default_piper or piper_model
    for name, hint in voices.items():
        print(f"  speaker {name} -> {hint or chain[0].name + ' default'}")
    if speaker_voices:
        unknown = sorted(set(speaker_voices) - set(voices))
        if unknown:
            print(f"  note: --speaker-voices names never seen in the subtitles: {', '.join(unknown)}")

    print(f"Target total duration: {total_duration_ms / 1000:.1f}s "
          f"({total_duration_ms // 60000}m {(total_duration_ms % 60000) // 1000}s)")

    cache_store = RenderCache(cache_dir, enabled=cache)
    if cache_store.enabled:
        print(f"Render cache: {cache_store.directory}")
    disabled: set[str] = set()
    failures: dict[str, int] = {e.name: 0 for e in chain}
    fallback_used: set[str] = set()
    semaphore = asyncio.Semaphore(effective_concurrency)

    async def render_unit(index: int, unit):
        hint = voices.get(unit.speaker) if unit.speaker else voice
        if hint == "__gender__":
            hint = None          # pyttsx3 picks its voice from the gender hint alone
        async with semaphore:
            try:
                for engine_obj in chain:
                    if engine_obj.name in disabled:
                        continue
                    voice_id = engine_obj.resolve_voice(hint, gender)
                    result = await _fit_with_engine(
                        engine_obj, unit, voice_id,
                        tolerance=tolerance, fit_attempts=fit_attempts,
                        cache=cache_store, tmpdir=tmpdir, index=index,
                    )
                    if result is not None:
                        failures[engine_obj.name] = 0
                        if engine_obj is not chain[0]:
                            fallback_used.add(engine_obj.name)
                        result["unit"] = index
                        return result
                    failures[engine_obj.name] += 1
                    if failures[engine_obj.name] >= max(1, fallback_after) and len(chain) > 1:
                        if engine_obj.name not in disabled:
                            disabled.add(engine_obj.name)
                            remaining = [e.name for e in chain if e.name not in disabled]
                            print(f"  {engine_obj.name} failed {failures[engine_obj.name]}x; "
                                  f"falling back to {remaining[0] if remaining else 'nothing'}")
                return None
            except Exception as e:
                # One bad line must never take the run down with it.
                print(f"  unit {index + 1} failed unexpectedly: {type(e).__name__}: {str(e)[:160]}")
                return None

    with tempfile.TemporaryDirectory() as tmpdir:
        gathered = await asyncio.gather(
            *(render_unit(i, u) for i, u in enumerate(units)), return_exceptions=True)
        results = []
        for index, outcome in enumerate(gathered):
            if isinstance(outcome, BaseException):
                print(f"  unit {index + 1} failed unexpectedly: "
                      f"{type(outcome).__name__}: {str(outcome)[:160]}")
                results.append(None)
            else:
                results.append(outcome)

        # ---- placement: strictly sequential, in subtitle order ----
        output = AudioSegment.silent(duration=max(1, total_duration_ms))
        records = []
        for i, unit in enumerate(units):
            result = results[i]
            if result is None:
                print(f"  [{i + 1}/{len(units)}] TTS failed on every engine, skipping unit")
                continue

            start_ms = unit.start_ms
            window_ms = unit.window_ms
            audio = result["audio"]
            engine_name = result["engine"]
            voice_id = result["voice"]
            if level:
                audio = level_segment(audio)

            # An engine that cannot speak the text (wrong-language voice, or SAPI with
            # that language absent) returns a valid but silent file. Say so instead of
            # silently placing nothing on the timeline.
            silent = is_silent(audio)
            if silent:
                print(f"  [{i + 1}/{len(units)}] the engine returned silence for this line "
                      f"(voice cannot speak this text?)")

            speech_ms = len(audio)
            safety_applied = False
            if speech_ms > window_ms and allow_stretch and max_speed is not None:
                stretched = stretch_audio(audio, window_ms, max_speed=max_speed)
                safety_applied = len(stretched) < speech_ms
                audio = stretched
                speech_ms = len(audio)

            clipped = bled = overlap_trimmed = 0
            compress_ratio = None
            action = decide_overflow(speech_ms - window_ms, overflow)
            if action == "compress":
                compressed = compress_to_fit(audio, window_ms, max_ratio=max_compress)
                if 0 < len(compressed) < speech_ms:
                    compress_ratio = speech_ms / len(compressed)
                audio = compressed
                speech_ms = len(audio)
                if speech_ms - window_ms > CUT_EPSILON_MS:
                    clipped = int(round(speech_ms - window_ms))
                    audio, speech_ms = audio[:window_ms], window_ms
            elif action == "cut":
                clipped = int(round(speech_ms - window_ms))
                audio, speech_ms = audio[:window_ms], window_ms
            elif action == "bleed":
                bled = int(round(speech_ms - window_ms))
            elif speech_ms < window_ms:
                audio = audio + AudioSegment.silent(duration=int(window_ms - speech_ms))
                speech_ms = len(audio)

            if unit.limit_ms is not None and speech_ms > unit.limit_ms + CUT_EPSILON_MS:
                overlap_trimmed = int(round(speech_ms - unit.limit_ms))
                audio, speech_ms = audio[:unit.limit_ms], unit.limit_ms

            output = output.overlay(audio, position=int(start_ms))

            fit_ratio = (speech_ms / window_ms) if window_ms else 0.0
            tags = []
            if silent:
                tags.append("SILENT")
            if unit.stitched:
                tags.append(f"STITCH({len(unit.cues)})")
            if unit.speaker:
                tags.append(f"SPEAKER({unit.speaker})")
            if result["renders"] > 1:
                tags.append("rate-fit")
            if safety_applied:
                tags.append("safety-atempo")
            if compress_ratio:
                tags.append(f"COMPRESS(x{compress_ratio:.2f})")
            if bled:
                tags.append(f"BLEED(+{bled}ms)")
            if clipped:
                tags.append(f"TRUNCATED(-{clipped}ms)")
            if overlap_trimmed:
                tags.append(f"OVERLAP-TRIM(-{overlap_trimmed}ms)")
            if i in overlap_tags and not (overlap_trimmed or bled or clipped):
                tags.append(overlap_tags[i])
            if result["cached"]:
                tags.append("cached")
            tag = (" " + " ".join(tags)) if tags else ""

            rate_label = f"{result['speed']:.2f}x"
            print(
                f"  [{i + 1}/{len(units)}] {format_timestamp(start_ms)} -> "
                f"{format_timestamp(unit.end_ms)} | {engine_name:9s} {rate_label:>7s} | "
                f"fit {fit_ratio:.2f}x{tag:18s} | \"{(unit.speak or unit.text)[:55]}\""
            )

            records.append({
                "unit": i + 1,
                "cues": [c.index for c in unit.cues],
                "speaker": unit.speaker,
                "start_ms": int(start_ms),
                "end_ms": int(unit.end_ms),
                "window_ms": int(window_ms),
                "engine": engine_name,
                "voice": voice_id,
                "speed": round(result["speed"], 4),
                "renders": result["renders"],
                "cached": result["cached"],
                "fit": round(fit_ratio, 4),
                "tags": tags,
                "silent": silent,
                "cut_ms": clipped,
                "bleed_ms": bled,
                "overlap_trim_ms": overlap_trimmed,
                "stitched": unit.stitched,
            })

    if len(output) < total_duration_ms:
        output = output + AudioSegment.silent(duration=total_duration_ms - len(output))

    stem = os.path.splitext(path)[0]
    output_path = (f"{stem}.preview{FORMATS[output_format]['ext']}" if preview_cues
                   else stem + FORMATS[output_format]["ext"])
    export_info = export_audio(
        output, output_path, output_format,
        loudnorm=loudnorm, music=music, music_volume=music_volume,
    )

    if export_info.get("ok") and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nAudio exported to: {output_path}")
        print(f"Format: {FORMATS[output_format]['description']}  |  File size: {size_mb:.1f} MB")
    else:
        # The report still matters here: it says what the run produced and why the
        # file is missing, instead of dying on a stat() of a file that is not there.
        print(f"\nError: the mix could not be written to {output_path}")
    print(f"Final duration: {len(output) / 1000:.1f}s "
          f"({len(output) // 60000}m {(len(output) % 60000) // 1000}s)")
    print(f"Target duration: {total_duration_ms / 1000:.1f}s "
          f"({total_duration_ms // 60000}m {(total_duration_ms % 60000) // 1000}s)")
    if export_info.get("loudnorm"):
        measured = export_info.get("measured") or {}
        detail = (f"measured {measured['input_i']:.1f} LUFS -> target {LOUDNESS_TARGET['I']:.0f} LUFS"
                  if measured else "single pass")
        print(f"Loudness: EBU R128 {'two-pass' if export_info.get('two_pass') else 'one-pass'} ({detail})")

    report = _build_report(
        loaded, units, records, output, total_duration_ms, output_path, output_format,
        chain=chain, cache_store=cache_store, export_info=export_info,
        voices=voices, lexicon=lexicon, fallback_used=fallback_used,
        overlap=overlap, overflow=overflow, stitch=stitch, metrics=metrics,
        strict=strict, music=music, preview_cues=preview_cues,
        piper_noise=piper_noise, piper_noise_w=piper_noise_w,
        byte_identical=byte_identical,
    )
    _print_report(report)
    if report_path:
        try:
            _write_report(report_path, report)
            print(f"  report              : {report_path}")
        except OSError as e:
            print(f"  (could not write report: {e})")
    print("=" * 64)
    return report


def _write_report(path: str, report: dict) -> None:
    """Write the JSON report through a staging file + rename.

    A report is a result someone may diff or gate on, so a crash mid-write must not
    leave a truncated JSON file that looks like a real one.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    staging = f"{path}.{os.getpid()}.part"
    try:
        with open(staging, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            with contextlib.suppress(OSError):
                os.remove(staging)


def _build_report(loaded, units, records, output, total_duration_ms, output_path, output_format,
                  *, chain, cache_store, export_info, voices, lexicon, fallback_used,
                  overlap, overflow, stitch, metrics, strict, music, preview_cues=None,
                  piper_noise=None, piper_noise_w=None, byte_identical=False) -> dict:
    fits = [r["fit"] for r in records] or [0.0]
    cuts = [(r["unit"], r["cut_ms"]) for r in records if r["cut_ms"]]
    bleeds = [(r["unit"], r["bleed_ms"]) for r in records if r["bleed_ms"]]
    overlap_trims = [(r["unit"], r["overlap_trim_ms"]) for r in records if r["overlap_trim_ms"]]
    silent_units = [(r["unit"], r["voice"]) for r in records if r["silent"]]
    duration_delta = len(output) - total_duration_ms

    report = {
        "input": os.path.basename(loaded.path or ""),
        "format_in": loaded.fmt,
        "encoding": loaded.encoding,
        "cues": len(loaded.cues),
        "repaired": loaded.repairs,
        "output": output_path,
        "format_out": output_format,
        "export_ok": bool(export_info.get("ok")),
        "target_ms": total_duration_ms,
        "final_ms": len(output),
        "duration_delta_ms": duration_delta,
        "units": len(records),
        "stitched_units": sum(1 for r in records if r["stitched"]),
        "renders": sum(r["renders"] for r in records),
        "rate_fitted": sum(1 for r in records if "rate-fit" in r["tags"]),
        "safety_atempo": sum(1 for r in records if "safety-atempo" in r["tags"]),
        "compressed": sum(1 for r in records if any(t.startswith("COMPRESS") for t in r["tags"])),
        "silent_units": {"count": len(silent_units), "units": silent_units},
        "cuts": {"count": len(cuts), "total_ms": sum(c for _, c in cuts), "units": cuts},
        "bleeds": {"count": len(bleeds), "total_ms": sum(b for _, b in bleeds), "units": bleeds},
        "overlap_trims": {"count": len(overlap_trims), "units": overlap_trims},
        "fit": {"min": min(fits), "mean": round(sum(fits) / len(fits), 4), "max": max(fits)},
        "engines": [e.name for e in chain],
        "fallback_engines": sorted(fallback_used),
        "speaker_voices": voices,
        "lexicon_entries": len(lexicon or {}),
        "cache": cache_store.stats,
        "loudness": {
            "enabled": bool(export_info.get("loudnorm")),
            "two_pass": bool(export_info.get("two_pass")),
            "measured": export_info.get("measured"),
        },
        "music": music,
        "policies": {"overflow": overflow, "overlap": overlap, "stitch": stitch},
        "preview_cues": preview_cues,
        "piper": {"noise": piper_noise, "noise_w": piper_noise_w,
                  "deterministic": piper_noise_w == PIPER_DETERMINISTIC_NOISE_W,
                  "byte_identical": byte_identical},
        "alignment": None,
        "sources": {
            name: sum(1 for r in records if r["engine"] == name)
            for name in ("edge_tts", "piper", "pyttsx3")
        },
        "units_detail": records,
    }

    if metrics:
        report["alignment"] = measure_alignment(output, records)

    reasons = []
    if report["cuts"]["count"]:
        reasons.append(f"{report['cuts']['count']} material cut(s)")
    if report["bleeds"]["count"]:
        reasons.append(f"{report['bleeds']['count']} line(s) bled past their cue")
    if fallback_used:
        reasons.append(f"degraded to {', '.join(sorted(fallback_used))}")
    if abs(duration_delta) > 60:
        reasons.append(f"duration off by {duration_delta} ms")
    if len(records) != len(units):
        reasons.append(f"{len(units) - len(records)} unit(s) produced no audio")
    if silent_units:
        reasons.append(f"{len(silent_units)} unit(s) came back silent "
                       f"(the voice cannot speak that text?)")
    if not report["export_ok"]:
        reasons.append(f"the mix could not be written to {output_path}")
    alignment = report.get("alignment")
    if alignment and alignment.get("within_150ms_pct") is not None and alignment["within_150ms_pct"] < 90:
        reasons.append(f"alignment {alignment['within_150ms_pct']}% within 150 ms")
    report["strict"] = {"requested": strict, "passed": not reasons, "reasons": reasons}
    return report


def _print_report(report: dict) -> None:
    fits = report["fit"]
    print("\n" + "=" * 64)
    print("QUALITY REPORT")
    print(f"  render units        : {report['units']} ({report['stitched_units']} stitched from several cues)")
    print(f"  cues covered        : {sum(len(r['cues']) for r in report['units_detail'])} of {report['cues']}")
    print(f"  engines             : {', '.join(report['engines'])}")
    if report["fallback_engines"]:
        print(f"  fell back to        : {', '.join(report['fallback_engines'])}")
    print(
        f"  TTS renders         : {report['renders']} "
        f"({report['rate_fitted']} rate-fitted, {report['safety_atempo']} safety-atempo, "
        f"{report['compressed']} compressed)"
    )
    cache = report["cache"]
    if cache.get("enabled"):
        print(f"  cache               : {cache['hits']} hit / {cache['misses']} miss")
    print(f"  fit ratio           : min {fits['min']:.2f}x  mean {fits['mean']:.2f}x  max {fits['max']:.2f}x")
    print(f"  material cuts       : {report['cuts']['count']}"
          + (f" ({report['cuts']['total_ms']} ms total)" if report["cuts"]["count"] else ""))
    if report["bleeds"]["count"]:
        print(f"  bleeding into next  : {report['bleeds']['count']} ({report['bleeds']['total_ms']} ms total)")
    if report["overlap_trims"]["count"]:
        print(f"  overlap trims       : {report['overlap_trims']['count']}")
    if report["silent_units"]["count"]:
        print(f"  silent renders      : {report['silent_units']['count']} "
              f"(check the voice can speak this language)")
    if report["speaker_voices"]:
        print(f"  speaker voices      : {len(report['speaker_voices'])}")
    if report["lexicon_entries"]:
        print(f"  pronunciation rules : {report['lexicon_entries']}")
    piper = report.get("piper") or {}
    if piper.get("byte_identical"):
        print("  reproducibility     : byte-identical renders")
    elif piper.get("deterministic"):
        print("  reproducibility     : repeatable timing (piper noise_w 0)")
    if report["loudness"]["enabled"]:
        print(f"  loudness            : EBU R128 {'two-pass' if report['loudness']['two_pass'] else 'one-pass'}")
    alignment = report.get("alignment")
    if alignment:
        if alignment["onsets_matched"]:
            print(
                f"  alignment           : {alignment['within_150ms_pct']}% within 150 ms "
                f"(median {alignment['median_abs_offset_ms']} ms / max {alignment['max_abs_offset_ms']} ms "
                f"over {alignment['onsets_matched']} units with a detected onset)"
            )
        if alignment["onsets_missing"]:
            print(f"  no onset detected   : {alignment['onsets_missing']} units "
                  f"(short/quiet lines; excluded from the offsets above)")
    print(f"  final vs target     : {report['final_ms']} ms vs {report['target_ms']} ms")
    if report["repaired"]:
        print(f"  input repairs       : {len(report['repaired'])}")
    if report["strict"]["requested"]:
        verdict = "PASSED" if report["strict"]["passed"] else "FAILED"
        print(f"  strict gate         : {verdict}"
              + ("" if report["strict"]["passed"] else f" ({'; '.join(report['strict']['reasons'])})"))
