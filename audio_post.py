"""Audio post-processing: time compression, levelling, loudness, music bed, export.

Everything here works on pydub segments or on WAVs on disk, and every ffmpeg call
degrades to a safe no-op rather than failing a run.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import tempfile

from pydub import AudioSegment

__all__ = [
    "trim_trailing_silence",
    "is_silent",
    "stretch_audio",
    "compress_to_fit",
    "level_segment",
    "measure_loudness",
    "mix_music",
    "export_audio",
    "ENCODER_ARGS",
    "FORMATS",
    "TARGET_SEGMENT_DBFS",
    "MAX_LEVEL_GAIN_DB",
    "PEAK_CEILING_DBFS",
    "ATEMPO_MAX_RATIO",
    "SILENCE_FLOOR_DBFS",
    "LOUDNESS_TARGET",
]

# ------------------------------------------------------------------ constants

TARGET_SEGMENT_DBFS = -20.0      # every line is matched to this RMS
MAX_LEVEL_GAIN_DB = 12.0         # never lift a near-silent artifact into noise
PEAK_CEILING_DBFS = -1.0         # levelling must leave this much peak headroom
ATEMPO_MAX_RATIO = 8.0           # bound for --overflow compress
SILENCE_FLOOR_DBFS = -60.0       # below this there is no speech worth placing

# EBU R128 target for the final mix. Two-pass is used so the result actually
# lands on the target instead of "close enough" (single-pass loudnorm adapts
# dynamically and drifts on a mix that is mostly silence).
LOUDNESS_TARGET = {"I": -16.0, "TP": -1.5, "LRA": 11.0}

# loudnorm resamples internally to 192 kHz, so the encoder is pinned to a sane
# delivery rate -- without this the exported file inherits 192 kHz and quadruples
# in size for no audible gain on speech.
OUTPUT_SAMPLE_RATE = "48000"

ENCODER_ARGS = {
    "m4a": ["-c:a", "aac", "-b:a", "192k", "-ar", OUTPUT_SAMPLE_RATE, "-movflags", "+faststart"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k", "-ar", OUTPUT_SAMPLE_RATE],
    "ogg": ["-c:a", "libvorbis", "-q:a", "5", "-ar", OUTPUT_SAMPLE_RATE],
    "wav": ["-c:a", "pcm_s16le", "-ar", OUTPUT_SAMPLE_RATE],
}

FORMATS = {
    "wav": {"ext": ".wav", "pydub_format": "wav", "description": "WAV (uncompressed)"},
    "mp3": {"ext": ".mp3", "pydub_format": "mp3", "description": "MP3"},
    "m4a": {"ext": ".m4a", "pydub_format": "ipod", "description": "AAC/M4A (better than MP3)"},
    "ogg": {"ext": ".ogg", "pydub_format": "ogg", "description": "OGG Vorbis"},
}


def _ffmpeg_error(exc) -> str:
    detail = getattr(exc, "stderr", None)
    return str(detail)[-200:] if detail else str(exc)[:200]


# ------------------------------------------------------------ time compression

def trim_trailing_silence(segment: AudioSegment) -> AudioSegment:
    """Drop leading/trailing silence. TTS clips tail off with a long pause;
    removing it shortens the clip with zero audible artifacts (unlike atempo)."""
    try:
        return segment.strip_silence(silence_len=200, silence_thresh=-45, padding=40)
    except Exception:
        return segment


def _apply_atempo(audio_segment: AudioSegment, ratio: float) -> AudioSegment:
    """Pitch-preserving compression by `ratio` (> 1) through ffmpeg atempo."""
    # atempo only accepts 0.5..100.0, so chain filters for larger factors.
    filters = []
    remaining = ratio
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")

    tmp_in_path = None
    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + "_stretched.wav"

        audio_segment.export(tmp_in_path, format="wav")
        cmd = ["ffmpeg", "-y", "-i", tmp_in_path, "-filter:a", ",".join(filters), tmp_out_path]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        return AudioSegment.from_wav(tmp_out_path)
    except subprocess.CalledProcessError as e:
        print(f"  Warning: ffmpeg atempo failed: {_ffmpeg_error(e)}")
        return audio_segment
    except FileNotFoundError:
        print("  Warning: ffmpeg not found, cannot time-compress audio")
        return audio_segment
    finally:
        for f in (tmp_in_path, tmp_out_path):
            if f and os.path.exists(f):
                os.unlink(f)


def stretch_audio(audio_segment: AudioSegment, target_duration_ms, max_speed: float = 1.25):
    """Bounded, pitch-preserving compression via ffmpeg atempo.

    Only speeds UP (never slows down) and never exceeds `max_speed`, so this is a
    small safety nudge on top of rate fitting -- not the primary mechanism it used
    to be. `max_speed=None` disables it entirely.
    """
    current_duration = len(audio_segment)
    if current_duration == 0 or target_duration_ms <= 0:
        return audio_segment
    if current_duration <= target_duration_ms or max_speed is None:
        return audio_segment

    ratio = min(current_duration / target_duration_ms, float(max_speed))
    if ratio <= 1.01:
        return audio_segment
    return _apply_atempo(audio_segment, ratio)


def compress_to_fit(audio_segment: AudioSegment, target_duration_ms, max_ratio: float = ATEMPO_MAX_RATIO):
    """Force speech into its window however fast that takes (`--overflow compress`).

    Unlike `stretch_audio` this is not a nudge: it is the "make it fit" option, so
    the only bound is `max_ratio` (a runaway cue cannot spin the atempo chain
    forever). Anything still overlong afterwards is trimmed and reported.
    """
    current_duration = len(audio_segment)
    if current_duration == 0 or target_duration_ms <= 0:
        return audio_segment
    if current_duration <= target_duration_ms:
        return audio_segment

    ratio = min(current_duration / target_duration_ms, float(max_ratio))
    if ratio <= 1.01:
        return audio_segment
    return _apply_atempo(audio_segment, ratio)


# ------------------------------------------------------------------- levelling

def is_silent(segment: AudioSegment) -> bool:
    """True when a render carries no speech at all.

    An engine asked to speak a language it does not have (an English voice reading
    Arabic, a SAPI voice with no such voice installed) can return a perfectly valid
    but silent file. That must be reported as a failure to speak, not passed off as
    audio that happens to sit in the timeline.
    """
    try:
        if len(segment) == 0:
            return True
        dbfs = segment.dBFS
        return dbfs == float("-inf") or dbfs < SILENCE_FLOOR_DBFS
    except Exception:
        return False


def level_segment(segment: AudioSegment) -> AudioSegment:
    """Even out per-line loudness so every cue sits at the same level.

    Matches RMS (dBFS) rather than peak: matching peaks makes a quiet-but-clipped
    line jump, which is what makes a mix of engines and lines sound uneven. The
    gain is capped so a near-silent artifact cannot be amplified into noise, and
    it is pulled back further if it would push the peak through the ceiling.
    """
    try:
        if segment.duration_seconds == 0:
            return segment
        dbfs = segment.dBFS
        if dbfs == float("-inf"):
            return segment

        gain = max(-MAX_LEVEL_GAIN_DB, min(MAX_LEVEL_GAIN_DB, TARGET_SEGMENT_DBFS - dbfs))
        # Keep peak headroom, so placing many lines on one track cannot clip.
        peak = segment.max_dBFS
        if peak != float("-inf"):
            gain = min(gain, PEAK_CEILING_DBFS - peak)
        if abs(gain) < 0.1:
            return segment
        return segment.apply_gain(gain)
    except Exception:
        return segment


# -------------------------------------------------------------------- loudness

_LOUDNORM_RE = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def measure_loudness(path: str) -> dict | None:
    """First pass of two-pass loudnorm: measure the file's loudness statistics."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", path,
        "-af", f"loudnorm=I={LOUDNESS_TARGET['I']}:TP={LOUDNESS_TARGET['TP']}:"
               f"LRA={LOUDNESS_TARGET['LRA']}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return None
    match = _LOUDNORM_RE.search(proc.stderr or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        measured = {
            "input_i": float(data["input_i"]),
            "input_tp": float(data["input_tp"]),
            "input_lra": float(data["input_lra"]),
            "input_thresh": float(data["input_thresh"]),
            "target_offset": float(data["target_offset"]),
        }
    except (ValueError, KeyError, TypeError):
        return None
    # Silence (or near-silence) measures as -inf, and feeding that back into the
    # filter makes ffmpeg fail with "Result too large". Two-pass needs real numbers.
    if not all(math.isfinite(value) for value in measured.values()):
        return None
    return measured


def loudnorm_filter(measured: dict | None) -> str:
    """Build the loudnorm filter, two-pass when measurements are available."""
    base = (f"loudnorm=I={LOUDNESS_TARGET['I']}:TP={LOUDNESS_TARGET['TP']}:"
            f"LRA={LOUDNESS_TARGET['LRA']}")
    if not measured or not all(math.isfinite(value) for value in measured.values()):
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )


# ------------------------------------------------------------------ music bed

def mix_music(speech_wav: str, music_path: str, out_wav: str, volume: float = 0.25,
              fade_ms: int = 500) -> bool:
    """Lay a music bed under the speech, ducking it whenever the voice speaks.

    `sidechaincompress` is triggered by the speech, so the bed drops away under
    every line and comes back in the gaps. The music is looped to cover the speech,
    faded in/out, and the mix is cut to the speech length (amix duration=first).
    """
    vol = max(0.0, min(1.0, float(volume)))
    fade_s = max(0.0, fade_ms / 1000.0)
    graph = (
        f"[1:a]volume={vol:.3f},afade=t=in:st=0:d={fade_s:.2f}[bed];"
        "[bed][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=5:release=300[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", speech_wav,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", graph,
        "-map", "[out]", "-c:a", "pcm_s16le", out_wav,
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=900)
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 0
    except Exception as e:
        print(f"  Warning: music mixing failed ({_ffmpeg_error(e)}); exporting without the bed")
        return False


# ----------------------------------------------------------------------- export

def _staging_path(output_path: str, output_format: str) -> str:
    """A sibling temp path that still ends in the real extension, so ffmpeg can infer
    the muxer -- and so an interrupted export never leaves a half-written deliverable
    sitting at the real output path.

    The pid keeps two concurrent runs on the same input from writing to the same
    staging file (and therefore from renaming each other's half-written output).
    """
    return f"{output_path}.{os.getpid()}.part{FORMATS[output_format]['ext']}"


def encode_with_ffmpeg(source_wav: str, output_path: str, output_format: str,
                       loudnorm_filter_str: str = None) -> bool:
    staging = _staging_path(output_path, output_format)
    cmd = ["ffmpeg", "-y", "-i", source_wav]
    if loudnorm_filter_str:
        cmd += ["-af", loudnorm_filter_str]
    cmd += ENCODER_ARGS[output_format]
    cmd.append(staging)
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=1800)
        if os.path.exists(staging) and os.path.getsize(staging) > 0:
            os.replace(staging, output_path)
            return True
        return False
    except Exception as e:
        print(f"  Warning: ffmpeg export failed ({_ffmpeg_error(e)})")
        return False
    finally:
        if os.path.exists(staging):
            with contextlib.suppress(OSError):
                os.remove(staging)


def _export_with_pydub(output: AudioSegment, output_path: str, output_format: str) -> bool:
    """Fallback export that still goes through a staging file + rename.

    This is the last line of defence, so it reports failure instead of raising: a
    run that got this far should end with a readable message, not a traceback.
    """
    staging = _staging_path(output_path, output_format)
    tmp_path = None
    try:
        if output_format == "m4a":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            output.export(tmp_path, format="wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-c:a", "aac", "-b:a", "192k",
                 "-ar", OUTPUT_SAMPLE_RATE, "-movflags", "+faststart", staging],
                capture_output=True, check=True, timeout=300,
            )
        elif output_format == "mp3":
            output.export(staging, format="mp3", bitrate="192k")
        elif output_format == "ogg":
            output.export(staging, format="ogg")
        else:
            output.export(staging, format="wav")
        os.replace(staging, output_path)
        return os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"  Warning: could not write {output_path} ({e})")
        return False
    finally:
        for path in (staging, tmp_path):
            if path and os.path.exists(path):
                with contextlib.suppress(OSError):
                    os.remove(path)


def export_audio(
    output: AudioSegment,
    output_path: str,
    output_format: str,
    *,
    loudnorm: bool = True,
    music: str = None,
    music_volume: float = 0.25,
) -> dict:
    """Export the final mix, optionally with a music bed and EBU R128 loudness.

    The speech is written to a temp WAV, optionally mixed with the ducked bed,
    then measured and normalized in two passes so the file actually lands on the
    target loudness. Returns a small dict describing what happened; if any ffmpeg
    step fails the plain pydub export is used so a run never dies at the last step.
    """
    info = {"ok": False, "loudnorm": False, "two_pass": False, "music": False,
            "measured": None}
    if not (loudnorm or music):
        info["ok"] = _export_with_pydub(output, output_path, output_format)
        return info

    tmp_wav = None
    mixed_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        output.export(tmp_wav, format="wav")

        source = tmp_wav
        if music:
            mixed_wav = tmp_wav + "_music.wav"
            if mix_music(tmp_wav, music, mixed_wav, music_volume):
                source = mixed_wav
                info["music"] = True

        loudnorm_str = None
        if loudnorm:
            measured = measure_loudness(source)
            info["measured"] = measured
            info["two_pass"] = measured is not None
            loudnorm_str = loudnorm_filter(measured)

        if encode_with_ffmpeg(source, output_path, output_format, loudnorm_str):
            info["loudnorm"] = bool(loudnorm)
            info["ok"] = True
            return info
        if loudnorm and encode_with_ffmpeg(source, output_path, output_format, None):
            print("  Note: exported without loudness normalization")
            info["ok"] = True
            return info
    finally:
        for path in (tmp_wav, mixed_wav):
            if path and os.path.exists(path):
                os.unlink(path)

    print("  Note: falling back to the plain pydub export")
    info["ok"] = _export_with_pydub(output, output_path, output_format)
    return info
