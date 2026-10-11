"""Objective measurements of a rendered voice.

There is no substitute for listening, but several things that make speech sound right
(or wrong) are measurable, and measuring them turns "it sounds flat" into a number:

* **register** -- median F0. A voice far outside the usual 80-260 Hz range is either
  the wrong gender for the text or an artifact.
* **expressiveness** -- F0 range in semitones across voiced frames. Under ~2 st is
  monotone; 4-8 st is normal conversational reading.
* **pacing** -- words per minute over the *voiced* audio only, so trailing silence in
  a subtitle window cannot flatter it.
* **distortion** -- clipped-sample percentage and true peak: an over-driven or
  over-normalized render is measurable before anyone hears it.
* **delivery** -- integrated loudness, and how much of the clip is silence.

Everything here works on a pydub segment or a file, needs only numpy (which arrives
with onnxruntime) and optional ffmpeg for loudness.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile

import numpy as np
from pydub import AudioSegment

__all__ = [
    "frame_pitch",
    "voice_metrics",
    "measure_file",
    "register_label",
    "MONOTONE_SEMITONES",
    "NATURAL_SEMITONES",
]

# Conversational reading spans roughly 4-8 semitones of pitch; anything under two is
# heard as flat, and a very wide span on a short clip is usually instability.
MONOTONE_SEMITONES = 2.0
NATURAL_SEMITONES = 4.0

# Below this frame RMS there is no voice to measure -- silence, not a low note.
_SILENCE_RMS = 10 ** (-45 / 20)
_VOICED_CORRELATION = 0.30


def _to_samples(audio: AudioSegment) -> tuple[np.ndarray, int]:
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    scale = float(1 << (8 * audio.sample_width - 1))
    return samples / scale, int(audio.frame_rate)


def _frames(samples: np.ndarray, sample_rate: int, frame_ms: int = 40, hop_ms: int = 20):
    size = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    if len(samples) < size:
        yield samples, hop
        return
    for start in range(0, len(samples) - size + 1, hop):
        yield samples[start:start + size], start


def frame_pitch(audio: AudioSegment, fmin: int = 60, fmax: int = 400) -> dict:
    """Autocorrelation pitch track: median/range F0 over the frames that carry voice.

    Returns ``{"median_f0_hz", "p10_f0_hz", "p90_f0_hz", "semitones", "voiced_pct",
    "silent_pct", "frames"}``; pitch fields are ``None`` when nothing was voiced.
    """
    samples, sample_rate = _to_samples(audio)
    if len(samples) == 0:
        return {"median_f0_hz": None, "p10_f0_hz": None, "p90_f0_hz": None,
                "semitones": None, "voiced_pct": 0.0, "silent_pct": 100.0, "frames": 0}

    min_lag = max(2, int(sample_rate / fmax))
    max_lag = int(sample_rate / fmin)
    pitches, voiced, silent, total = [], 0, 0, 0

    for frame, _start in _frames(samples, sample_rate):
        total += 1
        frame = frame - frame.mean()
        rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) else 0.0
        if rms < _SILENCE_RMS:
            silent += 1
            continue
        windowed = frame * np.hanning(len(frame))
        corr = np.correlate(windowed, windowed, mode="full")[len(frame) - 1:]
        if corr[0] <= 0 or max_lag >= len(corr):
            continue
        window = corr[min_lag:max_lag]
        if len(window) == 0:
            continue
        lag = int(np.argmax(window)) + min_lag
        strength = float(corr[lag] / corr[0])
        if strength < _VOICED_CORRELATION:
            continue
        # Parabolic refinement around the peak for sub-sample resolution.
        if 0 < lag < len(corr) - 1:
            a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
            denom = (a - 2 * b + c)
            offset = 0.5 * (a - c) / denom if denom else 0.0
            lag = lag + max(-1.0, min(1.0, offset))
        if lag > 0:
            voiced += 1
            pitches.append(sample_rate / lag)

    if not pitches:
        return {"median_f0_hz": None, "p10_f0_hz": None, "p90_f0_hz": None,
                "semitones": None, "voiced_pct": 0.0,
                "silent_pct": round(100.0 * silent / max(1, total), 1), "frames": total}

    p10, p90 = (float(v) for v in np.percentile(pitches, [10, 90]))
    semitones = 12.0 * math.log2(p90 / p10) if p10 > 0 else None
    return {
        "median_f0_hz": round(float(np.median(pitches)), 1),
        "p10_f0_hz": round(p10, 1),
        "p90_f0_hz": round(p90, 1),
        "semitones": round(semitones, 2) if semitones is not None else None,
        "voiced_pct": round(100.0 * voiced / max(1, total), 1),
        "silent_pct": round(100.0 * silent / max(1, total), 1),
        "frames": total,
    }


def clipped_percent(audio: AudioSegment, min_run: int = 3, threshold: float = 0.999) -> float:
    """Percentage of samples inside a *run* of full-scale samples.

    Distortion has flat tops, so a run is what matters: a clean tone whose peaks touch
    full scale for a sample or two is not clipped, while anything driven into the
    ceiling sits there for three samples or more.
    """
    samples, _ = _to_samples(audio)
    if len(samples) == 0:
        return 0.0
    at_full = np.abs(samples) >= threshold
    if not at_full.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([False], at_full, [False])).astype(np.int8)))
    lengths = edges[1::2] - edges[0::2]
    clipped = int(lengths[lengths >= min_run].sum())
    return round(100.0 * clipped / len(samples), 4)


def measure_loudness_file(path: str) -> dict | None:
    """Integrated loudness and true peak straight from ffmpeg's loudnorm pass."""
    cmd = ["ffmpeg", "-hide_banner", "-i", path, "-af",
           "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.S)
    if not match:
        return None
    try:
        import json
        data = json.loads(match.group(0))
        return {"lufs": float(data["input_i"]), "true_peak_db": float(data["input_tp"])}
    except (ValueError, KeyError, TypeError):
        return None


def register_label(median_f0: float | None) -> str:
    """A rough human label for a pitch median, for reporting only."""
    if median_f0 is None:
        return "no voice detected"
    if median_f0 < 80:
        return "very low"
    if median_f0 < 155:
        return "male-ish"
    if median_f0 < 255:
        return "female-ish"
    return "high/child"


def expressiveness_label(semitones: float | None) -> str:
    if semitones is None:
        return "unmeasured"
    if semitones < MONOTONE_SEMITONES:
        return "flat"
    if semitones < NATURAL_SEMITONES:
        return "limited"
    return "expressive"


def voice_metrics(audio: AudioSegment, words: int = None, elapsed_s: float = None) -> dict:
    """Everything measurable about one rendered clip, in one dict."""
    duration_s = len(audio) / 1000.0
    pitch = frame_pitch(audio)
    voiced_s = duration_s * pitch["voiced_pct"] / 100.0
    metrics = {
        "duration_s": round(duration_s, 3),
        "peak_dbfs": round(audio.max_dBFS, 2) if duration_s else None,
        "rms_dbfs": round(audio.dBFS, 2) if duration_s and audio.dBFS != float("-inf") else None,
        "clipped_pct": clipped_percent(audio),
        **pitch,
    }
    metrics["register"] = register_label(pitch["median_f0_hz"])
    metrics["expressiveness"] = expressiveness_label(pitch["semitones"])
    if words:
        metrics["words_per_minute"] = round(words / voiced_s * 60.0, 1) if voiced_s > 0 else None
        metrics["ms_per_char"] = None            # filled by the caller, which knows the text
    if elapsed_s and elapsed_s > 0:
        metrics["realtime_factor"] = round(duration_s / elapsed_s, 2)
    return metrics


def measure_file(path: str, words: int = None, elapsed_s: float = None) -> dict:
    """Measure a rendered file: this module's metrics plus ffmpeg's loudness."""
    audio = AudioSegment.from_file(path)
    metrics = voice_metrics(audio, words=words, elapsed_s=elapsed_s)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "measure.wav")
        audio.export(wav, format="wav")
        loud = measure_loudness_file(wav)
    if loud:
        metrics.update(loud)
    return metrics
