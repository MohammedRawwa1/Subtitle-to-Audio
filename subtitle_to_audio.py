import os
import sys
import re
import asyncio
import tempfile
import argparse
import subprocess
from pysubparser import parser
from pydub import AudioSegment


def check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def time_to_ms(t):
    """Convert a time object to milliseconds."""
    return ((t.hour * 60 + t.minute) * 60 + t.second) * 1000 + t.microsecond / 1000


def stretch_audio(audio_segment, target_duration_ms):
    """
    Compress audio to fit target_duration_ms, preserving pitch.
    Short clips are intentionally not slowed down: trailing silence is more
    intelligible than artificially slow speech.
    """
    current_duration = len(audio_segment)
    if current_duration == 0 or target_duration_ms <= 0:
        return audio_segment

    if current_duration <= target_duration_ms:
        return audio_segment

    ratio = current_duration / target_duration_ms

    # If already very close to target (within 5%), skip stretching to avoid artifacts.
    if abs(ratio - 1.0) < 0.05:
        return audio_segment

    # atempo filter speed factor: speed > 1 speeds up, speed < 1 slows down
    speed = ratio

    # atempo filter only supports values between 0.5 and 100.0
    # Chain multiple filters for larger adjustments
    filters = []
    remaining = speed
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    filter_str = ",".join(filters)

    tmp_in_path = None
    tmp_out_path = None

    try:
        # Create temp files for ffmpeg I/O
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path + "_stretched.wav"

        audio_segment.export(tmp_in_path, format="wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_in_path,
            "-filter:a", filter_str,
            tmp_out_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

        result = AudioSegment.from_wav(tmp_out_path)
        return result

    except subprocess.CalledProcessError as e:
        print(f"  Warning: ffmpeg stretch failed: {e.stderr[:200] if e.stderr else e}")
        return audio_segment
    except FileNotFoundError:
        print("  Warning: ffmpeg not found, cannot stretch audio")
        return audio_segment
    finally:
        for f in [tmp_in_path, tmp_out_path]:
            if f and os.path.exists(f):
                os.unlink(f)


# ---------------------------------------------------------------------------
# TTS backends: edge_tts (natural, online) with pyttsx3 fallback (offline)
# ---------------------------------------------------------------------------

# Mapping chosen by the user. The key is the edge_tts ShortName (e.g. 'en-US-JennyNeural').
# When edge_tts is unavailable/offline, we fall back to pyttsx3 and pick the closest
# available SAPI5 voice by gender preference.
EDGE_VOICE_DEFAULT = "en-US-JennyNeural"  # natural female voice
EDGE_VOICE_MALE = "en-US-BrianNeural"     # natural male voice

# Voices the auto-picker considers, grouped by gender + vibe.
# The auto detector scores each SRT by simple cue words; the highest-scoring group
# wins, and within that group we prefer the most "conversational/warm" voice first.
AUTO_VOICES = {
    "female": [
        "en-US-AriaNeural",   # warm, conversational, good for personal storytelling
        "en-US-JennyNeural",  # clear neutral female
        "en-US-SarahNeural",  # slightly softer female
    ],
    "male": [
        "en-US-GuyNeural",    # warm male storytelling voice
        "en-US-BrianNeural",  # clear neutral male
        "en-US-DavisNeural",  # softer male
    ],
    "neutral": [
        "en-US-JennyNeural",  # default fallback if unsure
    ],
}

FORMATS = {
    "wav": {"ext": ".wav", "pydub_format": "wav", "description": "WAV (uncompressed)"},
    "mp3": {"ext": ".mp3", "pydub_format": "mp3", "description": "MP3"},
    "m4a": {"ext": ".m4a", "pydub_format": "ipod", "description": "AAC/M4A (better than MP3)"},
    "ogg": {"ext": ".ogg", "pydub_format": "ogg", "description": "OGG Vorbis"},
}


def _detect_voice_gender_from_text(text: str) -> str:
    """Guess a preferred TTS gender from the SRT text content.

    Looks for first-person pronouns and gendered/self-descriptive cues in the
    concatenated subtitle text. Returns 'female', 'male', or 'neutral'.
    """
    lowered = text.lower()

    # Neutral, content-agnostic default.
    best = "neutral"
    female_score = 0
    male_score = 0

    # Pronoun cues.
    female_score += lowered.count("she") + lowered.count("her")
    male_score += lowered.count("he") + lowered.count("him") + lowered.count("his")

    # Self-description cues that often indicate the speaker's gender.
    female_cues = [
        "woman", "women", "female", "ladies", "girlfriend", "wife",
        "mom", "mother", "daughter", "grandma", "actress", "femme",
        "makeup", "goddess", "queens", "princess", "damsel",
    ]
    male_cues = [
        "man", "men", "male", "guy", "boy", "boyfriend", "husband",
        "dad", "father", "son", "grandpa", "actor", "gentleman",
        "bros", "dude", "brother", "stallion",
    ]

    for c in female_cues:
        female_score += lowered.count(c)
    for c in male_cues:
        male_score += lowered.count(c)

    if female_score > male_score and female_score > 0:
        best = "female"
    elif male_score > female_score and male_score > 0:
        best = "male"

    return best


def _pick_auto_voice(text: str, gender: str = None) -> str:
    """Return the best edge_tts voice ShortName.

    If gender is explicitly provided, it always takes priority over
    automatic gender detection.
    """
    try:
        # Explicit --gender always wins.
        if gender in ("female", "male"):
            selected_gender = gender
        else:
            selected_gender = _detect_voice_gender_from_text(text)

        candidates = AUTO_VOICES.get(
            selected_gender,
            AUTO_VOICES["neutral"]
        )

        return candidates[0] if candidates else EDGE_VOICE_DEFAULT

    except Exception:
        return EDGE_VOICE_DEFAULT


def _estimate_tts_duration_chars_ms(text: str, rate_percent: str) -> int:
    """Rough per-character duration estimate for edge_tts at a given rate.

    Calibrated from samples at en-US-JennyNeural:
      +0%   -> ~6.77s for 101 chars  (~67.0 ms/char)
      +10%  -> ~6.17s for 101 chars  (~61.1 ms/char)
      +20%  -> ~5.64s for 101 chars  (~55.8 ms/char)
      +30%  -> ~5.23s for 101 chars  (~51.8 ms/char)
      -10%  -> ~7.51s for 101 chars  (~74.4 ms/char)

    We fit a simple linear model: ms_per_char = a + b * rate_percent.
    """
    # Parse signed percentage like '+10%' / '-20%' / '+0%'
    m = re.match(r"([+-])(\d+)%", rate_percent)
    if not m:
        raise ValueError(f"Invalid rate format: {rate_percent!r}")
    sign = 1 if m.group(1) == "+" else -1
    pct = sign * int(m.group(2))

    # Calibrated coefficients from the samples above.
    base_ms_per_char = 67.0     # at +0%
    ms_per_char_per_pct = -0.52 # each +1% reduces duration by ~0.52 ms/char

    ms_per_char = base_ms_per_char + ms_per_char_per_pct * pct
    return max(1, int(ms_per_char * len(text)))


def _rate_percent_for_window(text: str, target_ms: int) -> str:
    """Choose an edge_tts rate percentage so the generated audio fits the SRT window.

    We want: estimated_duration_ms(text, rate) ~= target_ms.
    Using the linear model above, solve for rate_percent.
    """
    if target_ms <= 0 or not text:
        return "+0%"

    base_ms_per_char = 67.0
    ms_per_char_per_pct = -0.52

    desired_ms_per_char = target_ms / max(1, len(text))
    # desired = base + coef * pct  =>  pct = (desired - base) / coef
    pct = (desired_ms_per_char - base_ms_per_char) / ms_per_char_per_pct

    # Clamp to a range that still sounds natural. Beyond about +/-35% edge_tts starts
    # to sound noticeably sped up/slow and robotic, so we cap there and let the light
    # atempo safety-stretch handle the rest.
    pct = max(-35, min(35, int(round(pct))))
    return f"{pct:+d}%"


async def _tts_edge(text: str, voice: str, rate_percent: str, out_path: str) -> bool:
    """Render `text` with edge_tts into `out_path` (mp3). Returns True on success."""
    try:
        import edge_tts
    except Exception as e:
        print(f"  edge_tts import failed: {e}")
        return False

    try:
        comm = edge_tts.Communicate(
            text,
            voice,
            rate=rate_percent,
            volume="+0%",
            pitch="+0Hz",
        )
        await comm.save(out_path)
        return True
    except Exception as e:
        print(f"  edge_tts render failed: {e}")
        return False


def _tts_pyttsx3(text: str, voice_id: str, rate_wpm: int, out_path: str) -> bool:
    """Render `text` with pyttsx3 into `out_path` (wav). Returns True on success."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate_wpm)
        if voice_id:
            voices = engine.getProperty("voices")
            chosen = None
            for v in voices:
                if voice_id.lower() in v.id.lower() or voice_id.lower() in v.name.lower():
                    chosen = v
                    break
            if chosen is not None:
                engine.setProperty("voice", chosen.id)
            else:
                print(f"  pyttsx3: requested voice {voice_id!r} not found, using default")
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"  pyttsx3 render failed: {e}")
        return False


def _choose_pyttsx3_voice(gender: str) -> str:
    """Return a pyttsx3 voice id matching the requested gender, or empty string."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        for v in voices:
            name = (v.name or "").lower()
            vid = (v.id or "").lower()
            blob = f"{name} {vid}"
            if gender == "male" and ("male" in blob or "david" in blob):
                return v.id
            if gender == "female" and ("female" in blob or "zira" in blob):
                return v.id
        # Fallback: return first voice id if nothing matched.
        return voices[0].id if voices else ""
    except Exception as e:
        print(f"  pyttsx3 voice discovery failed: {e}")
        return ""


def _needs_m4a_ffmpeg_export(output_format: str) -> bool:
    return output_format == "m4a"


def _export_audio(output: AudioSegment, output_path: str, output_format: str) -> None:
    """Export the final mixed audio to disk in the requested format."""
    if output_format == "m4a":
        # AAC export via ffmpeg for best quality
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            output.export(tmp_path, format="wav")
            cmd = [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    elif output_format == "mp3":
        output.export(output_path, format="mp3", bitrate="192k")
    elif output_format == "ogg":
        output.export(output_path, format="ogg")
    else:
        output.export(output_path, format="wav")


async def generate_audio(
    path: str,
    *,
    voice: str = EDGE_VOICE_DEFAULT,
    gender: str = "female",
    output_format: str = "m4a",
    base_rate_percent: str = "+0%",
    pyttsx3_rate_wpm: int = 150,
    stretch_tolerance: float = 0.05,
    retry_on_stretch_failure: bool = True,
) -> None:
    """Generate synchronized audio from an SRT subtitle file.

    For each subtitle segment we render at the requested natural rate. If speech
    would overrun the SRT window, it is lightly pitch-preserve time-compressed with
    ffmpeg; shorter speech keeps its cadence and leaves silence before the next cue.

    If edge_tts is unavailable or fails for a segment, the function falls back to
    pyttsx3 for that segment (and reuses that path for the rest of the run if needed).
    """
    print(f"Generating audio for {path}")
    print(f"Primary TTS: edge_tts voice={voice}  |  fallback: pyttsx3")

    subtitles = list(parser.parse(path))
    if not subtitles:
        print("No subtitles found in the file!")
        return

    # Calculate total duration from the last subtitle's end time.
    last_subtitle = subtitles[-1]
    total_duration_ms = int(time_to_ms(last_subtitle.end))

    print(f"Found {len(subtitles)} subtitle entries")
    print(f"Target total duration: {total_duration_ms / 1000:.1f}s "
          f"({total_duration_ms // 60000}m {(total_duration_ms % 60000) // 1000}s)")
    print(f"Stretch tolerance: {stretch_tolerance*100:.0f}% (only stretch if off by more)")

    # Create a silent base track matching the total subtitle duration.
    output = AudioSegment.silent(duration=max(1, total_duration_ms))

    # The fallback pyttsx3 voice id, resolved once if needed.
    pyttsx3_voice_id: str = ""
    using_fallback: bool = False

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, subtitle in enumerate(subtitles):
            # Clean up any stray newlines/spaces so TTS reads it naturally.
            text = subtitle.text.replace("\r", "").replace("\n", " ").strip()
            if not text:
                print(f"  [{i+1}/{len(subtitles)}] Skipping (empty text)")
                continue

            start_ms = time_to_ms(subtitle.start)
            end_ms = time_to_ms(subtitle.end)
            target_duration_ms = end_ms - start_ms

            if target_duration_ms <= 0:
                print(f"  [{i+1}/{len(subtitles)}] Skipping (zero/negative duration)")
                continue

            # Render at a natural rate. A subtitle window is an upper bound for
            # speech, not a reason to slow a short sentence down.
            rate_percent = base_rate_percent

            tmp_path = os.path.join(tmpdirname, f"seg_{i:04d}.mp3")
            rendered = False

            if not using_fallback:
                # Try edge_tts first.
                ok = await _tts_edge(text, voice, rate_percent, tmp_path)
                if ok and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    rendered = True
                    source = "edge_tts"
                else:
                    # Edge failed or produced an empty file -> switch to fallback.
                    print(f"  [{i+1}/{len(subtitles)}] edge_tts failed, switching to pyttsx3 fallback")
                    using_fallback = True
                    pyttsx3_voice_id = _choose_pyttsx3_voice(gender)

            if not rendered:
                # pyttsx3 fallback: render to wav, then load with pydub.
                wav_path = os.path.join(tmpdirname, f"seg_{i:04d}.wav")
                ok = _tts_pyttsx3(text, pyttsx3_voice_id, pyttsx3_rate_wpm, wav_path)
                if ok and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    tmp_path = wav_path
                    rendered = True
                    source = "pyttsx3"

            if not rendered:
                print(f"  [{i+1}/{len(subtitles)}] TTS failed for both engines, skipping segment")
                continue

            # Load whichever file was produced.
            try:
                if tmp_path.lower().endswith(".mp3"):
                    audio_segment = AudioSegment.from_mp3(tmp_path)
                else:
                    audio_segment = AudioSegment.from_wav(tmp_path)
            except Exception as e:
                print(f"  [{i+1}/{len(subtitles)}] Failed to load rendered audio: {e}")
                continue

            current_duration = len(audio_segment)

            # Only compress an overlong segment. Short segments keep their natural
            # cadence and are padded with silence until the next subtitle cue.
            ratio = current_duration / target_duration_ms if target_duration_ms else 1.0
            do_stretch = ratio > 1.0 + stretch_tolerance

            if do_stretch:
                stretched = stretch_audio(audio_segment, target_duration_ms)
            else:
                stretched = audio_segment

            # Trim to exact target duration as a final safety measure.
            if len(stretched) > target_duration_ms:
                stretched = stretched[:target_duration_ms]
            elif len(stretched) < target_duration_ms:
                # Pad with a tiny bit of silence if TTS came out short (rare after tuning).
                stretched = stretched + AudioSegment.silent(duration=target_duration_ms - len(stretched))

            # Overlay the rendered audio at the correct position on the timeline.
            output = output.overlay(stretched, position=int(start_ms))

            # Progress feedback.
            if do_stretch:
                speed_label = f"{ratio:.2f}x (compressed)"
            else:
                speed_label = f"{ratio:.2f}x (natural + silence)"
            text_preview = text[:55].replace("  ", " ")
            print(f"  [{i+1}/{len(subtitles)}] {subtitle.start} -> {subtitle.end} | "
                  f"{source:9s} rate={rate_percent:4s} | {speed_label:24s} | \"{text_preview}\"")

    # Export final audio.
    fmt = FORMATS[output_format]
    output_path = os.path.splitext(path)[0] + fmt["ext"]
    _export_audio(output, output_path, output_format)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nAudio exported to: {output_path}")
    print(f"Format: {fmt['description']}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Final duration: {len(output) / 1000:.1f}s "
          f"({len(output) // 60000}m {(len(output) % 60000) // 1000}s)")
    print(f"Target duration: {total_duration_ms / 1000:.1f}s "
          f"({total_duration_ms // 60000}m {(total_duration_ms % 60000) // 1000}s)")


if __name__ == "__main__":
    # Check for ffmpeg before starting (needed for light safety-stretch + m4a export).
    if not check_ffmpeg():
        print("Error: ffmpeg is required for light audio stretching and M4A export.")
        print("Install ffmpeg and make sure it's available in your PATH.")
        print("  Windows: winget install ffmpeg  (or download from https://ffmpeg.org)")
        print("  macOS:   brew install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        sys.exit(1)

    arg_parser = argparse.ArgumentParser(
        description=(
            "Generate synchronized audio from SRT subtitle files using a natural"
            " edge_tts voice (with pyttsx3 fallback). Per-segment speech rate is chosen"
            " so the rendered audio fits each SRT time window, keeping the voice natural."
        )
    )
    arg_parser.add_argument("-p", "--path", help="subtitle file path (.srt)", required=True)
    arg_parser.add_argument("-f", "--format", help="output audio format",
                            choices=list(FORMATS.keys()), default="m4a")
    arg_parser.add_argument(
        "--voice",
        help="edge_tts voice ShortName (for example en-US-JennyNeural or en-US-BrianNeural). Default: en-US-JennyNeural",
        default=EDGE_VOICE_DEFAULT,
    )
    arg_parser.add_argument(
        "--gender",
        help="gender hint used only for the pyttsx3 fallback voice (male/female). Default: female",
        choices=["male", "female"],
        default="female",
    )
    arg_parser.add_argument(
        "--base-rate",
        help="edge_tts base rate percentage override such as +0 percent or +10 percent. Default: +0 percent",
        default="+0%",
    )
    arg_parser.add_argument(
        "--pyttsx3-rate",
        help="pyttsx3 fallback speech rate in words per minute. Default: 150",
        type=int,
        default=150,
    )
    arg_parser.add_argument(
        "--stretch-tolerance",
        help="only time-stretch a segment if its duration is off by more than this fraction. Default: 0.05, i.e. 5 percent tolerance",
        type=float,
        default=0.05,
    )
    arg_parser.add_argument(
        "--auto-voice",
        action="store_true",
        help="auto-detect the best TTS voice from the SRT text content (gender cues + preferred style).",
    )

    args = arg_parser.parse_args()

    voice = args.voice
    if args.auto_voice:
        # Read the SRT text once to pick a voice.
        try:
            from pysubparser import parser as _parser
            raw = list(_parser.parse(args.path))
            joined = " ".join(
                s.text.replace("\r", " ").replace("\n", " ").strip()
                for s in raw
                if (s.text or "").strip()
            )
            picked = _pick_auto_voice(joined, args.gender)
            print(f"Auto-picked voice for this SRT: {picked}")
            voice = picked
        except Exception as e:
            print(f"Auto-voice detection failed ({e}), falling back to --voice default")

    asyncio.run(generate_audio(
        path=args.path,
        voice=voice,
        gender=args.gender,
        output_format=args.format,
        base_rate_percent=args.base_rate,
        pyttsx3_rate_wpm=args.pyttsx3_rate,
        stretch_tolerance=args.stretch_tolerance,
    ))
