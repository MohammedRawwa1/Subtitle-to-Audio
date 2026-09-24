# subtitle-to-audio

Generate synchronized audio from .srt subtitle files. The generated audio matches the exact duration of the subtitle timeline while keeping short lines at a natural speaking pace. Only speech that would overrun its subtitle window is pitch-preserving time-compressed.

## How It Works

1. Parses the SRT file to get text and timing for each subtitle entry
2. Generates TTS audio for each segment using edge_tts, with pyttsx3 as an offline fallback
3. Uses ffmpeg's `atempo` filter only when a segment would overrun its exact time window (pitch preserved)
4. Assembles all segments on a timeline matching the SRT timestamps
5. The final audio duration matches the subtitle file's total duration

## Table of Contents

## Dependencies

* [pydub](https://github.com/jiaaro/pydub): Used for audio segment manipulations.
* [pysub-parser](https://pypi.org/project/pysub-parser/): Used for parsing .srt files.
* [pyttsx3](https://github.com/nateshmbhat/pyttsx3): An offline text-to-speech library used for dubbing.
* [ffmpeg](https://ffmpeg.org/): Required for pitch-preserving time stretching (atempo filter).

## Installation

1. Clone the repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
3. Install ffmpeg (required for audio stretching):
```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

## Usage

```bash
python subtitle_to_audio.py -p test/test.srt -r 150 -v 0 -f m4a
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--path` | Path to the .srt subtitle file | *required* |
| `-r`, `--rate` | Base TTS speech rate (words per minute) | `150` |
| `-v`, `--voice-idx` | Voice selection (0 or 1) | `0` |
| `-f`, `--format` | Output format: `wav`, `mp3`, `m4a`, `ogg` | `m4a` |

### Examples

```bash
# AAC output (default) - best quality-to-size ratio
python subtitle_to_audio.py -p test/test.srt -f m4a

# WAV output (uncompressed)
python subtitle_to_audio.py -p test/test.srt -f wav

# MP3 output
python subtitle_to_audio.py -p test/test.srt -f mp3

# With a longer subtitle file
python subtitle_to_audio.py -p srt/"Course Overview and Introduction.srt" -r 180 -f m4a
```

The output file is saved alongside the input .srt file with the matching extension.

## Sync Behavior

- If a subtitle segment has short text but a long time window, the speech stays natural and the remaining time is silence
- If a subtitle segment has long text but a short time window, the speech speeds up to fit
- Silence between subtitle segments is preserved naturally
- Overlapping subtitles are mixed together
- The total audio duration always matches the subtitle file's timeline
