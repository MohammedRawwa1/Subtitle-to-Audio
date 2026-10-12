# subtitle-to-audio

Turn a subtitle file into a voice-over that keeps the subtitle timing: every line is
spoken at a rate chosen to fit its own time window, so the audio stays on the timeline
without dragging the voice out or squashing it into a warp.

## Just run it

```bash
python subtitle_to_audio.py myfile.srt              # -> myfile.m4a, next to the subtitle
python subtitle_to_audio.py myfile.srt --preview    # first 3 lines only, to check the voice
python subtitle_to_audio.py myfile.srt --voices     # what this machine can actually speak with
python subtitle_to_audio.py --guide                 # every option on one page, in the tool
```

Nothing is required but the subtitle file: the engine is auto-picked, the voice is chosen
for the text, and the output lands beside the input. Every other choice is a flag you can
add when you want it — `--guide` prints the whole list, `--voices` prints what is installed.

## Architecture

Each concern lives in one module, so it can be tested (and replaced) on its own:

| module | responsibility |
|---|---|
| `subtitle_to_audio.py` | command line: options, wiring, exit codes |
| `subtitle_io.py` | read/repair subtitles (.srt/.vtt/.ass/.ssa), text normalization, pronunciation lexicon |
| `pipeline_core.py` | pure logic: rate fitting, cue stitching, speaker labels, overlap and overflow rules |
| `tts_engines.py` | edge_tts / piper / pyttsx3 behind one interface, availability probing, render cache |
| `audio_post.py` | atempo, peak-safe levelling, two-pass EBU R128, music bed, export |
| `voice_metrics.py` | objective voice measurements: register, expressiveness, pacing, distortion |
| `pipeline.py` | orchestration: concurrent rendering, placement, metrics, report |

## Engines

`speed` is a dimensionless multiplier (1.0 = the engine's natural rate) across all
three, which is why one fitter works for every voice:

| engine | kind | rate control | needs |
|---|---|---|---|
| `edge_tts` | neural, online | rate percent | network to Microsoft's endpoint |
| `piper` | neural, **offline** | `length_scale` | a voice model (~63 MB) |
| `pyttsx3` | SAPI5/espeak, offline | words per minute | nothing (always available) |

`--engine auto` (default) puts the best available first and falls back down the chain:
edge_tts → piper → pyttsx3. Online engines are **probed once at startup**, so a blocked
endpoint is skipped instead of being discovered three lines in. Microsoft refuses the
handshake from some networks outright; `--edge-proxy URL` (or `$EDGE_TTS_PROXY` /
`$HTTPS_PROXY`) is the way through that.

```bash
pip install -r requirements.txt
python scripts/download_voice.py --pack   # three offline voices -> models/
```

More than one model in `models/` means `--auto-speakers` can give each character their
own offline voice: `--pack` installs one female, one male and one British voice.

### Other languages

Piper voices are not English-only, so a non-English subtitle file works the same way:

```bash
python scripts/download_voice.py ar_JO-kareem-medium   # 63 MB, one voice
python subtitle_to_audio.py course.srt --preview       # the Arabic voice is picked automatically
```

The offline voice is chosen from the subtitles' writing system (Arabic, Cyrillic, Greek,
CJK, Hebrew, Thai, Devanagari hook into it; Latin prefers English), so installing a
voice for one language cannot silently re-voice another language's file. `--piper-model`
still overrides it outright.

Measured on three Arabic cues (Arabic-Indic digits in the timestamps included): 3/3 units
spoken, 11.0 s for an 11.0 s timeline, median pitch 113 Hz, 10.6 semitones of range,
82.5% voiced frames, no clipping. If you ask a voice to speak a language it does not have
(an English voice reading Arabic, say), it returns a valid but **silent** file — that is
now reported as `silent renders` and fails `--strict` instead of quietly filling the
timeline with nothing.

## Reproducibility

Piper's duration predictor is stochastic, so an unmodified run can differ from the one
before it. `--deterministic` turns that noise off, which makes every line's length -- and
therefore every fitting decision -- repeatable; adding `--piper-noise 0` freezes the
waveform noise too, and the delivered file is then **byte-identical** between runs.

Measured over three renders of one 52-word paragraph (`scripts/benchmark_voices.py`):

| voice | length spread | F0 range |
|---|---|---|
| `en_US-lessac-medium` | 5.18% | 13.3 st |
| `en_US-ryan-medium` | 2.40% | 10.8 st |
| `en_US-lessac-medium --deterministic` | **0.00%** | 13.6 st |

The cost of reproducibility is small and measured: the same voice reads about 7% faster
with the duration noise off. `--piper-noise` and `--piper-noise-w` expose the knobs
directly if you want to trade stability against liveliness yourself.

## Voice quality, measured

There is no listening test in this repository, but the things listeners notice are
measurable, so `python scripts/benchmark_voices.py` renders a fixed paragraph through
every available voice and reports them (median pitch, pitch range in semitones, words
per minute over *voiced* audio only, loudness, clipping, and how far the render drifts):

| voice | realtime | ms/char | wpm | LUFS | clip | F0 median | register | F0 range | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `piper en_US-lessac-medium` | 8.8x | 47.3 | 275 | −16.9 | 0.00% | 194 Hz | female-ish | 13.3 st | expressive |
| `piper en_US-ryan-medium` | 9.8x | 47.9 | 272 | −15.9 | 0.00% | 145 Hz | male-ish | 10.8 st | expressive |
| `piper en_US-lessac-medium [det]` | 8.9x | 45.0 | 294 | −17.1 | 0.00% | 191 Hz | female-ish | 13.6 st | expressive |
| `pyttsx3 ZIRA` | 92.5x | 50.5 | 252 | −20.3 | 0.00% | 178 Hz | female-ish | 6.7 st | expressive |

Loudness and peak are measured on the raw engine render, before the EBU R128 pass. Read
the range column as the sanity check it is: under ~2 semitones is monotone, and the
offline SAPI fallback is measurably flatter (6.7 st) than piper (~11-14 st) while
running ten times faster. `tests/test_voice_metrics.py` asserts a delivered file really
does land in a plausible register, with movement, near the loudness target, and no
clipping -- so a regression in voice quality has somewhere to show up.

## Usage

# English example
python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice en-US-JennyNeural --format m4a --preview 8

python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice en-US-JennyNeural --format m4a --max-rate 15 --max-speed 1.10 --overflow bleed --preview 8

python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice en-US-JennyNeural --format m4a --metrics --report ".\myfile-report.json"

python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice en-US-JennyNeural --format m4a --max-rate 15 --max-speed 1.10 --overflow bleed --metrics --report ".\myfile-report.json"

# Arabic example
python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice ar-SA-ZariyahNeural --format m4a --max-rate 15 --max-speed 1.10 --overflow bleed --preview 8

python subtitle_to_audio.py ".\myfile.srt" --engine edge_tts --voice ar-SA-ZariyahNeural --format m4a --max-rate 15 --max-speed 1.10 --overflow bleed --metrics --report ".\myfile-report.json"


```bash
python subtitle_to_audio.py movie.srt                     # the simple case
python subtitle_to_audio.py movie.srt --preview 8         # first 8 cues only
python subtitle_to_audio.py movie.srt --strict --report movie.json --metrics
python subtitle_to_audio.py movie.vtt --overflow bleed    # let long lines run past their cue
python subtitle_to_audio.py show.ass --auto-speakers --deterministic   # one voice per speaker
python subtitle_to_audio.py talk.srt --lexicon names.json --say "nginx=engine x"
python subtitle_to_audio.py two-hander.srt --music bed.mp3 --music-volume 0.2
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| *(positional)*, `-p`, `--path` | Subtitle file (`.srt`, `.vtt`, `.ass`, `.ssa`, `.sub`, `.txt`) | *required* |
| `-f`, `--format` | Output: `wav`, `mp3`, `m4a`, `ogg` | `m4a` |
| `--preview [N]` | Speak only the first N cues, write `<name>.preview.<ext>` | off (`3` when bare) |
| `--voices` | List the voices installed on this machine and exit | off |
| `--guide` | Print every option on one page and exit | off |
| `--engine` | `auto`, `edge_tts`, `piper`, `pyttsx3` | `auto` |
| `--voice` | edge_tts voice ShortName | `en-US-JennyNeural` |
| `--piper-model` | piper `.onnx` voice (else `$PIPER_MODEL`, else the installed voice matching the subtitles' language) | auto |
| `--piper-noise` / `--piper-noise-w` | piper expressiveness / rhythm noise | model values |
| `--deterministic` | turn piper's duration noise off so re-runs match | off |
| `--edge-proxy` | proxy URL for the online engine (else `$EDGE_TTS_PROXY`, `$HTTPS_PROXY`) | – |
| `--gender` | Gender hint for fallback voices | `female` |
| `--auto-voice` | Pick the voice from the subtitle text | off |
| `--speaker-voices` | Per-speaker voice, `NAME=VOICE`, repeatable | – |
| `--auto-speakers` | Detect `NAME:` tags and rotate voices | off |
| `--base-rate` | Natural starting rate; fitting never goes below it | `+0%` |
| `--max-rate` | Fastest rate the fitter may use, as an edge_tts rate offset (piper/pyttsx3 use their own natural ranges instead) | `30` |
| `--pyttsx3-rate` / `--max-wpm` | Fallback WPM range | `165` / `280` |
| `--max-speed` | Cap on the safety atempo (`1.0` = none) | `1.25` |
| `--no-stretch` | Disable the safety atempo | off |
| `--stretch-tolerance` / `--fit-attempts` / `--tts-retries` | Fitting controls | `0.05` / `3` / `2` |
| `--overflow` | `cut` trim · `bleed` run past the cue · `compress` atempo to fit | `cut` |
| `--max-compress` | Speed-up ceiling for `--overflow compress` before it trims | `8.0` |
| `--overlap` | `mix` · `trim` earlier · `shift` later | `mix` |
| `--no-stitch` | Do not merge sentences split across cues | off |
| `--no-normalize` | Keep raw text (no tag stripping/expansion) | off |
| `--no-level` | Skip per-line levelling | off |
| `--no-loudnorm` | Skip the EBU R128 pass | off |
| `--music` / `--music-volume` | Looped, ducked background bed | – / `0.25` |
| `--lexicon` / `--say` | Pronunciation overrides (JSON file / `WORD=SAY`) | – |
| `--concurrency` | Lines rendered in parallel | `4` |
| `--cache-dir` / `--no-cache` / `--clear-cache` | Render cache (keyed by engine+voice+speed+text+noise) | `.tts_cache` |
| `--fallback-after` | Failures before an engine is dropped for the run | `1` |
| `--metrics` | Measure onset alignment in the final audio | off |
| `--report` | Write the JSON quality report here | – |
| `--strict` | Exit `3` if the run degraded | off |

### Exit codes

| code | meaning |
|---|---|
| `0` | success |
| `1` | setup problem (missing/unreadable input, nothing to speak, no usable engine, no ffmpeg) or the mix could not be written |
| `3` | `--strict` and the run degraded (cuts, bleeds, no audio for a cue, wrong duration, alignment < 90%, engine fallback, export failure) |
| `130` | interrupted (Ctrl-C) |

## Behaviour

- A line with little text in a long window keeps the base rate; the rest is silence.
- A line with too much text speeds up, up to `--max-rate`, then the bounded safety
  atempo, then `--overflow` decides.
- Timing is guaranteed to the inaudible epsilon (50 ms): a line may end up to that far
  past its window, and only the remainder is trimmed, which is what keeps the total
  duration on the subtitle timeline.
- Conversations split across cues are stitched into one phrase; `NAME:` tags become
  speaker metadata (never spoken) instead of restarting the reader.
- Per line: trimmed of TTS silence, levelled to −20 dBFS RMS with peak headroom,
  placed on the exact cue start. The mix then gets a two-pass EBU R128 pass
  (targets I=−16 LUFS, TP=−1.5 dBTP) and a 48 kHz encoder.
- The total duration matches the subtitle timeline, except with `--overflow bleed`.

## Report

Every run ends with a summary: units, cues covered, engine actually used, renders,
cache hits, fit ratios, cuts, bleeds, silent renders, loudness mode, and whether the
renders were repeatable. `--metrics` adds onset alignment (units matched, within 150 ms,
median/mean/max offset, and how many units had no detectable onset — those are excluded
rather than matched to a distant neighbour). `--report out.json` writes the same data
plus per-unit detail, through a staging file and a rename, so a crash cannot leave a
truncated report behind.

## Robustness

- One line failing on one engine falls back to the next engine, and a line that fails
  everywhere is skipped, reported, and fails `--strict` — it never ends the run.
- A render that comes back **silent** (the voice cannot speak that language) is counted,
  printed, tagged `SILENT`, and fails `--strict` — not passed off as audio.
- Non-Latin subtitles cannot crash a redirected run: output is re-encoded as UTF-8, so
  Arabic/Cyrillic/CJK text survives `> log.txt` on a Windows console codepage.
- Silent input cannot break the loudness pass: a `-inf` measurement falls back to
  one-pass normalization instead of an ffmpeg failure.
- An unreachable online engine is detected by a startup probe, not three lines in.
- A corrupt or truncated cache entry is re-rendered instead of failing the line.
- Fresh renders, cache entries, reports and the exported audio are all written through
  a staging file and renamed, so no interrupt leaves a half-written deliverable.
- If the mix cannot be written, the report still says what the run produced
  (`export_ok: false`) and the exit code is `1` rather than a traceback.
- Missing music, missing ffmpeg, unreadable input and an empty subtitle file all end in
  a readable message, not a stack trace.

## Performance

Renders are content-addressed (engine + voice + speed + text + the engine's noise
settings) and cached in `.tts_cache`, so re-runs and long jobs resume instead of
re-synthesising. Placement and logging stay strictly sequential, so output never
depends on completion order.

Measured on the 82-cue / 5-minute fixture (8 CPU cores, piper, wav export without
loudnorm, so this isolates rendering):

| run | renders | wall time |
|---|---|---|
| cold cache | 89 | 46.0 s |
| warm cache | **0** (112 hits) | **13.0 s** (3.5x faster) |

Parallelism is engine-aware: `--concurrency` defaults to 4 for a network engine
(where renders are latency-bound) and 1 for piper, because onnxruntime already uses
every core — measured 0.98x at 4 and 0.84x at 8 for piper, so extra concurrency only
adds contention there. Piper runs at roughly 9x realtime on CPU (`--deterministic`
included).

## Tests

```bash
python -m pytest tests -q                    # everything (e2e needs ffmpeg + a voice model)
python -m pytest tests -q -m "not slow"      # fast unit tests only
ruff check .
```

| file | covers |
|---|---|
| `tests/test_subtitle_io.py` | formats, malformed repair, normalization |
| `tests/test_core.py` | fitting, stitching, speakers, overlap, overflow, levelling, engines, cache, lexicon, ffmpeg stages |
| `tests/test_cli.py` | documented flags, exit codes, cache, lexicon, speakers, strict gate |
| `tests/test_overflow_policy.py` | one end-to-end run per `--overflow` policy |
| `tests/test_robustness.py` | engine retries and probing, corrupt cache recovery, per-line fault isolation, atomic writes, export failure, clean exits, reproducible renders, two real offline voices |
| `tests/test_voice_metrics.py` | pitch tracking, distortion detection, pacing, and the benchmark script's output |
| `tests/test_hard_srt.py` | the hostile 5-minute SRT (`tests/fixtures/hard_5min.srt`), end to end |

## Notes and limits

- A blocked endpoint or a missing piper voice degrades the voice, never the timing;
  the report always says which engine spoke.
- `--overflow cut` (default) drops the remainder of an impossible line and reports
  `TRUNCATED(-Nms)`; `bleed` keeps it whole and can exceed the timeline.
- With `--deterministic` the timing is repeatable but the rhythm noise is gone; without
  it, line lengths move by a few percent between runs. Both are measured above.
- Per-line levelling matches RMS; the EBU R128 pass is on the finished mix.
- The music bed is looped and sidechain-ducked; it is not per-cue scoring.
- `--overflow compress` is capped at `--max-compress` (default 8x) and then trims what
  is still over; `bleed` exceeds the timeline by design.
- Naturalness is measured, not judged: register, pitch range, pacing, clipping and
  loudness are reported and regression-tested, but which voice sounds right to a
  listener is still a listening decision.
- One voice speaks one language: a mismatch produces silence, which is reported rather
  than hidden. Download the voice that matches the subtitles.
