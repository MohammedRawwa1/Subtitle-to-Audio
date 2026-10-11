"""Unit tests for the pure core, audio post-processing and the engine layer."""

import os
import re
import subprocess

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import audio_post as ap
import pipeline_core as pc
import tts_engines as te
from subtitle_io import apply_lexicon
from tests.conftest import needs_ffmpeg


def cue(index, start, end, text):
    from subtitle_io import Cue
    return Cue(index=index, start_ms=start, end_ms=end, text=text)


def groups(units):
    return [(u.cues[0].index, u.cues[-1].index) for u in units]


# ------------------------------------------------------------ cue stitching

def test_continuation_is_stitched():
    units = pc.build_units([
        cue(1, 0, 2000, "We can still make the crossing"),
        cue(2, 2200, 4000, "if nobody panics before dawn."),
    ])
    assert groups(units) == [(1, 2)]
    assert units[0].stitched
    assert units[0].text == "We can still make the crossing if nobody panics before dawn."
    assert units[0].window_ms == 4000          # covers both cues, including the gap


def test_sentence_end_never_stitched():
    cues = [cue(1, 0, 1000, "Shields up."), cue(2, 1000, 2000, "Come about, now.")]
    assert groups(pc.build_units(cues)) == [(1, 1), (2, 2)]


def test_large_gap_is_not_stitched():
    cues = [cue(1, 0, 1000, "no punctuation here"), cue(2, 5000, 6000, "continues later")]
    assert groups(pc.build_units(cues)) == [(1, 1), (2, 2)]


def test_rapid_fire_caps_chain_only_back_to_back():
    cues = [cue(1, 0, 1000, "First line"), cue(2, 1000, 2000, "Second line")]
    assert groups(pc.build_units(cues)) == [(1, 2)]

    cues = [cue(1, 0, 1000, "First line"), cue(2, 1400, 2400, "Second line")]
    assert groups(pc.build_units(cues)) == [(1, 1), (2, 2)]

    cues = [cue(1, 0, 1000, "first line"), cue(2, 1400, 2400, "second line")]
    assert groups(pc.build_units(cues)) == [(1, 2)]


def test_speaker_label_starts_a_new_unit():
    cues = [
        cue(1, 0, 1000, "I told you it was him"),
        cue(2, 1100, 2200, "DETECTIVE: that is not what"),
        cue(3, 2300, 3000, "the evidence says."),
    ]
    assert groups(pc.build_units(cues)) == [(1, 1), (2, 3)]


def test_hyphenated_word_is_healed_across_cues():
    units = pc.build_units([cue(1, 0, 1000, "The door is open-"), cue(2, 1000, 2000, "ing slowly.")])
    assert units[0].text == "The door is opening slowly."


def test_stitch_limits_are_respected():
    cues = [cue(i, i * 1000, i * 1000 + 900, "and then") for i in range(1, 8)]
    units = pc.build_units(cues)
    assert all(len(u.cues) <= pc.STITCH_MAX_CUES for u in units)
    assert sum(len(u.cues) for u in units) == len(cues)

    long_cues = [
        cue(1, 0, 9000, "a very long line with no end marker"),
        cue(2, 9000, 20000, "and another long line with no end marker"),
    ]
    assert groups(pc.build_units(long_cues)) == [(1, 1), (2, 2)]


def test_stitch_disabled_keeps_every_cue_separate():
    cues = [cue(1, 0, 2000, "no punctuation"), cue(2, 2000, 4000, "continues here")]
    assert groups(pc.build_units(cues, stitch=False)) == [(1, 1), (2, 2)]


def test_every_cue_is_covered_exactly_once():
    cues = [
        cue(1, 0, 1000, "one"), cue(2, 1000, 2000, "two."), cue(3, 2100, 3000, "three"),
        cue(4, 3100, 4000, "four"), cue(5, 9000, 9500, "five."),
    ]
    units = pc.build_units(cues)
    assert [c.index for u in units for c in u.cues] == [c.index for c in cues]


# ------------------------------------------------------------------ speakers

def test_speaker_detection_requires_a_known_or_auto_label():
    assert pc.speaker_of("NARRATOR: the end is near") == ""
    assert pc.speaker_of("NARRATOR: the end is near", auto=True) == "NARRATOR"
    assert pc.speaker_of("Jenny: hello there", known={"JENNY"}) == "JENNY"
    assert pc.speaker_of("Note: this matters") == ""


def test_speaker_label_is_not_spoken():
    unit = pc.build_units([cue(1, 0, 2000, "JENNY: the reactor is holding")],
                          known_speakers={"JENNY"})[0]
    assert unit.speaker == "JENNY"
    assert unit.speak == "the reactor is holding"
    assert unit.text == "JENNY: the reactor is holding"


def test_unknown_label_is_left_in_the_speech():
    unit = pc.build_units([cue(1, 0, 2000, "Note: this matters")])[0]
    assert unit.speaker == ""
    assert unit.speak == "Note: this matters"


def test_speakers_never_stitch_into_each_other():
    cues = [
        cue(1, 0, 1000, "The relay is quiet"),
        cue(2, 1100, 2200, "COMMS: and holding steady"),
    ]
    units = pc.build_units(cues, known_speakers={"COMMS"})
    assert groups(units) == [(1, 1), (2, 2)]


# ------------------------------------------------------------- overlap policy

def test_overlap_mix_leaves_units_alone():
    cues = [cue(1, 0, 3000, "Two contacts."), cue(2, 800, 3800, "Not answering.")]
    units, tags = pc.apply_overlap_policy(pc.build_units(cues), "mix", 3800)
    assert [u.place_ms for u in units] == [0, 800]
    assert all(u.limit_ms is None for u in units)
    assert tags == {}


def test_overlap_trim_shortens_the_earlier_unit():
    cues = [cue(1, 0, 3000, "Two contacts."), cue(2, 800, 3800, "Not answering.")]
    units, tags = pc.apply_overlap_policy(pc.build_units(cues), "trim", 3800)
    assert units[0].limit_ms == 800
    assert tags == {0: "OVERLAP-TRIM"}


def test_overlap_shift_moves_the_later_unit():
    cues = [cue(1, 0, 3000, "Two contacts."), cue(2, 800, 3800, "Not answering.")]
    units, tags = pc.apply_overlap_policy(pc.build_units(cues), "shift", 3800)
    assert units[0].start_ms == 0
    assert units[1].start_ms == 3000 and units[1].end_ms == 6000   # window length kept
    assert tags == {1: "SHIFT(+2200ms)"}


# ------------------------------------------------------------- overflow rules

@pytest.mark.parametrize("overrun, policy, expected", [
    (-10, "cut", "keep"),
    (0, "bleed", "keep"),
    (39, "compress", "keep"),          # inside the inaudible epsilon
    (41, "cut", "cut"),
    (500, "bleed", "bleed"),
    (500, "compress", "compress"),
    (500, "nonsense", "cut"),
])
def test_decide_overflow(overrun, policy, expected):
    assert pc.decide_overflow(overrun, policy) == expected


def test_format_timestamp():
    assert pc.format_timestamp(300_002) == "00:05:00,002"
    assert pc.format_timestamp(0) == "00:00:00,000"


# ---------------------------------------------------------------- rate fitting

def test_curve_k_fits_two_points():
    assert pc.curve_k(1.0, 2000, 1.3, 800) == pytest.approx(3.49, abs=0.05)
    assert pc.curve_k(0, 1, 0, 1) == 2.0


def test_solve_rate_steps_up_when_overrunning():
    nxt = pc.solve_rate([(1.0, 3000)], 1000, 0.05, 1.3)
    assert nxt is not None and 1.0 < nxt <= 1.3


def test_solve_rate_stops_when_already_fitting():
    assert pc.solve_rate([(1.0, 500)], 1000, 0.05, 1.3) is None
    assert pc.solve_rate([(1.0, 1400), (1.3, 950)], 1000, 0.05, 1.3) is None


def test_solve_rate_walks_back_from_an_overlong_take():
    assert pc.solve_rate([(1.0, 2000), (1.3, 800)], 1000, 0.05, 1.3) == pytest.approx(1.22, abs=0.02)


# ------------------------------------------------------------------- engines

def test_engines_share_the_speed_abstraction():
    edge = te.EdgeEngine(base_speed=1.0, max_speed=1.3)
    piper = te.PiperEngine(model="x.onnx", base_speed=1.0, max_speed=1.3)
    sapi = te.Pyttsx3Engine(base_wpm=165, max_wpm=280, base_speed=1.0, max_speed=1.3)
    for engine in (edge, piper, sapi):
        assert engine.ms_per_char(1.0) > engine.ms_per_char(engine.max_speed)
        assert engine.base_speed == 1.0
        assert engine.max_speed >= engine.base_speed


def test_piper_speed_ceiling_is_bounded_by_its_length_scale():
    piper = te.PiperEngine(model="x.onnx", base_speed=1.0, max_speed=10.0)
    assert piper.max_speed == pytest.approx(1 / te.PIPER_MIN_LENGTH_SCALE)


def test_pyttsx3_speed_ceiling_follows_its_wpm_range():
    sapi = te.Pyttsx3Engine(base_wpm=165, max_wpm=280, base_speed=1.0, max_speed=10.0)
    assert sapi.max_speed == pytest.approx(280 / 165)


@pytest.mark.parametrize("engine_name", ["edge_tts", "piper", "pyttsx3"])
def test_first_guess_speed_never_slows_and_respects_the_ceiling(engine_name):
    engine = {
        "edge_tts": lambda: te.EdgeEngine(base_speed=1.0, max_speed=1.3),
        "piper": lambda: te.PiperEngine(model="x.onnx", base_speed=1.0, max_speed=1.3),
        "pyttsx3": lambda: te.Pyttsx3Engine(base_wpm=165, max_wpm=280, base_speed=1.0, max_speed=1.3),
    }[engine_name]()

    assert te.first_guess_speed(engine, "Short.", 10_000) == engine.base_speed
    guess = te.first_guess_speed(engine, "word " * 60, 3_000)
    assert engine.base_speed < guess <= engine.max_speed


@pytest.mark.parametrize("raw, expected", [("+10%", 10), ("-20", -20), ("0%", 0), ("junk", 0)])
def test_parse_rate_pct(raw, expected):
    assert te.parse_rate_pct(raw) == expected


def test_auto_voice_pool_is_ordered_by_quality():
    assert te.pick_auto_voice("She walked into the room", "female") in te.AUTO_VOICES["female"]
    assert te.pick_auto_voice("He said nothing at all", "male") in te.AUTO_VOICES["male"]
    assert te.pick_auto_voice("", None) == te.EDGE_VOICE_DEFAULT or te.pick_auto_voice("", None)


def test_build_chain_keeps_the_preferred_order():
    try:
        chain = te.build_chain("auto", verbose=False)
    except RuntimeError:
        pytest.skip("no TTS engine available")
    names = [e.name for e in chain]
    assert names == [n for n in te.AUTO_ORDER if n in names]


def test_offline_engines_keep_their_own_ceiling():
    """--max-rate is an edge_tts rate offset; it must not cap piper/pyttsx3, which can
    simply speak faster instead of forcing a cut."""
    try:
        chain = te.build_chain("auto", max_speed=1.3, piper_model=te.find_piper_model(),
                               verbose=False)
    except RuntimeError:
        pytest.skip("no TTS engine available")
    for engine in chain:
        if engine.name == "piper":
            assert engine.max_speed > 1.5
        elif engine.name == "pyttsx3":
            assert engine.max_speed > 1.5
        else:
            assert engine.max_speed == pytest.approx(1.3)


def test_explicit_engine_goes_first_when_available():
    try:
        chain = te.build_chain("piper", piper_model=te.find_piper_model(), verbose=False)
    except RuntimeError:
        pytest.skip("no TTS engine available")
    if any(e.name == "piper" for e in chain):
        assert chain[0].name == "piper"


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError):
        te.build_chain("nope", verbose=False)


# ------------------------------------------------------------------ the cache

def test_cache_roundtrip_and_key_isolation(tmp_path):
    cache = te.RenderCache(str(tmp_path / "cache"))
    segment = Sine(440).to_audio_segment(duration=300)
    source = str(tmp_path / "seg.wav")
    segment.export(source, format="wav")

    key = cache.key("piper", "voice.onnx", 1.0, "hello there")
    assert cache.get(key) is None
    cache.record_miss()
    stored = cache.put(key, source)
    assert os.path.exists(stored)
    assert cache.get(stored and key) == stored

    assert cache.key("piper", "voice.onnx", 1.0, "hello there") == key       # stable
    assert cache.key("piper", "voice.onnx", 1.1, "hello there") != key       # speed matters
    assert cache.key("edge_tts", "voice.onnx", 1.0, "hello there") != key    # engine matters
    assert cache.key("piper", "other.onnx", 1.0, "hello there") != key       # voice matters
    assert cache.key("piper", "voice.onnx", 1.0, "different") != key         # text matters
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1


def test_cache_can_be_disabled(tmp_path):
    cache = te.RenderCache(str(tmp_path / "cache"), enabled=False)
    assert cache.get("abc") is None
    assert cache.put("abc", "/nope.wav") == "/nope.wav"
    assert cache.stats["enabled"] is False


def test_cache_clear(tmp_path):
    cache = te.RenderCache(str(tmp_path / "cache"))
    cache.put("abc", _tiny_wav(tmp_path))
    assert cache.clear() == 1
    assert cache.get("abc") is None


def _tiny_wav(tmp_path):
    path = str(tmp_path / "tiny.wav")
    Sine(440).to_audio_segment(duration=100).export(path, format="wav")
    return path


# -------------------------------------------------------------------- lexicon

@pytest.mark.parametrize("text, lexicon, expected", [
    ("the SQL server is down", {"SQL": "sequel"}, "the sequel server is down"),
    ("nginx and NGINX", {"nginx": "engine x"}, "engine x and engine x"),
    ("Kubernetes cluster", {"Kubernetes": "koober-net-eez"}, "koober-net-eez cluster"),
    ("MongoDB is fine", {"Mongo DB": "mongo d b"}, "MongoDB is fine"),
    ("SQLite is not SQL", {"SQL": "sequel"}, "SQLite is not sequel"),
    ("no rules here", {}, "no rules here"),
    ("hyphen-heavy", {"heavy": "HEAVY"}, "hyphen-heavy"),
])
def test_apply_lexicon(text, lexicon, expected):
    assert apply_lexicon(text, lexicon) == expected


def test_lexicon_prefers_the_longest_match():
    assert apply_lexicon("New York City", {"New York": "new york", "New": "brand new"}) == "new york City"


# ------------------------------------------------------------------ levelling

def test_level_segment_lifts_a_quiet_line():
    quiet = Sine(440).to_audio_segment(duration=600).apply_gain(-25)
    before = quiet.dBFS
    after = ap.level_segment(quiet).dBFS
    assert before < -24
    assert abs(after - ap.TARGET_SEGMENT_DBFS) <= 1.5
    assert after > before


def test_level_segment_attenuates_a_loud_line():
    loud = Sine(440).to_audio_segment(duration=600)
    assert ap.level_segment(loud).dBFS < loud.dBFS


def test_level_segment_caps_the_gain():
    very_quiet = Sine(440).to_audio_segment(duration=600).apply_gain(-40)
    assert ap.level_segment(very_quiet).dBFS <= very_quiet.dBFS + ap.MAX_LEVEL_GAIN_DB + 0.5


def test_level_segment_respects_the_peak_ceiling():
    """A line that is quiet on average but clips must not be pushed further up."""
    spiky = Sine(440).to_audio_segment(duration=600).apply_gain(-3)
    levelled = ap.level_segment(spiky)
    assert levelled.max_dBFS <= ap.PEAK_CEILING_DBFS + 0.5


def test_level_segment_leaves_silence_alone():
    assert len(ap.level_segment(AudioSegment.silent(duration=500))) == 500


# ----------------------------------------------------------- ffmpeg-dependent

@needs_ffmpeg
def test_trim_trailing_silence_shortens_the_clip():
    spoken = Sine(440).to_audio_segment(duration=400)
    padded = AudioSegment.silent(duration=200) + spoken + AudioSegment.silent(duration=600)
    assert len(ap.trim_trailing_silence(padded)) < len(padded)


@needs_ffmpeg
def test_safety_atempo_is_bounded():
    seg = Sine(440).to_audio_segment(duration=1000)
    stretched = ap.stretch_audio(seg, 300, max_speed=1.25)
    assert len(stretched) == pytest.approx(800, abs=60)
    assert len(stretched) > 300


@needs_ffmpeg
def test_stretch_is_a_no_op_without_a_cap():
    seg = Sine(440).to_audio_segment(duration=1000)
    assert len(ap.stretch_audio(seg, 300, max_speed=None)) == 1000
    assert len(ap.stretch_audio(seg, 2000, max_speed=1.25)) == 1000


@needs_ffmpeg
def test_compress_to_fit_actually_fits():
    seg = Sine(440).to_audio_segment(duration=4000)
    assert len(ap.compress_to_fit(seg, 1000)) == pytest.approx(1000, abs=40)


@needs_ffmpeg
def test_compress_to_fit_respects_its_ratio_ceiling():
    seg = Sine(440).to_audio_segment(duration=8000)
    assert len(ap.compress_to_fit(seg, 500, max_ratio=2.0)) == pytest.approx(4000, abs=60)


@needs_ffmpeg
def test_compress_to_fit_leaves_fitting_audio_alone():
    seg = Sine(440).to_audio_segment(duration=500)
    assert len(ap.compress_to_fit(seg, 1000)) == 500


def _integrated_lufs(path):
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True, timeout=300,
    )
    values = re.findall(r"I:\s*(-?\d+\.\d+)\s*LUFS", proc.stderr)
    assert values, proc.stderr[-400:]
    return float(values[-1])


@needs_ffmpeg
def test_export_hits_the_loudness_target_two_pass(tmp_path):
    out = str(tmp_path / "loud.wav")
    quiet = Sine(440).to_audio_segment(duration=3000).apply_gain(-30)
    info = ap.export_audio(quiet, out, "wav", loudnorm=True)
    assert info["loudnorm"] is True
    assert abs(_integrated_lufs(out) - ap.LOUDNESS_TARGET["I"]) <= 1.0


@needs_ffmpeg
def test_export_can_skip_loudness(tmp_path):
    out = str(tmp_path / "raw.wav")
    quiet = Sine(440).to_audio_segment(duration=1500).apply_gain(-30)
    info = ap.export_audio(quiet, out, "wav", loudnorm=False)
    assert info["loudnorm"] is False
    assert _integrated_lufs(out) < -25.0


@needs_ffmpeg
def test_measure_loudness_reports_the_first_pass(tmp_path):
    path = str(tmp_path / "tone.wav")
    Sine(440).to_audio_segment(duration=1500).apply_gain(-20).export(path, format="wav")
    measured = ap.measure_loudness(path)
    assert measured and measured["input_i"] < 0
    assert "input_thresh" in measured and "target_offset" in measured


@needs_ffmpeg
def test_music_bed_is_ducked_under_the_voice(tmp_path):
    """The bed must sit lower while speech plays than in the gaps.

    The trigger (200 Hz) and the bed (5 kHz) are far apart so the bed level can be
    measured alone with a high-pass filter, without speech contaminating it.
    """
    bed = Sine(5000).to_audio_segment(duration=4000).apply_gain(-12)
    music = str(tmp_path / "music.wav")
    bed.export(music, format="wav")

    speech = AudioSegment.silent(duration=4000).overlay(
        Sine(200).to_audio_segment(duration=1500), position=0)
    speech_wav = str(tmp_path / "speech.wav")
    speech.export(speech_wav, format="wav")

    mixed = str(tmp_path / "mixed.wav")
    assert ap.mix_music(speech_wav, music, mixed, volume=0.6, fade_ms=0)
    assert len(AudioSegment.from_wav(mixed)) == pytest.approx(4000, abs=60)

    hp = str(tmp_path / "hp.wav")
    subprocess.run(["ffmpeg", "-y", "-i", mixed, "-af", "highpass=f=4000", hp],
                   capture_output=True, check=True, timeout=300)
    bed_only = AudioSegment.from_wav(hp)
    during_voice = bed_only[200:1300].dBFS
    in_the_gap = bed_only[2500:3800].dBFS
    assert in_the_gap - during_voice > 3.0, f"ducking too weak: {during_voice} vs {in_the_gap}"


@needs_ffmpeg
def test_export_pins_a_sane_sample_rate(tmp_path):
    """loudnorm resamples to 192 kHz internally; the export must not inherit it."""
    out = str(tmp_path / "rate.wav")
    ap.export_audio(Sine(440).to_audio_segment(duration=1000).apply_gain(-25), out, "wav")
    assert AudioSegment.from_wav(out).frame_rate == 48000


@pytest.mark.parametrize("name", ["wav", "mp3", "m4a", "ogg"])
def test_formats_are_declared_consistently(name):
    assert ap.FORMATS[name]["ext"] == f".{name}"
    assert name in ap.ENCODER_ARGS
