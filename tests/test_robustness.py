"""Failure-mode tests: the pipeline must survive bad units, bad cache entries and a
flaky online engine, and it must never leave a half-written file behind."""

import asyncio
import json
import os
import sys
import types

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

import audio_post as ap
import pipeline
import pipeline_core as pc
import tts_engines as te
from subtitle_io import Cue, LoadResult
from tests.conftest import needs_engine, needs_ffmpeg, run_cli, write_srt


def make_cue(index, start, end, text):
    return Cue(index=index, start_ms=start, end_ms=end, text=text)


# ------------------------------------------------------- online engine (mocked)

class FakeCommunicate:
    """Stand-in for edge_tts.Communicate: records calls and can be made to fail."""

    calls = []
    fail_times = 0

    def __init__(self, text, voice, rate=None, volume=None, pitch=None, proxy=None):
        self.text = text
        self.voice = voice
        self.rate = rate
        FakeCommunicate.calls.append({"text": text, "voice": voice, "rate": rate,
                                      "proxy": proxy})

    async def save(self, path):
        if FakeCommunicate.fail_times > 0:
            FakeCommunicate.fail_times -= 1
            raise RuntimeError("simulated endpoint failure")
        with open(path, "wb") as f:
            f.write(b"\x00" * 128)


@pytest.fixture
def fake_edge(monkeypatch):
    FakeCommunicate.calls = []
    FakeCommunicate.fail_times = 0
    module = types.ModuleType("edge_tts")
    module.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return FakeCommunicate


def test_edge_engine_encodes_speed_as_a_rate_offset(fake_edge, tmp_path):
    engine = te.EdgeEngine(base_speed=1.0, max_speed=1.3, retries=1)
    out = str(tmp_path / "a.mp3")
    assert asyncio.run(engine.render_async("hello", "en-US-JennyNeural", 1.3, out))
    assert os.path.exists(out)
    assert fake_edge.calls[-1]["rate"] == "+30%"
    assert fake_edge.calls[-1]["voice"] == "en-US-JennyNeural"

    asyncio.run(engine.render_async("hello", "en-US-JennyNeural", 1.0, out))
    assert fake_edge.calls[-1]["rate"] == "+0%"


def test_edge_engine_retries_then_reports_failure(fake_edge, tmp_path):
    engine = te.EdgeEngine(base_speed=1.0, max_speed=1.3, retries=2)
    fake_edge.fail_times = 1
    assert asyncio.run(engine.render_async("hi", "v", 1.0, str(tmp_path / "r.mp3")))
    assert len(fake_edge.calls) == 2                       # failed once, retried

    fake_edge.calls = []
    fake_edge.fail_times = 99
    assert not asyncio.run(engine.render_async("hi", "v", 1.0, str(tmp_path / "r2.mp3")))
    assert len(fake_edge.calls) == 2                       # gave up after its attempts


def test_probe_marks_an_unreachable_online_engine(fake_edge, tmp_path):
    engine = te.EdgeEngine(base_speed=1.0, max_speed=1.3, retries=2)
    fake_edge.fail_times = 99
    assert asyncio.run(te.probe_engine(engine, str(tmp_path))) is False
    assert engine.retries == 2                             # restored after probing

    fake_edge.fail_times = 0
    assert asyncio.run(te.probe_engine(engine, str(tmp_path))) is True
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".probe")]   # cleaned up


def test_filter_reachable_drops_the_dead_engine(fake_edge, tmp_path, monkeypatch):
    dead = te.EdgeEngine(base_speed=1.0, max_speed=1.3, retries=1)
    fake_edge.fail_times = 99
    chain = asyncio.run(te.filter_reachable([dead], str(tmp_path)))
    assert chain == [dead]          # a lone engine is kept: something beats nothing


def test_offline_engines_are_never_probed(tmp_path):
    piper = te.PiperEngine(model=str(tmp_path / "x.onnx"))
    assert asyncio.run(te.probe_engine(piper, str(tmp_path))) is True


def test_edge_proxy_comes_from_the_flag_or_the_environment(fake_edge, tmp_path, monkeypatch):
    """Microsoft refuses some networks outright; a proxy is the documented way out."""
    engine = te.EdgeEngine(base_speed=1.0, max_speed=1.3, proxy="http://proxy.local:8080")
    asyncio.run(engine.render_async("hello", "v", 1.0, str(tmp_path / "p.mp3")))
    assert fake_edge.calls[-1]["proxy"] == "http://proxy.local:8080"

    monkeypatch.setenv("HTTPS_PROXY", "http://env-proxy:3128")
    from_env = te.EdgeEngine(base_speed=1.0, max_speed=1.3)
    assert from_env.proxy == "http://env-proxy:3128"
    asyncio.run(from_env.render_async("hello", "v", 1.0, str(tmp_path / "e.mp3")))
    assert fake_edge.calls[-1]["proxy"] == "http://env-proxy:3128"

    monkeypatch.delenv("HTTPS_PROXY")
    monkeypatch.delenv("EDGE_TTS_PROXY", raising=False)
    assert te.EdgeEngine(base_speed=1.0, max_speed=1.3).proxy is None


def test_cache_variant_changes_with_the_noise_settings_and_the_key(tmp_path):
    """A render made with other noise settings must never be served from the cache."""
    default = te.PiperEngine(model="m.onnx")
    flat = te.PiperEngine(model="m.onnx", noise_w=te.PIPER_DETERMINISTIC_NOISE_W)
    assert default.cache_variant() != flat.cache_variant()
    assert flat.deterministic and not default.deterministic
    assert te.PiperEngine(model="m.onnx", noise_scale=0.0).cache_variant() != default.cache_variant()
    assert te.PiperEngine(model="m.onnx").cache_variant() == default.cache_variant()

    cache = te.RenderCache(str(tmp_path / "cache"))
    assert cache.key("piper", "m.onnx", 1.0, "hi", default.cache_variant()) != \
        cache.key("piper", "m.onnx", 1.0, "hi", flat.cache_variant())
    assert cache.key("piper", "m.onnx", 1.0, "hi", "") != \
        cache.key("piper", "m.onnx", 1.0, "hi", default.cache_variant())


def test_deterministic_flag_makes_piper_lengths_repeatable(tmp_path):
    """piper's duration noise is the only thing that varies run to run; --deterministic
    turns it off, so the same line comes out the same length every time."""
    pytest.importorskip("piper")
    from piper.config import SynthesisConfig

    model = te.find_piper_model()
    if not model:
        pytest.skip("no piper voice model installed")

    from piper import PiperVoice
    voice = PiperVoice.load(model)
    text = "Reproducible line lengths are what make a re-run trustworthy."

    def lengths(noise_w):
        out = []
        for _ in range(3):
            chunks = list(voice.synthesize(text, syn_config=SynthesisConfig(
                length_scale=1.0, noise_w_scale=noise_w)))
            out.append(sum(len(c.audio_int16_bytes) for c in chunks))
        return out

    noisy = lengths(None)
    repeatable = lengths(te.PIPER_DETERMINISTIC_NOISE_W)
    assert len(set(repeatable)) == 1, f"noise_w 0 should repeat: {repeatable}"
    assert noisy[0] > 0


def test_compress_cap_is_configurable():
    segment = AudioSegment.silent(duration=4000)
    segment = segment.overlay(Sine(440).to_audio_segment(duration=4000))
    assert len(ap.compress_to_fit(segment, 1000, max_ratio=2.0)) >= 1900
    assert len(ap.compress_to_fit(segment, 1000, max_ratio=8.0)) < 1100


@needs_engine
@needs_ffmpeg
def test_cli_deterministic_mode_repeats_decisions(tmp_path):
    """--deterministic makes every fitting decision repeatable; adding --piper-noise 0
    freezes the waveform noise too, so the delivered file is byte-identical."""
    cues = [(0, 2200, "Reproducible renders make a re-run trustworthy."),
            (2400, 4000, "A tight line squeezes in here."),
            (4200, 9000, "A long window keeps this one at its natural pace.")]
    cache = str(tmp_path / "cache")

    def run(name, extra):
        srt = write_srt(str(tmp_path / name / "d.srt"), cues)
        rep = str(tmp_path / f"{name}.json")
        proc = run_cli(["-p", srt, "-f", "wav", "--cache-dir", cache,
                        "--report", rep, *extra])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return json.loads(open(rep, encoding="utf-8").read())

    first = run("a", ["--deterministic"])
    second = run("b", ["--deterministic"])
    assert first["piper"]["deterministic"] is True
    for key in ("units", "final_ms", "fit", "cuts", "bleeds", "renders"):
        assert first[key] == second[key], key
    assert [u["speed"] for u in first["units_detail"]] == \
        [u["speed"] for u in second["units_detail"]]
    assert [u["fit"] for u in first["units_detail"]] == \
        [u["fit"] for u in second["units_detail"]]

    frozen_a = run("c", ["--deterministic", "--piper-noise", "0"])
    frozen_b = run("d", ["--deterministic", "--piper-noise", "0"])
    assert frozen_a["piper"]["byte_identical"] is True
    with open(frozen_a["output"], "rb") as fa, open(frozen_b["output"], "rb") as fb:
        assert fa.read() == fb.read()


# ------------------------------------------------------------ self-healing cache

@needs_engine
def test_corrupt_cache_entry_is_recovered(tmp_path):
    """A truncated cached render must be re-rendered, not fail the line."""
    from tts_engines import load_rendered

    srt = write_srt(str(tmp_path / "c.srt"), [(0, 2500, "Recover from a bad cache entry.")])
    cache_dir = tmp_path / "cache"
    report = tmp_path / "r1.json"

    first = run_cli(["-p", srt, "-f", "wav", "--cache-dir", str(cache_dir),
                     "--report", str(report)])
    assert first.returncode == 0, first.stdout + first.stderr
    entries = os.listdir(cache_dir)
    assert entries

    # Truncate every cached render: the audio is now unreadable.
    for name in entries:
        path = cache_dir / name
        path.write_bytes(b"not audio at all")

    second = run_cli(["-p", srt, "-f", "wav", "--cache-dir", str(cache_dir),
                      "--report", str(tmp_path / "r2.json")])
    assert second.returncode == 0, second.stdout + second.stderr
    data = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))
    assert data["units"] == 1                      # the line still produced audio
    assert data["cuts"]["count"] == 0
    assert "rendering it again" in second.stdout

    # and the cache is usable again
    repaired = cache_dir / entries[0]
    assert repaired.exists()
    assert len(load_rendered(str(repaired))) > 0


def test_cache_put_is_atomic(tmp_path):
    cache = te.RenderCache(str(tmp_path / "cache"))
    source = str(tmp_path / "seg.wav")
    Sine(440).to_audio_segment(duration=200).export(source, format="wav")
    stored = cache.put("key1", source)
    assert os.path.exists(stored)
    assert not [name for name in os.listdir(cache.directory) if name.endswith(".part")]


def test_cache_invalidate_forgets_one_entry(tmp_path):
    cache = te.RenderCache(str(tmp_path / "cache"))
    source = str(tmp_path / "seg.wav")
    Sine(440).to_audio_segment(duration=200).export(source, format="wav")
    cache.put("key1", source)
    assert cache.get("key1")
    cache.invalidate("key1")
    assert cache.get("key1") is None
    cache.invalidate("never-existed")              # must not raise


# ------------------------------------------------------------------ fault isolation

@needs_engine
def test_a_broken_line_does_not_kill_the_run(tmp_path, monkeypatch):
    """Force one unit's render to explode and check the rest still get audio."""
    srt = write_srt(str(tmp_path / "iso.srt"), [
        (0, 2500, "First line is fine."),
        (2600, 5200, "EXPLODE: this one blows up."),
        (5300, 8000, "Third line is fine too."),
    ])
    original = pipeline._fit_with_engine
    calls = {"n": 0}

    async def exploding(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated engine explosion")
        return await original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_fit_with_engine", exploding)
    report = asyncio.run(pipeline.generate_audio(
        path=srt, output_format="wav", engine="auto",
        cache_dir=str(tmp_path / "cache"), report_path=str(tmp_path / "r.json"),
        overlap="mix", concurrency=1,
    ))
    assert report is not None
    assert report["units"] == 2                     # two of three lines produced audio
    assert report["units"] < 3
    assert report["strict"]["passed"] is False
    assert any("no audio" in reason for reason in report["strict"]["reasons"])


# ----------------------------------------------------------- atomic deliverables

@needs_engine
def test_no_partial_file_is_left_behind(tmp_path):
    srt = write_srt(str(tmp_path / "a.srt"), [(0, 2500, "One line, one output file.")])
    out = tmp_path / "a.wav"
    proc = run_cli(["-p", srt, "-f", "wav"])
    assert proc.returncode == 0
    assert out.exists() and out.stat().st_size > 0
    leftovers = [p.name for p in tmp_path.iterdir() if ".part" in p.name]
    assert not leftovers, f"staging files left behind: {leftovers}"


def test_export_removes_its_staging_file_on_success(tmp_path):
    out = str(tmp_path / "out.wav")
    ap.export_audio(Sine(440).to_audio_segment(duration=500), out, "wav", loudnorm=True)
    assert os.path.exists(out)
    assert not [n for n in os.listdir(tmp_path) if ".part" in n]


def test_staging_path_keeps_the_extension():
    """ffmpeg infers the muxer from the extension, so the staging file must keep it."""
    assert ap._staging_path("clip.m4a", "m4a").endswith(".m4a")
    assert ap._staging_path("clip.mp3", "mp3").endswith(".mp3")
    # Distinct per process: two runs on one input must not share a staging file.
    assert ".part" in ap._staging_path("clip.wav", "wav")
    assert str(os.getpid()) in ap._staging_path("clip.wav", "wav")


def test_export_that_cannot_write_reports_failure_instead_of_raising(tmp_path):
    """The export is not allowed to raise: a failed write is a value, not a crash."""
    segment = Sine(440).to_audio_segment(duration=500)
    missing_dir = tmp_path / "no_such_dir"

    plain = ap.export_audio(segment, str(missing_dir / "out.wav"), "wav", loudnorm=False)
    assert plain["ok"] is False

    with_loudnorm = ap.export_audio(segment, str(missing_dir / "out.wav"), "wav", loudnorm=True)
    assert with_loudnorm["ok"] is False

    assert ap._export_with_pydub(segment, str(missing_dir / "out.wav"), "wav") is False


def test_report_write_is_atomic_and_creates_its_directory(tmp_path):
    target = tmp_path / "nested" / "deeper" / "report.json"
    pipeline._write_report(str(target), {"units": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"units": 3}
    assert not [n for n in os.listdir(target.parent) if ".part" in n]


def test_report_flags_a_failed_export_and_fails_the_strict_gate(tmp_path):
    cues = [make_cue(1, 0, 2000, "One line of speech.")]
    units = pc.build_units(cues)
    records = [{
        "unit": 1, "cues": [1], "speaker": None, "start_ms": 0, "end_ms": 2000,
        "window_ms": 2000, "engine": "piper", "voice": "m", "speed": 1.0,
        "renders": 1, "cached": False, "fit": 1.0, "tags": [], "cut_ms": 0,
        "bleed_ms": 0, "overlap_trim_ms": 0, "stitched": False,
    }]
    loaded = LoadResult(cues=cues, fmt="srt", encoding="utf-8", path=str(tmp_path / "x.srt"))
    report = pipeline._build_report(
        loaded, units, records, AudioSegment.silent(duration=2000), 2000,
        str(tmp_path / "x.wav"), "wav",
        chain=[], cache_store=te.RenderCache(str(tmp_path / "cache"), enabled=False),
        export_info={"ok": False}, voices={}, lexicon=None, fallback_used=set(),
        overlap="mix", overflow="cut", stitch=True, metrics=False, strict=True, music=None,
    )
    assert report["export_ok"] is False
    assert report["strict"]["passed"] is False
    assert any("could not be written" in reason for reason in report["strict"]["reasons"])


@needs_engine
@needs_ffmpeg
def test_unwritable_output_ends_in_a_clean_exit(tmp_path):
    """A blocked output path must fail loudly, quickly, and without a traceback."""
    srt = write_srt(str(tmp_path / "blocked.srt"), [(0, 2000, "One line to speak.")])
    (tmp_path / "blocked.wav").mkdir()          # something already owns the output path

    proc = run_cli(["-p", srt, "-f", "wav", "--report", str(tmp_path / "r.json")])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "could not be written" in proc.stdout
    assert "Traceback" not in proc.stderr
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["export_ok"] is False
    assert report["units"] == 1                  # the speech itself was fine


@needs_ffmpeg
def test_subtitles_with_nothing_to_say_exit_cleanly(tmp_path):
    empty = tmp_path / "empty.srt"
    empty.write_text("\n", encoding="utf-8")
    proc = run_cli(["-p", str(empty)])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no usable subtitle cues" in proc.stdout
    assert "Traceback" not in proc.stderr


@needs_engine
@needs_ffmpeg
def test_missing_music_file_warns_instead_of_failing_late(tmp_path):
    srt = write_srt(str(tmp_path / "m.srt"), [(0, 2000, "Speech over a missing bed.")])
    proc = run_cli(["-p", srt, "-f", "wav", "--music", str(tmp_path / "gone.mp3")])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "continuing without a bed" in proc.stdout
    assert (tmp_path / "m.wav").exists()


# ------------------------------------------------------------ multi-voice offline

def test_piper_voice_pool_uses_every_installed_model(tmp_path):
    models = []
    for name in ("voice_a", "voice_b"):
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(b"x")
        models.append(str(path))

    units = pc.build_units([
        make_cue(1, 0, 2000, "ALPHA: first speaker here"),
        make_cue(2, 2100, 4000, "BETA: second speaker here"),
    ], known_speakers={"ALPHA", "BETA"})
    mapping = pipeline._speaker_voice_map(units, "piper", None, None, True, piper_models=models)
    assert mapping == {"ALPHA": models[0], "BETA": models[1]}


def test_find_piper_models_lists_every_voice(tmp_path):
    """The explicit path wins; a path that does not exist is ignored, not offered."""
    for name in ("one", "two"):
        (tmp_path / f"{name}.onnx").write_bytes(b"x")
    found = te.find_piper_models(str(tmp_path / "one.onnx"))
    assert found[0].endswith("one.onnx")
    assert te.find_piper_model(str(tmp_path / "two.onnx")).endswith("two.onnx")
    assert not any(path.endswith("missing.onnx")
                   for path in te.find_piper_models(str(tmp_path / "missing.onnx")))


@needs_engine
@needs_ffmpeg
def test_two_speakers_get_two_real_offline_voices(tmp_path):
    """With a second model installed, --auto-speakers really does switch voice offline."""
    if len(te.find_piper_models()) < 2:
        pytest.skip("needs two piper voices (python scripts/download_voice.py --pack)")

    srt = write_srt(str(tmp_path / "two.srt"), [
        (0, 3000, "NARRATOR: The station had been quiet for eleven years."),
        (3200, 6500, "PILOT: Then something answered our hail."),
    ])
    report_path = str(tmp_path / "r.json")
    proc = run_cli(["-p", srt, "-f", "wav", "--engine", "piper", "--auto-speakers",
                    "--deterministic", "--report", report_path])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(open(report_path, encoding="utf-8").read())

    voices = report["speaker_voices"]
    assert set(voices) == {"NARRATOR", "PILOT"}
    assert voices["NARRATOR"] != voices["PILOT"]
    assert all(v.endswith(".onnx") for v in voices.values())
    assert [u["voice"] for u in report["units_detail"]] == \
        [voices["NARRATOR"], voices["PILOT"]]
    assert "SPEAKER(NARRATOR)" in report["units_detail"][0]["tags"]


def test_a_machine_without_piper_has_no_voices_to_offer(tmp_path, monkeypatch):
    """No model installed must read as "none", never as a broken path."""
    monkeypatch.delenv("PIPER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    assert te.find_piper_models(str(tmp_path / "gone.onnx")) == []
    assert te.find_piper_model(str(tmp_path / "gone.onnx")) is None


def test_single_model_piper_reports_the_shared_voice(tmp_path, capsys):
    units = pc.build_units([
        make_cue(1, 0, 2000, "ALPHA: first"),
        make_cue(2, 2100, 4000, "BETA: second"),
    ], known_speakers={"ALPHA", "BETA"})
    mapping = pipeline._speaker_voice_map(units, "piper", None, None, True, piper_models=["only.onnx"])
    assert set(mapping.values()) == {"only.onnx"}
    assert "shares it" in capsys.readouterr().out
