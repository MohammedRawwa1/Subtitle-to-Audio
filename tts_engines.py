"""TTS engines behind one interface, plus a persistent render cache.

Every engine speaks the same language: `speed` is a dimensionless multiplier where
1.0 is the engine's natural rate and higher is faster. That is what lets the rate
fitter work against any of them without special cases.

    edge_tts  neural, online  (best when reachable)   speed -> rate percent
    piper     neural, offline (best without network)  speed -> length_scale
    pyttsx3   SAPI, offline   (always available)      speed -> words per minute

`build_chain()` returns the engines to try, in order, for `auto` or an explicit
choice, and the pipeline falls back down that chain if a render fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import hashlib
import importlib.util
import os
import re
import shutil
import threading
import wave

from pydub import AudioSegment

__all__ = [
    "RenderCache",
    "build_chain",
    "describe_engines",
    "filter_reachable",
    "first_guess_speed",
    "probe_engine",
    "load_rendered",
    "parse_rate_pct",
    "PIPER_DETERMINISTIC_NOISE_W",
    "EDGE_VOICE_DEFAULT",
    "AUTO_VOICES",
    "pick_auto_voice",
    "detect_voice_gender",
    "find_piper_model",
    "find_piper_models",
    "detect_language",
    "order_piper_models",
]

CACHE_VERSION = 2              # bump to invalidate stale cached renders

# Piper's duration predictor is stochastic: its noise (``noise_w``) is what makes a line
# come out a few percent longer or shorter each time. Setting it to zero removes the
# randomness, so a render's length -- and therefore every fitting decision -- is
# repeatable. Measured on en_US-lessac-medium, the same line 3x:
#   noise_w 0.8 (default)  157440 / 155136 / 154880 samples   durations differ
#   noise_w 0.0            145408 / 145408 / 145408 samples   durations identical
#   noise_w 0.0 + noise 0.0                                  audio byte-identical
PIPER_DETERMINISTIC_NOISE_W = 0.0

# ------------------------------------------------------------------- edge_tts

EDGE_VOICE_DEFAULT = "en-US-JennyNeural"
ARABIC_EDGE_VOICE = "ar-AE-HamdanNeural"

AUTO_VOICES = {
    "female": ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-SarahNeural"],
    "male": ["en-US-GuyNeural", "en-US-BrianNeural", "en-US-DavisNeural"],
    "neutral": ["en-US-JennyNeural"],
}

_FEMALE_CUES = [
    "woman", "women", "female", "ladies", "girlfriend", "wife", "mom", "mother",
    "daughter", "grandma", "actress", "femme", "makeup", "goddess", "queens",
    "princess", "damsel",
]
_MALE_CUES = [
    "man", "men", "male", "guy", "boy", "boyfriend", "husband", "dad", "father",
    "son", "grandpa", "actor", "gentleman", "bros", "dude", "brother", "stallion",
]

# ---------------------------------------------------------------------- piper

# Calibrated on en_US-lessac-medium (22 kHz): 51.4 ms/char at length_scale 1.0 and
# duration ~ length_scale ** 0.65.
PIPER_MS_PER_CHAR = 51.4
PIPER_LS_EXPONENT = 0.65
PIPER_MIN_LENGTH_SCALE = 0.45
PIPER_DEFAULT_DIR = "models"

# ------------------------------------------------------------- calibrations

EDGE_MS_PER_CHAR = 67.0        # edge_tts at +0%
EDGE_MS_PER_CHAR_SLOPE = -52.0  # per +100% rate
PYTTSX3_MS_PER_CHAR = 56.0     # SAPI at its base WPM


def parse_rate_pct(rate) -> int:
    """Parse '+10%' / '-20' / '10' into a signed integer percentage."""
    try:
        match = re.match(r"^([+-]?)(\d+)%?$", str(rate).strip())
        if not match:
            return 0
        return (-1 if match.group(1) == "-" else 1) * int(match.group(2))
    except Exception:
        return 0


def detect_voice_gender(text: str) -> str:
    """Guess a preferred TTS gender from first-person and gendered cue words."""
    lowered = (text or "").lower()
    female = lowered.count("she") + lowered.count("her")
    male = lowered.count("he") + lowered.count("him") + lowered.count("his")
    for c in _FEMALE_CUES:
        female += lowered.count(c)
    for c in _MALE_CUES:
        male += lowered.count(c)
    if female > male and female > 0:
        return "female"
    if male > female and male > 0:
        return "male"
    return "neutral"


def pick_auto_voice(text: str, gender: str = None) -> str:
    """Best edge_tts ShortName for this text; Arabic script selects an Arabic voice."""
    try:
        if detect_language(text) == "ar":
            return ARABIC_EDGE_VOICE
        chosen = gender if gender in ("female", "male") else detect_voice_gender(text)
        candidates = AUTO_VOICES.get(chosen, AUTO_VOICES["neutral"])
        return candidates[0] if candidates else EDGE_VOICE_DEFAULT
    except Exception:
        return EDGE_VOICE_DEFAULT


# Writing systems -> the language code piper voices use. Latin (and anything not
# listed) returns None, which is treated as "prefer English".
SCRIPT_LANGUAGES = (
    (re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]"), "ar"),
    (re.compile(r"[\u0590-\u05ff]"), "he"),
    (re.compile(r"[\u0400-\u04ff]"), "ru"),
    (re.compile(r"[\u0370-\u03ff]"), "el"),
    (re.compile(r"[\u4e00-\u9fff]"), "zh"),
    (re.compile(r"[\u3040-\u30ff]"), "ja"),
    (re.compile(r"[\uac00-\ud7af\u1100-\u11ff]"), "ko"),
    (re.compile(r"[\u0e00-\u0e7f]"), "th"),
    (re.compile(r"[\u0900-\u097f]"), "hi"),
)


def detect_language(text: str, sample: int = 4000) -> str | None:
    """Guess the language from the writing system: None means Latin/unknown.

    Script is enough to pick a voice: an Arabic file must not be read by an English
    voice, and vice versa. It is a hint, never a hard requirement -- an explicit
    --piper-model always wins.
    """
    if not text:
        return None
    head = text[:sample]
    found = {code: len(pattern.findall(head)) for pattern, code in SCRIPT_LANGUAGES}
    found = {code: count for code, count in found.items() if count}
    return max(found, key=found.get) if found else None


def order_piper_models(models: list[str], preferred: str = "en") -> list[str]:
    """Voices for `preferred` first, the rest behind them as fallbacks."""
    prefix = f"{preferred.lower()}_"

    def key(path: str):
        return (0 if os.path.basename(path).lower().startswith(prefix) else 1, path)

    return sorted(models, key=key)


def find_piper_models(explicit: str = None, prefer_lang: str = None) -> list[str]:
    """Every usable piper voice: an explicit path first, then models/*.onnx.

    Discovered voices are ordered so ones matching `prefer_lang` (the subtitles'
    script) come first, then English, then the rest -- so installing a voice for
    another language cannot silently change the voice of an English file. More than
    one model means multi-voice is possible offline: the speaker pool is filled from
    these, so a two-hander can have two offline neural voices.
    """
    pinned: list[str] = []
    for candidate in (explicit, os.getenv("PIPER_MODEL")):
        if candidate and os.path.exists(candidate):
            resolved = os.path.abspath(candidate)
            if resolved not in pinned:
                pinned.append(resolved)

    discovered: list[str] = []
    for directory in (PIPER_DEFAULT_DIR, os.path.expanduser("~/.local/share/piper")):
        for found in glob.glob(os.path.join(directory, "**", "*.onnx"), recursive=True):
            resolved = os.path.abspath(found)
            if resolved not in pinned and resolved not in discovered:
                discovered.append(resolved)

    return pinned + order_piper_models(discovered, preferred=prefer_lang or "en")


def find_piper_model(explicit: str = None, prefer_lang: str = None) -> str | None:
    """The default piper voice (first of `find_piper_models`)."""
    models = find_piper_models(explicit, prefer_lang=prefer_lang)
    return models[0] if models else None


# ------------------------------------------------------------------- engines

class TTSEngine:
    """One text-to-speech backend, addressed through a dimensionless speed."""

    name = "base"
    label = "base"
    is_async = True          # False -> render() blocks and is run in a thread
    needs_network = False    # True -> probed before the run (installed != reachable)

    def __init__(self, *, base_speed: float = 1.0, max_speed: float = 1.3):
        self.base_speed = max(1e-6, float(base_speed))
        self.max_speed = max(self.base_speed, float(max_speed))

    # --- introspection -----------------------------------------------------
    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def cache_variant(self) -> str:
        """Settings that change the audio but are not in the cache key otherwise.

        Without this, a run with a different seed (or noise setting) could be served a
        render made with the previous one.
        """
        return ""

    def ms_per_char(self, speed: float) -> float:
        """First-guess milliseconds of speech per character at `speed`."""
        raise NotImplementedError

    def resolve_voice(self, voice_hint: str = None, gender: str = None) -> str:
        return voice_hint or ""

    # --- rendering ---------------------------------------------------------
    def render(self, text: str, voice: str, speed: float, out_path: str) -> bool:
        """Render `text` at `speed` into `out_path`. May block; may be a coroutine."""
        raise NotImplementedError

    async def render_async(self, text: str, voice: str, speed: float, out_path: str) -> bool:
        if self.is_async:
            return await self.render(text, voice, speed, out_path)
        return await asyncio.to_thread(self.render, text, voice, speed, out_path)


class EdgeEngine(TTSEngine):
    name = "edge_tts"
    label = "edge_tts (neural, online)"
    needs_network = True

    def __init__(self, *, base_speed: float = 1.0, max_speed: float = 1.3, retries: int = 2,
                 proxy: str = None):
        super().__init__(base_speed=base_speed, max_speed=max_speed)
        self.retries = max(1, int(retries))
        # Microsoft refuses the handshake from some networks/IP ranges outright. A
        # proxy is the way out, so it is honoured from the flag, then the environment.
        self.proxy = (proxy or os.getenv("EDGE_TTS_PROXY") or os.getenv("HTTPS_PROXY")
                      or os.getenv("https_proxy") or None)

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("edge_tts") is None:
            return False, "edge_tts is not installed"
        return True, "ok"

    def ms_per_char(self, speed: float) -> float:
        return max(12.0, EDGE_MS_PER_CHAR + EDGE_MS_PER_CHAR_SLOPE * (speed - 1.0))

    def resolve_voice(self, voice_hint: str = None, gender: str = None) -> str:
        return voice_hint or EDGE_VOICE_DEFAULT

    @staticmethod
    def _fmt(speed: float) -> str:
        return f"{int(round((speed - 1.0) * 100)):+d}%"

    async def render(self, text: str, voice: str, speed: float, out_path: str) -> bool:
        try:
            import edge_tts
        except Exception as e:
            print(f"  edge_tts import failed: {e}")
            return False
        backoff = 1.0
        for attempt in range(self.retries):
            try:
                comm = edge_tts.Communicate(text, voice, rate=self._fmt(speed),
                                            volume="+0%", pitch="+0Hz",
                                            proxy=self.proxy)
                await comm.save(out_path)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return True
            except Exception as e:
                print(f"  edge_tts render failed: {type(e).__name__}: {str(e)[:120]}")
                with contextlib.suppress(OSError):
                    os.remove(out_path)      # never leave a partial render behind
            if attempt < self.retries - 1:
                await asyncio.sleep(backoff * (attempt + 1))
        return False


class PiperEngine(TTSEngine):
    """Offline neural TTS (piper). One ONNX voice, loaded once per worker thread."""

    name = "piper"
    label = "piper (neural, offline)"
    is_async = False
    _local = threading.local()

    def __init__(self, *, model: str, base_speed: float = 1.0, max_speed: float = 1.3,
                 noise_scale: float = None, noise_w: float = None):
        natural_max = 1.0 / PIPER_MIN_LENGTH_SCALE
        super().__init__(base_speed=base_speed, max_speed=min(max_speed, natural_max))
        self.model = model
        self.noise_scale = noise_scale
        self.noise_w = noise_w

    @property
    def deterministic(self) -> bool:
        return self.noise_w == PIPER_DETERMINISTIC_NOISE_W

    def cache_variant(self) -> str:
        return f"noise={self.noise_scale};noise_w={self.noise_w}"

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("piper") is None:
            return False, "piper-tts is not installed (pip install piper-tts)"
        if not self.model or not os.path.exists(self.model):
            return False, ("no piper voice model found; pass --piper-model or put a "
                           "*.onnx voice in models/")
        return True, "ok"

    def ms_per_char(self, speed: float) -> float:
        return max(6.0, PIPER_MS_PER_CHAR * (self.base_speed / max(speed, 1e-6)) ** PIPER_LS_EXPONENT)

    def resolve_voice(self, voice_hint: str = None, gender: str = None) -> str:
        if voice_hint and os.path.exists(voice_hint):
            return os.path.abspath(voice_hint)
        return self.model

    def _voice_for_thread(self, model_path: str):
        """One PiperVoice per thread: onnxruntime sessions are safe to share, but a
        per-thread voice keeps the phonemizer state private and lets us render in
        parallel without a global lock."""
        cache = getattr(PiperEngine._local, "voices", None)
        if cache is None:
            cache = PiperEngine._local.voices = {}
        voice = cache.get(model_path)
        if voice is None:
            from piper import PiperVoice
            voice = PiperVoice.load(model_path)
            cache[model_path] = voice
        return voice

    def render(self, text: str, voice: str, speed: float, out_path: str) -> bool:
        try:
            from piper.config import SynthesisConfig
        except Exception as e:
            print(f"  piper import failed: {e}")
            return False

        model_path = voice or self.model
        length_scale = max(PIPER_MIN_LENGTH_SCALE, min(4.0, self.base_speed / max(speed, 1e-6)))
        tmp_wav = None
        try:
            pv = self._voice_for_thread(model_path)
            config = SynthesisConfig(length_scale=length_scale,
                                     noise_scale=self.noise_scale,
                                     noise_w_scale=self.noise_w)
            chunks = list(pv.synthesize(text, syn_config=config))
            if not chunks:
                return False
            tmp_wav = out_path + ".part"
            with wave.open(tmp_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(pv.config.sample_rate)
                for chunk in chunks:
                    w.writeframes(chunk.audio_int16_bytes)
            os.replace(tmp_wav, out_path)
            return os.path.getsize(out_path) > 0
        except Exception as e:
            print(f"  piper render failed: {type(e).__name__}: {str(e)[:160]}")
            return False
        finally:
            if tmp_wav:
                with contextlib.suppress(OSError):
                    os.remove(tmp_wav)  # only exists if the write above did not finish


class Pyttsx3Engine(TTSEngine):
    """Offline SAPI5 fallback. Single COM engine, so renders are serialized."""

    name = "pyttsx3"
    label = "pyttsx3 (SAPI5, offline)"
    is_async = False

    def __init__(self, *, base_wpm: int = 165, max_wpm: int = 280, base_speed: float = 1.0,
                 max_speed: float = 1.3):
        super().__init__(base_speed=base_speed, max_speed=min(max_speed, max_wpm / base_wpm))
        self.base_wpm = int(base_wpm)
        self.max_wpm = int(max_wpm)
        self._engine = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("pyttsx3") is None:
            return False, "pyttsx3 is not installed"
        return True, "ok"

    def ms_per_char(self, speed: float) -> float:
        return max(12.0, PYTTSX3_MS_PER_CHAR * self.base_speed / max(speed, 1e-6))

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
        return self._engine

    def resolve_voice(self, voice_hint: str = None, gender: str = None) -> str:
        wanted = (gender or "female").lower()
        try:
            engine = self._get_engine()
            voices = engine.getProperty("voices") or []
            for v in voices:
                blob = f"{(v.name or '').lower()} {(v.id or '').lower()}"
                if wanted == "male" and ("male" in blob or "david" in blob):
                    return v.id
                if wanted == "female" and ("female" in blob or "zira" in blob):
                    return v.id
            return voices[0].id if voices else ""
        except Exception as e:
            print(f"  pyttsx3 voice discovery failed: {e}")
            return ""

    def render(self, text: str, voice: str, speed: float, out_path: str) -> bool:
        wpm = max(self.base_wpm, min(self.max_wpm, int(round(self.base_wpm * speed))))
        with self._lock:            # one SAPI engine, one render at a time
            try:
                engine = self._get_engine()
                engine.setProperty("rate", wpm)
                if voice:
                    engine.setProperty("voice", voice)
                engine.save_to_file(text, out_path)
                engine.runAndWait()
                return os.path.exists(out_path) and os.path.getsize(out_path) > 0
            except Exception as e:
                print(f"  pyttsx3 render failed: {e}")
                self._engine = None
                return False


ENGINE_CLASSES = {
    "edge_tts": EdgeEngine,
    "piper": PiperEngine,
    "pyttsx3": Pyttsx3Engine,
}

# Preferred order for --engine auto: best quality first, always-available last.
AUTO_ORDER = ("edge_tts", "piper", "pyttsx3")


def build_chain(
    requested: str = "auto",
    *,
    base_speed: float = 1.0,
    max_speed: float = 1.3,
    edge_retries: int = 2,
    piper_model: str = None,
    base_wpm: int = 165,
    max_wpm: int = 280,
    edge_proxy: str = None,
    piper_noise: float = None,
    piper_noise_w: float = None,
    verbose: bool = True,
) -> list[TTSEngine]:
    """Engines to try, best first. `auto` keeps every available engine in the chain;
    an explicit choice goes first but keeps the others as a safety net so a run
    cannot end up in silence.

    `max_speed` caps the online engine only; see the note on `natural_ceiling`.
    """
    # `max_speed` is the ceiling for the online engine (a rate-offset range, which is
    # what --max-rate expresses). The offline engines are capped by their own natural
    # range instead -- piper by its length_scale floor, pyttsx3 by --max-wpm -- because
    # a percent-offset cap would artificially force cuts on lines they can simply speak.
    natural_ceiling = 10.0

    def make(name: str) -> TTSEngine:
        if name == "edge_tts":
            return EdgeEngine(base_speed=base_speed, max_speed=max_speed,
                              retries=edge_retries, proxy=edge_proxy)
        if name == "piper":
            return PiperEngine(model=piper_model, base_speed=base_speed,
                               max_speed=natural_ceiling,
                               noise_scale=piper_noise, noise_w=piper_noise_w)
        return Pyttsx3Engine(base_wpm=base_wpm, max_wpm=max_wpm,
                             base_speed=base_speed, max_speed=natural_ceiling)

    order = list(AUTO_ORDER)
    if requested != "auto":
        if requested not in ENGINE_CLASSES:
            raise ValueError(f"unknown engine: {requested}")
        order.remove(requested)
        order.insert(0, requested)

    chain, notes = [], []
    for name in order:
        engine = make(name)
        ok, reason = engine.available()
        notes.append(f"  engine {name:9s} {'available' if ok else 'unavailable'}: {reason}")
        if ok:
            chain.append(engine)
    if verbose:
        for note in notes:
            print(note)
    if not chain:
        raise RuntimeError(
            "no usable TTS engine: install edge-tts (online), piper-tts with a voice "
            "model (offline, neural), or pyttsx3 (offline)"
        )
    return chain


def describe_engines(chain) -> str:
    return " -> ".join(e.name for e in chain)


async def probe_engine(engine: TTSEngine, out_dir: str, timeout: float = 25.0) -> bool:
    """Is an online engine actually usable right now?

    Importing `edge_tts` proves nothing about a blocked endpoint, and finding out
    three lines into a run is too late: it costs renders and makes the strict gate
    report a degradation that is really just the network. So the engine is asked to
    say one short phrase, once, before the run starts.
    """
    if not engine.needs_network:
        return True

    saved_retries = getattr(engine, "retries", None)
    if saved_retries is not None:
        engine.retries = 1
    probe_path = os.path.join(out_dir, f".probe_{engine.name}")
    try:
        ok = await asyncio.wait_for(
            engine.render_async("Ready.", engine.resolve_voice(None, None), engine.base_speed, probe_path),
            timeout=timeout,
        )
    except Exception:
        ok = False
    finally:
        if saved_retries is not None:
            engine.retries = saved_retries
        for candidate in (probe_path, probe_path + ".mp3", probe_path + ".part"):
            try:
                os.remove(candidate)
            except OSError:
                pass
    return bool(ok)


async def filter_reachable(chain, out_dir: str):
    """Drop online engines that cannot be reached, keeping the offline ones."""
    if len(chain) < 2:
        return chain
    reachable = []
    for engine in chain:
        if await probe_engine(engine, out_dir):
            reachable.append(engine)
        else:
            hint = ""
            if engine.name == "edge_tts" and not getattr(engine, "proxy", None):
                hint = ("  (if a proxy or VPN is available, pass --edge-proxy URL "
                        "or set HTTPS_PROXY)")
            print(f"  {engine.name}: unreachable (endpoint refused); skipping it for this "
                  f"run{hint}")
    return reachable or chain


def first_guess_speed(engine: TTSEngine, text: str, window_ms: float) -> float:
    """Cheapest starting point for the rate fitter: speak at natural speed, scale by
    how far over the window the estimated speech would land."""
    if window_ms <= 0 or not text:
        return engine.base_speed
    natural_ms = engine.ms_per_char(engine.base_speed) * len(text)
    if natural_ms <= window_ms:
        return engine.base_speed
    guess = engine.base_speed * (natural_ms / window_ms)
    return max(engine.base_speed, min(engine.max_speed, guess))


# --------------------------------------------------------------------- cache

class RenderCache:
    """Content-addressed cache of rendered lines, so re-runs and long jobs resume.

    The key covers the engine, voice, speed, text and the engine's own variant
    (`cache_variant()`: piper's seed and noise settings), so a changed setting can
    never serve stale audio. A miss copies the freshly rendered file in; a hit skips
    TTS entirely, which is what makes iterating on a 2-hour file practical.
    """

    def __init__(self, directory: str = None, enabled: bool = True):
        self.enabled = enabled
        self.directory = os.path.abspath(directory or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".tts_cache"))
        self.hits = 0
        self.misses = 0
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except OSError:
                self.enabled = False

    def key(self, engine: str, voice: str, speed: float, text: str, variant: str = "") -> str:
        blob = f"{CACHE_VERSION}|{engine}|{voice}|{variant}|{speed:.6f}|{text}".encode()
        return hashlib.sha256(blob).hexdigest()[:32]

    def _find(self, key: str) -> str | None:
        for ext in (".wav", ".mp3"):
            path = os.path.join(self.directory, key + ext)
            try:
                if os.path.getsize(path) > 0:
                    return path
            except OSError:
                continue                 # missing, locked or unreadable: treat as a miss
        return None

    def get(self, key: str) -> str | None:
        if not self.enabled:
            return None
        path = self._find(key)
        if path:
            self.hits += 1
        return path

    def put(self, key: str, rendered_path: str) -> str:
        """Copy a fresh render into the cache (atomically) and return its path.

        Written through a temp file and renamed, so an interrupted run can never
        leave a half-written entry behind for the next run to trip over.
        """
        if not self.enabled:
            return rendered_path
        ext = ".mp3" if rendered_path.lower().endswith(".mp3") else ".wav"
        target = os.path.join(self.directory, key + ext)
        try:
            if os.path.getsize(target) > 0:
                return target
        except OSError:
            pass
        try:
            staging = f"{target}.{os.getpid()}.part"
            shutil.copyfile(rendered_path, staging)
            os.replace(staging, target)
            return target
        except OSError:
            return rendered_path

    def invalidate(self, key: str) -> None:
        """Forget one entry (used when a cached file turns out to be unreadable)."""
        for ext in (".wav", ".mp3"):
            try:
                os.remove(os.path.join(self.directory, key + ext))
            except OSError:
                pass

    def record_miss(self) -> None:
        self.misses += 1

    @property
    def stats(self) -> dict:
        return {"enabled": self.enabled, "directory": self.directory,
                "hits": self.hits, "misses": self.misses}

    def clear(self) -> int:
        removed = 0
        if os.path.isdir(self.directory):
            for path in glob.glob(os.path.join(self.directory, "*")):
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        return removed

    def remove_directory(self) -> bool:
        """Remove the cache directory after a completed delivery."""
        if not os.path.isdir(self.directory):
            return False
        self.clear()
        os.rmdir(self.directory)
        return True


def load_rendered(path: str) -> AudioSegment:
    """Read a rendered line (cached or fresh) back as audio."""
    return AudioSegment.from_mp3(path) if path.lower().endswith(".mp3") else AudioSegment.from_wav(path)
