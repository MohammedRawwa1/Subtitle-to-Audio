"""Download piper voice models into models/.

Piper is the offline neural engine: no API key, no network at render time, and far
more natural than the SAPI fallback. Voices live on HuggingFace; name any voice from
rhasspy/piper-voices, or use --pack to install a small multi-speaker set so
--auto-speakers can give each character a different offline voice.

    python scripts/download_voice.py                     # en_US-lessac-medium
    python scripts/download_voice.py en_US-ryan-high
    python scripts/download_voice.py --pack              # three distinct voices
    python scripts/download_voice.py --list              # what --pack installs

Every model file lands in models/*.onnx, which the pipeline picks up automatically:
more than one model means --auto-speakers has something to rotate.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
DEFAULT_VOICE = "en_US-lessac-medium"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One female, one male, one British: enough for a two-hander or a narrator plus two
# characters, and all "medium" quality so they are a fair match for each other.
PACK = ("en_US-lessac-medium", "en_US-ryan-medium", "en_GB-alba-medium")


def voice_urls(name: str) -> tuple[str, str]:
    """en_US-lessac-medium -> the /en/en_US/lessac/medium/ paths used on the hub."""
    try:
        locale, speaker, quality = name.split("-", 2)
    except ValueError:
        raise SystemExit(f"voice must look like en_US-lessac-medium, got {name!r}") from None
    folder = f"{BASE}/{locale.split('_')[0]}/{locale}/{speaker}/{quality}/{name}"
    return folder + ".onnx", folder + ".onnx.json"


def download(url: str, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"have {path}")
        return
    print(f"downloading {os.path.basename(path)} ...")
    tmp = f"{path}.{os.getpid()}.part"
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, path)
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")
        raise SystemExit(1) from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("voices", nargs="*", default=None,
                        help=f"piper voice name(s) (default: {DEFAULT_VOICE})")
    parser.add_argument("--pack", action="store_true",
                        help="install the multi-speaker set instead of a single voice")
    parser.add_argument("--list", action="store_true", help="show what --pack installs")
    parser.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "models"),
                        help="destination directory (default: project models/)")
    args = parser.parse_args(argv)

    if args.list:
        for name in PACK:
            print(name)
        return 0

    names = list(args.voices or ())
    if args.pack or not names:
        names = (list(PACK) if args.pack else [DEFAULT_VOICE])

    os.makedirs(args.dir, exist_ok=True)
    for name in names:
        for url in voice_urls(name):
            download(url, os.path.join(args.dir, os.path.basename(url)))

    installed = sorted(os.path.basename(p) for p in os.listdir(args.dir) if p.endswith(".onnx"))
    print(f"\n{len(installed)} voice(s) in {args.dir}: {', '.join(installed)}")
    if len(installed) > 1:
        print("More than one voice means --auto-speakers can give each speaker their own.")
    print("The pipeline picks up models/*.onnx automatically, or pass --piper-model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
