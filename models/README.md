# Piper voice models

Voice model binaries are intentionally not committed. They are large and have
voice-specific licenses; review the license for a voice before redistributing it.
The application discovers `*.onnx` files in this directory automatically.

From the repository root, install the default English voice:

```powershell
python scripts/download_voice.py
```

Or download the bundled multi-speaker selection (three voices):

```powershell
python scripts/download_voice.py --pack
```

The downloader writes both each `.onnx` model and its `.onnx.json` configuration
here, even when invoked from another working directory. To install elsewhere, pass
`--dir PATH`.
