# Setup guide

## Before you start

Use a separate Python environment for each version. Python 3.12 was used to check the source during import. The original project did not record exact package versions, so the dependency lists are a starting point rather than a tested package lock.

You need a working microphone and speakers. The first run downloads speech models. Mark 2 also downloads the TEN turn detection model, which may need substantial memory and time to load.

In LM Studio, load a chat model and start its local server on port `1234`. Each assistant sends requests to `http://localhost:1234/v1/chat/completions` using the model name `local-model`. If your server needs the loaded model's exact name, change the `model` value in the assistant file to match it.

## First version

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python voice_assistant.py
```

On Windows, use `py -3 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead of the first two commands. You may also need system audio libraries and FFmpeg, depending on your platform and speech package installation.

This version records five seconds of audio at a time. Wait for its listening message before speaking.

## Parakeet experiment

From the repository root, in a separate environment:

```sh
python -m pip install -r requirements-parakeet.txt
python parakeet_stt.py
```

This file uses `nvidia/parakeet-ctc-1.1b` and `latishab/turnsense`. Their compatibility with the installed Transformers and ONNX packages has not been verified during import. It is preserved as an experiment.

## Mark 2

Use macOS, because this assistant reads replies with the built-in `say` command. From a separate environment in the repository root:

```sh
python -m pip install -r mark_2/requirements.txt
python mark_2/voice_assistant_mark_2.py
```

The assistant uses RealtimeSTT and `TEN-framework/TEN_Turn_Detection`. It checks for a completed turn after roughly 300 milliseconds without a new text update, and stops waiting after 1.5 seconds. Some older source comments mention different timings; the settings at the top of the file control the actual values.

The other files in `mark_2/` are separate Apple speech experiments and permission helpers. They are not used by `voice_assistant_mark_2.py`. The Python Apple speech helper needs these extra macOS packages:

```sh
python -m pip install pyobjc-framework-Cocoa pyobjc-framework-Speech pyobjc-framework-AVFoundation
```

The original compiled helpers are excluded. Their Swift sources and the permission app's settings are included as `SpeechAuth-Info.plist`.

## Mark 3

Use macOS with Xcode Command Line Tools installed. From a separate environment in the repository root:

```sh
python -m pip install -r mark_3/requirements.txt
sh mark_3/build_speech_app.sh
python mark_3/voice_assistant_mark_3.py
```

The build script compiles the supplied Swift source, copies its settings, and signs the helper locally. It creates `mark_3/stt_binary.app`, which stays out of Git. Allow the requested speech recognition and microphone access yourself when macOS asks.

Python sends recorded audio to the Apple speech helper through `/tmp/stt.sock`. The helper writes a log to `/tmp/stt_binary.log`. Apple speech uses on-device recognition when the system supports it; availability depends on your Mac.

Mark 3 detects a pause of about half a second, then starts a short acknowledgement while waiting for the reply. Its Whisper fallback code is present but disabled in the supplied main flow.

## If something does not work

- **No model reply:** check that LM Studio is running on port `1234`, a chat model is loaded, and the model name matches your server.
- **No speech detected:** check the selected microphone, input volume, and system microphone permission.
- **No spoken reply:** Mark 2 and Mark 3 require macOS. Check your speaker volume and output device.
- **Mark 3 cannot connect:** build the helper, check speech recognition permission, and read `/tmp/stt_binary.log`.
- **Slow first run:** allow time for model downloads and loading. The larger models need more memory.

Say `exit` or `quit`, or press `Ctrl+C`, to stop. Conversation text is kept in memory for the current session and printed in the terminal. Use sample details when demonstrating the project.

## What was checked

During the 2 October 2026 import, all six Python files passed a Python 3.12 syntax check. Both supplied Apple settings files could be read, and the included Python and Swift source files matched the ZIP exactly. The Swift build and full voice conversations still need to be checked on macOS with LM Studio and the required models available.
