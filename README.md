# Conversational Agent

A voice assistant that collects details for a car repair booking through a spoken conversation.

**Group project · January–May 2026 · Python and Swift**

The assistant asks for a name, car registration, mileage, and car model, then reads the details back. The project explores different ways to recognise speech, decide when someone has finished speaking, and start a spoken reply sooner.

## What it does

- Listens through a microphone and turns speech into text.
- Uses a model running in LM Studio to continue the conversation.
- Remembers earlier messages during a session.
- Reads replies aloud.
- Includes three versions and a separate speech recognition experiment.

The booking versions collect and confirm details. They do not save appointments or connect to a booking calendar.

## Versions

| Version | Approach | Main file |
| --- | --- | --- |
| First version | Records short clips, uses Whisper to recognise speech, then reads the full reply aloud. | [`voice_assistant.py`](voice_assistant.py) |
| Speech experiment | Tries Parakeet speech recognition and TurnSense to check whether someone has finished speaking. | [`parakeet_stt.py`](parakeet_stt.py) |
| Mark 2 | Uses RealtimeSTT and TEN turn detection, and reads each sentence as the reply arrives. | [`mark_2/voice_assistant_mark_2.py`](mark_2/voice_assistant_mark_2.py) |
| Mark 3 | Uses Apple speech recognition, listens for pauses, and plays a short acknowledgement while preparing the reply. | [`mark_3/voice_assistant_mark_3.py`](mark_3/voice_assistant_mark_3.py) |

Mark 2 and Mark 3 use macOS speech tools. Mark 3 also includes a Swift helper built from source. The Parakeet file is a general voice assistant experiment rather than a booking assistant.

## How it works

```text
You speak → speech becomes text → LM Studio prepares a reply → the assistant speaks
```

Later versions change how the assistant decides you have finished speaking and how soon it starts reading a reply. The code includes timing output, but this archive does not include a measured comparison between versions.

## Try it

Start with the first version. You need Python, a microphone, speakers, and [LM Studio](https://lmstudio.ai/) with a chat model loaded and its local server running on port `1234`.

```sh
git clone https://github.com/nm2064/conversational-agent.git
cd conversational-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python voice_assistant.py
```

On Windows, create the environment with `py -3 -m venv .venv`, then activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

Allow microphone access when your system asks. Say `exit` or `quit`, or press `Ctrl+C`, to stop. Speech models download on first use. See the [setup guide](docs/setup.md) for the other versions and the macOS helper build.

## Team

- [Noman Maqsudi](https://github.com/nm2064)
- Aryan Jangir
- Liam Clark Moreno
- Karan Kothari

## Project history

Developed from January to May 2026 as a group project. The versions explore different ways to recognise speech, decide when someone has finished speaking, and deliver spoken replies.

The Python and Swift source files are preserved as supplied. Generated programs, Python cache files, and macOS housekeeping files are left out; the setup guide explains how to build the Mark 3 helper again.

## Current checks and limits

All six Python files passed a syntax check with Python 3.12, and the supplied Apple settings files were checked. Full voice conversations and the Swift build have not been tested during this import, which was prepared on Windows. The project has no saved appointment system or automated end-to-end tests, and the original package versions were not recorded.
