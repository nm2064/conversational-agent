import json
import time
import re
import os
import subprocess
import threading
import queue
import random
import tempfile
import socket
import struct
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading modules...", flush=True)

import sounddevice as sd
import webrtcvad
import whisper
import requests

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

SAMPLE_RATE    = 16000
FRAME_MS       = 30
FRAME_SAMPLES  = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_LIMIT_MS = 500
SILENCE_FRAMES   = SILENCE_LIMIT_MS // FRAME_MS

VAD_AGGRESSIVENESS = 2

STT_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stt_binary.app")
STT_SOCKET = "/tmp/stt.sock"
WHISPER_MODEL = "small.en"

whisper_model = None

CLAUSE_DELIMITERS = re.compile(r'(?<=[.!?])\s+')

FILLER_WORDS = ["alright", "ok", "sure", "right", "got it", "uh huh", "let me see", "one moment"]

SAY_VOICE = None  # None = system default voice (matches original behaviour).
SAY_RATE = "175"

FILLER_CACHE_DIR = "/tmp/mark3_fillers"
_filler_audio = {}  # word -> path to .aiff

# TODO (later optimization): keep a persistent `say` process running for the
# non-filler path and pipe text to its stdin, eliminating the per-sentence
# spawn cost for LLM response sentences too.


def prerender_fillers():
    os.makedirs(FILLER_CACHE_DIR, exist_ok=True)
    # Tag the voice/rate in the filename so changing them re-renders instead of reusing stale audio.
    tag = re.sub(r'\W+', '_', f"{SAY_VOICE or 'default'}_{SAY_RATE}")
    for word in FILLER_WORDS:
        fname = f"{tag}__{re.sub(r'\W+', '_', word)}.wav"
        path = os.path.join(FILLER_CACHE_DIR, fname)
        if not os.path.exists(path):
            # Render at 48 kHz so afplay doesn't resample at playback time
            # (afplay's default resampler muffles 22050 → 48000).
            cmd = ["say", "-r", SAY_RATE,
                   "--file-format=WAVE", "--data-format=LEI16@48000",
                   "-o", path, word]
            if SAY_VOICE:
                cmd[1:1] = ["-v", SAY_VOICE]
            subprocess.run(cmd, check=False)
        if os.path.exists(path):
            _filler_audio[word] = path

conversation_history = []


def load_whisper():
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper ({WHISPER_MODEL})...", flush=True)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper loaded.", flush=True)
    return whisper_model


_tts_queue = queue.Queue()


def _tts_worker():
    while True:
        sentence = _tts_queue.get()
        if sentence is None:
            break
        try:
            cached = _filler_audio.get(sentence)
            if cached:
                subprocess.run(["afplay", cached], check=False)
            else:
                cmd = ["say", "-r", SAY_RATE, sentence]
                if SAY_VOICE:
                    cmd[1:1] = ["-v", SAY_VOICE]
                subprocess.run(cmd, check=False)
        except Exception:
            pass
        _tts_queue.task_done()


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text):
    print(f"\nAssistant: {text}", flush=True)
    _tts_queue.put(text)
    _tts_queue.join()


def clean_chunk(text):
    if "<|" in text or "|>" in text:
        text = re.sub(r'<\|[^|]*\|>[^<]*', '', text)
    text = re.sub(r'\{[^}]*"next_action"[^}]*\}', '', text)
    return text


SYSTEM_PROMPT = """You are a car repair booking assistant. Your job is to help customers book a slot for car repair by having a natural conversation.

You need to collect the following information from the customer:
1. Full name
2. Car registration / license plate
3. Vehicle mileage
4. Car model

Important rules:
- Ask for ONE piece of information at a time
- Respond ONLY with natural conversational text that will be spoken aloud
- Do NOT include any JSON, code, special tokens, or formatting
- Do NOT include any internal notes or developer comments
- Keep responses short and friendly

Once you have all 4 pieces of information, confirm the details back to the customer."""


def query_and_speak(user_input, filler_used):
    conversation_history.append({"role": "user", "content": user_input})

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + f'\n\nYou just said "{filler_used}" as a filler word before this response. Continue your reply in a tone that flows naturally from that word — do not repeat it, just let it set the tone. Be subtle about it.'},
            *conversation_history,
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,
    }

    full_response = []
    sentence_buffer = ""

    print("\nAssistant: ", end="", flush=True)

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                content = chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if not content:
                continue

            cleaned = clean_chunk(content)
            if not cleaned:
                continue

            full_response.append(cleaned)
            sentence_buffer += cleaned
            print(cleaned, end="", flush=True)

            parts = CLAUSE_DELIMITERS.split(sentence_buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        _tts_queue.put(sentence)
                sentence_buffer = parts[-1]

        leftover = sentence_buffer.strip()
        if leftover:
            _tts_queue.put(leftover)

        assistant_message = "".join(full_response).strip()
        if not assistant_message or len(assistant_message) < 3:
            assistant_message = "Could you please continue?"
        conversation_history.append({"role": "assistant", "content": assistant_message})

        _tts_queue.join()

    except requests.exceptions.ConnectionError:
        msg = "I couldn't connect to LM Studio. Please make sure it's running on port 1234."
        print(msg)
        speak(msg)
    except Exception as e:
        msg = f"Error communicating with LLM: {str(e)}"
        print(msg)
        speak(msg)


def listen_and_stream_transcribe(on_vad_end=None):
    """Stream mic audio to the STT daemon live during VAD capture.
    Calls on_vad_end() as soon as silence is detected (before waiting for the
    final transcript), so the caller can start filler TTS in parallel.
    Returns (text, t_vad_end). Empty text means no usable speech."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("\nListening... (speak now)", flush=True)

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(20)
        sock.connect(STT_SOCKET)
    except Exception as e:
        print(f"[STT] Can't connect: {e}", flush=True)
        return "", time.time()

    silent_frames = 0
    speech_detected = False

    def send_frame(frame_bytes):
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        payload = samples.tobytes()
        sock.sendall(struct.pack("<I", len(payload)) + payload)

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=FRAME_SAMPLES,
        ) as stream:
            while True:
                raw_frame, _ = stream.read(FRAME_SAMPLES)
                frame_bytes = bytes(raw_frame)
                is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

                if is_speech:
                    send_frame(frame_bytes)
                    speech_detected = True
                    silent_frames = 0
                elif speech_detected:
                    send_frame(frame_bytes)
                    silent_frames += 1
                    if silent_frames >= SILENCE_FRAMES:
                        break
    except Exception as e:
        print(f"[audio] {e}", flush=True)

    t_vad_end = time.time()

    try:
        sock.sendall(struct.pack("<I", 0))
    except Exception:
        pass

    if not speech_detected:
        try: sock.close()
        except Exception: pass
        return "", t_vad_end

    if on_vad_end is not None:
        try: on_vad_end()
        except Exception: pass

    try:
        hdr = b""
        while len(hdr) < 4:
            chunk = sock.recv(4 - len(hdr))
            if not chunk: break
            hdr += chunk
        if len(hdr) < 4:
            return "", t_vad_end
        n = struct.unpack("<I", hdr)[0]
        out = b""
        while len(out) < n:
            chunk = sock.recv(min(n - len(out), 65536))
            if not chunk: break
            out += chunk
        return out.decode("utf-8", errors="replace").strip(), t_vad_end
    except Exception as e:
        print(f"[STT] Recv error: {e}", flush=True)
        return "", t_vad_end
    finally:
        try: sock.close()
        except Exception: pass


def _stt_socket_alive():
    if not os.path.exists(STT_SOCKET):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect(STT_SOCKET)
        s.close()
        return True
    except OSError:
        return False


def ensure_stt_daemon():
    if _stt_socket_alive():
        return
    if os.path.exists(STT_SOCKET):
        try:
            os.unlink(STT_SOCKET)
        except OSError:
            pass
    print("[STT] Launching Apple STT daemon...", flush=True)
    subprocess.Popen(["open", STT_APP])
    deadline = time.time() + 15
    while time.time() < deadline:
        if _stt_socket_alive():
            print("[STT] Daemon ready.", flush=True)
            return
        time.sleep(0.1)
    raise RuntimeError(f"STT daemon did not come up within 15s (check /tmp/stt_binary.log)")


# Whisper fallback disabled for now — testing Apple STT binary only.


def main():
    print("=" * 60)
    print("Car Repair Booking Assistant  [Mark 3 — VAD + Apple STT]")
    print("=" * 60)
    print(f"STT: ./stt_binary (Apple) or Whisper {WHISPER_MODEL} (fallback)")
    print(f"VAD: webrtcvad (aggressiveness={VAD_AGGRESSIVENESS}, silence={SILENCE_LIMIT_MS}ms)")
    print("LLM: LM Studio (streaming) | TTS: macOS say (streaming)")
    print("Filler: queued at VAD end, flows into LLM response")
    print("=" * 60)

    print("\nTesting LM Studio connection...")
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        print("LM Studio is running" if r.status_code == 200 else "LM Studio may have issues")
    except Exception:
        print("Could not connect to LM Studio. Make sure it's running on port 1234.")

    print("[TTS] Pre-rendering filler audio...", flush=True)
    prerender_fillers()
    print(f"[TTS] Cached {len(_filler_audio)} fillers in {FILLER_CACHE_DIR}", flush=True)

    if not os.path.isdir(STT_APP):
        print(f"[STT] App bundle not found at {STT_APP} — STT will not work.")
    else:
        print(f"[STT] App bundle found: {STT_APP}")
        ensure_stt_daemon()

    speak("Hello! I'm here to help you book a car repair slot. Let's get started. What is your full name?")

    print("\n" + "=" * 60)
    print("Say 'exit' or 'quit' to stop. Press Ctrl+C to force quit.")
    print("=" * 60)

    try:
        while True:
            _tts_queue.join()
            time.sleep(0.05)

            filler_used = random.choice(FILLER_WORDS)
            user_text, t_vad_end = listen_and_stream_transcribe(
                on_vad_end=lambda: _tts_queue.put(filler_used)
            )

            if not user_text or len(user_text.strip()) < 2:
                print("(No transcript, listening again...)")
                _tts_queue.join()
                continue

            print(f"Recognized: {user_text}")

            if user_text.lower().strip().rstrip('.') in ['exit', 'quit', 'goodbye', 'bye', 'stop']:
                _tts_queue.join()
                speak("Goodbye! Have a great day!")
                break

            query_and_speak(user_text, filler_used)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        _tts_queue.put(None)


if __name__ == "__main__":
    main()
