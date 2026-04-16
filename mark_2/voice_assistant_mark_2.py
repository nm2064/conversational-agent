import json
import sys
import time
import re
import os
import subprocess
import threading
import queue

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading modules...", flush=True)

import requests
from ten_turn_detection import TENTurnDetector


LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

STT_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
STT_REALTIME_MODEL = "tiny.en"

SILENCE_SHORT_MS = 300   
SILENCE_HARD_MS = 1500 

SENTENCE_DELIMITERS = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])$')

conversation_history = []
recorder = None

# ── Streaming TTS via macOS `say` ────────────────────────────────────
_tts_queue = queue.Queue()


def _tts_worker():
    """Background thread: speaks each sentence using macOS `say`."""
    while True:
        sentence = _tts_queue.get()
        if sentence is None:
            break
        try:
            subprocess.run(["say", "-r", "175", sentence], check=False)
        except Exception:
            pass
        _tts_queue.task_done()


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text):
    """Speak a complete string (blocks until done)."""
    print(f"\nAssistant: {text}", flush=True)
    _tts_queue.put(text)
    _tts_queue.join()


# ── Response cleaning / LLM ────────────────────────────────────────
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


def query_and_speak(user_input):
    """
    Stream LLM tokens, split into sentences, and queue each sentence to
    the `say` worker immediately — speech starts as soon as the first
    sentence is complete. After streaming ends, block until all speech
    finishes, then return so the caller starts listening again.
    """
    conversation_history.append({"role": "user", "content": user_input})

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        llm_start = time.time()
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

            # Queue each complete sentence immediately for speech
            parts = SENTENCE_DELIMITERS.split(sentence_buffer)
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        _tts_queue.put(sentence)
                sentence_buffer = parts[-1]

        # Queue any remaining text
        leftover = sentence_buffer.strip()
        if leftover:
            _tts_queue.put(leftover)

        llm_elapsed = time.time() - llm_start
        print(f"\n[LLM streamed in {llm_elapsed:.2f}s]")

        assistant_message = "".join(full_response).strip()
        if not assistant_message or len(assistant_message) < 3:
            assistant_message = "Could you please continue?"
        conversation_history.append({"role": "assistant", "content": assistant_message})

        # Block until all queued sentences have been spoken
        _tts_queue.join()

    except requests.exceptions.ConnectionError:
        msg = "I couldn't connect to LM Studio. Please make sure it's running on port 1234."
        print(msg)
        speak(msg)
    except Exception as e:
        msg = f"Error communicating with LLM: {str(e)}"
        print(msg)
        speak(msg)


# ── RealtimeSTT realtime callback for TEN turn detection ─────────────
# This fires as partial transcription updates come in from RealtimeSTT.
# We track the latest text + timestamp so the main listen loop can
# detect silence windows and consult TEN.

_realtime_text = ""
_realtime_update_time = 0.0
_realtime_lock = threading.Lock()


def on_realtime_transcription_update(text):
    """Called by RealtimeSTT whenever the live partial transcription changes."""
    global _realtime_text, _realtime_update_time
    if not text or len(text.strip()) < 2:
        return
    with _realtime_lock:
        _realtime_text = text.strip()
        _realtime_update_time = time.time()


# ── Listen with TEN turn detection ──────────────────────────────────
def listen_with_turn_detection(ten):
    """
    Use RealtimeSTT for audio capture + Whisper transcription.
    Monitor the realtime partial transcription and apply TEN turn detection:
      After 150ms silence → ask TEN:
        finished   → stop recording and trigger LLM
        unfinished → keep listening
        wait       → suppress, keep listening
      Hard fallback at 800ms silence → trigger LLM regardless.
    """
    global recorder, _realtime_text, _realtime_update_time

    print("\nListening... (speak now)", flush=True)

    # Reset realtime tracking
    with _realtime_lock:
        _realtime_text = ""
        _realtime_update_time = 0.0

    # Reset silence to a very long value — we control stopping via TEN
    recorder.post_speech_silence_duration = 3.0

    last_ten_text = None
    started_speaking = False

    # Start recording in a background thread — recorder.text() blocks
    result_box = [None]
    done_event = threading.Event()

    def record_thread():
        result_box[0] = recorder.text()
        done_event.set()

    t = threading.Thread(target=record_thread, daemon=True)
    t.start()

    while not done_event.is_set():
        time.sleep(0.02)

        with _realtime_lock:
            current_text = _realtime_text
            last_update = _realtime_update_time

        if not current_text or last_update == 0.0:
            continue

        if not started_speaking:
            started_speaking = True

        silence_ms = (time.time() - last_update) * 1000

        # ── Hard fallback: 800ms of silence → force stop ──
        if silence_ms >= SILENCE_HARD_MS:
            print(f"\n[Silence {silence_ms:.0f}ms ≥ {SILENCE_HARD_MS}ms → hard fallback]", flush=True)
            recorder.abort()
            done_event.wait(timeout=2)
            return current_text

        # ── At 150ms silence: consult TEN ──
        if silence_ms >= SILENCE_SHORT_MS and last_ten_text != current_text:
            last_ten_text = current_text
            ten_start = time.time()
            decision = ten.detect(current_text)
            ten_elapsed = (time.time() - ten_start) * 1000

            # Check if new words arrived during TEN inference
            with _realtime_lock:
                post_text = _realtime_text
                post_update = _realtime_update_time

            if post_text != current_text:
                print(f"\n[TEN={decision} in {ten_elapsed:.0f}ms — ignored, new speech]", flush=True)
                continue

            print(f"\n[TEN={decision} in {ten_elapsed:.0f}ms | silence={silence_ms:.0f}ms]", flush=True)

            if decision == "finished":
                recorder.abort()
                done_event.wait(timeout=2)
                return current_text
            elif decision == "wait":
                pass  # suppress — keep listening
            # "unfinished" — keep listening

    # Normal RealtimeSTT completion (long silence timeout)
    text = result_box[0]
    return text.strip() if text else ""


# ── Main loop ───────────────────────────────────────────────────────
def main():
    global recorder

    print("=" * 60)
    print("Car Repair Booking Assistant  [Mark 2 — TEN Turn Detection]")
    print("=" * 60)
    print(f"STT: RealtimeSTT ({STT_MODEL})")
    print("LLM: LM Studio (streaming) | TTS: macOS say (streaming)")
    print("Turn detection: TEN-framework/TEN_Turn_Detection (Qwen 7B)")
    print(f"Silence: {SILENCE_SHORT_MS}ms (TEN check) / {SILENCE_HARD_MS}ms (hard fallback)")
    print("=" * 60)

    # 1. Load TEN turn detection model
    ten = TENTurnDetector()
    ten.load()

    # 2. Initialize RealtimeSTT
    print("\nInitializing RealtimeSTT...", flush=True)
    from RealtimeSTT import AudioToTextRecorder

    recorder = AudioToTextRecorder(
        model=STT_MODEL,
        language="en",
        spinner=False,
        realtime_model_type=STT_REALTIME_MODEL,
        enable_realtime_transcription=True,
        on_realtime_transcription_stabilized=on_realtime_transcription_update,
        post_speech_silence_duration=3.0,
        silero_sensitivity=0.4,
        webrtc_sensitivity=3,
        min_length_of_recording=0.5,
        min_gap_between_recordings=0,
        beam_size=5,
        beam_size_realtime=3,
        no_log_file=True,
        silero_use_onnx=True,
    )
    print("RealtimeSTT ready.", flush=True)

    # 3. Test LM Studio
    print("\nTesting LM Studio connection...")
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("LM Studio is running")
        else:
            print("LM Studio responded but may have issues")
    except Exception:
        print("Could not connect to LM Studio. Make sure it's running on port 1234.")

    # 4. Greeting
    speak("Hello! I'm here to help you book a car repair slot. Let's get started. What is your full name?")

    print("\n" + "=" * 60)
    print("Say 'exit' or 'quit' to stop. Press Ctrl+C to force quit.")
    print("=" * 60)

    try:
        while True:
            user_text = listen_with_turn_detection(ten)

            if not user_text:
                print("(No speech detected, listening again...)")
                continue

            if len(user_text.strip()) < 2:
                continue

            print(f"Recognized: {user_text}")

            if user_text.lower().strip().rstrip('.') in [
                'exit', 'quit', 'goodbye', 'bye', 'stop',
            ]:
                speak("Goodbye! Have a great day!")
                break

            query_and_speak(user_text)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        _tts_queue.put(None)


if __name__ == "__main__":
    main()
