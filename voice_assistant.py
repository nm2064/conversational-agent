import json
import sys
import time
import numpy as np
import tempfile
import os

print("Loading modules...", flush=True)

import sounddevice as sd
import whisper
import requests
import pyttsx3

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
WHISPER_MODEL = "small.en"
SAMPLE_RATE = 16000

conversation_history = []
whisper_model = None


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return whisper_model


def speak(text):
    print(f"\nAssistant: {text}", flush=True)
    try:
        tts_start = time.time()
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        tts_elapsed = time.time() - tts_start
        print(f"[TTS synthesis took {tts_elapsed:.2f}s]")
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"TTS Error: {e} - retrying...")
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except:
            print("TTS failed, continuing without speech")


def clean_response(text):
    import re
    if "<|" in text or "|>" in text:
        text = re.sub(r'<\|[^|]*\|>[^<]*', '', text)
    text = re.sub(r'\{[^}]*"next_action"[^}]*\}', '', text)
    text = text.strip()
    if not text or len(text) < 3:
        return "Could you please continue?"
    return text


def query_llm(user_input):
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    system_prompt = """You are a car repair booking assistant. Your job is to help customers book a slot for car repair by having a natural conversation.

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

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            *conversation_history
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False
    }
    
    try:
        llm_start = time.time()
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        llm_elapsed = time.time() - llm_start
        print(f"[LLM took {llm_elapsed:.2f}s]")

        assistant_message = result['choices'][0]['message']['content']
        
        assistant_message = clean_response(assistant_message)
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    except requests.exceptions.ConnectionError:
        return "I couldn't connect to LM Studio. Please make sure it's running on port 1234."
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"


def listen_for_speech(duration_seconds=5):
    print("\nListening... (speak now)")
    
    try:
        audio_data = sd.rec(
            int(duration_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        audio_data = audio_data.flatten()
        
        if np.abs(audio_data).max() < 0.01:
            return ""
        
        print("Transcribing...")
        model = load_whisper_model()
        stt_start = time.time()
        result = model.transcribe(audio_data, language="en", fp16=False)
        stt_elapsed = time.time() - stt_start
        print(f"[Whisper STT took {stt_elapsed:.2f}s]")
        text = result["text"].strip()
        
        if text:
            print(f"Recognized: {text}")
        
        return text
        
    except Exception as e:
        print(f"Error recording/transcribing: {e}")
        return ""


def main():
    print("=" * 50)
    print("Car Repair Booking Assistant")
    print("=" * 50)
    print("STT: Whisper small.en | LLM: LM Studio | TTS: pyttsx3")
    print("=" * 50)
    
    load_whisper_model()
    
    print("\nTesting LM Studio connection...")
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("LM Studio is running")
        else:
            print("LM Studio responded but may have issues")
    except:
        print("Could not connect to LM Studio. Make sure it's running on port 1234.")
    
    speak("Hello! I'm here to help you book a car repair slot. Let's get started. What is your full name?")
    
    print("\n" + "=" * 50)
    print("Say 'exit' or 'quit' to stop. Press Ctrl+C to force quit.")
    print("=" * 50)
    
    try:
        while True:
            user_text = listen_for_speech(duration_seconds=5)
            
            if not user_text:
                print("(No speech detected, listening again...)")
                continue
            
            if len(user_text.strip()) < 2:
                continue
            
            if user_text.lower().strip() in ['exit', 'quit', 'goodbye', 'bye', 'stop', 'exit.', 'quit.', 'goodbye.', 'bye.', 'stop.']:
                speak("Goodbye! Have a great day!")
                break
            
            print("\nThinking...")
            response = query_llm(user_text)
            
            speak(response)
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
