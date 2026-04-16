"""
Voice Assistant using Parakeet STT (GPU) + TurnSense + LM Studio (LLM) + pyttsx3 (TTS)
Automatically listens, detects end of speech, queries LLM, speaks response, and loops back
"""

import numpy as np
import sounddevice as sd
import torch
import threading
import queue
import time
import requests
import pyttsx3

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# LLM Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# Global state
audio_queue = queue.Queue()
is_running = True
turn_complete = False
final_transcription = ""

# Conversation history for context
conversation_history = []

# TTS engine (lazy initialization)
tts_engine = None


def get_tts_engine():
    """Lazy initialize TTS engine"""
    global tts_engine
    if tts_engine is None:
        print("Initializing TTS engine...", flush=True)
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 1.0)
    return tts_engine


def speak(text):
    """Convert text to speech using pyttsx3"""
    print(f"\n🔊 Assistant: {text}", flush=True)
    engine = get_tts_engine()
    engine.say(text)
    engine.runAndWait()


def query_llm(user_input):
    """Send text to LM Studio and get response"""
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a helpful voice assistant. Keep your responses concise and conversational since they will be spoken aloud."},
            *conversation_history
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": False
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    except requests.exceptions.ConnectionError:
        return "I couldn't connect to LM Studio. Please make sure it's running on port 1234."
    except Exception as e:
        return f"Error communicating with LLM: {str(e)}"


def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream"""
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def get_device():
    """Determine the best available device for inference"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32


def load_stt_model(device, dtype):
    """Load Parakeet-CTC-1.1B model"""
    print("Loading Parakeet-CTC-1.1B STT model...")
    from transformers import AutoModelForCTC, AutoProcessor
    
    model_name = "nvidia/parakeet-ctc-1.1b"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    
    print(f"✅ STT Model loaded on {device.upper()}")
    return model, processor


def load_turnsense_model():
    """Load TurnSense model using ONNX"""
    print("Loading TurnSense turn-detection model...")
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    
    model_id = "latishab/turnsense"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")
    
    available_providers = ort.get_available_providers()
    if 'CoreMLExecutionProvider' in available_providers:
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        provider_name = "CoreML (GPU)"
    else:
        providers = ['CPUExecutionProvider']
        provider_name = "CPU"
    
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"✅ TurnSense loaded on {provider_name}")
    
    return session, tokenizer


def check_turn_complete(turnsense_session, tokenizer, text):
    """Check if the user has finished speaking using TurnSense"""
    if not text or len(text.strip()) < 3:
        return False
    
    formatted_text = f"<|user|> {text}"
    inputs = tokenizer(
        formatted_text,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="np"
    )
    
    ort_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    
    all_logits = turnsense_session.run(None, ort_inputs)[0]
    logits = all_logits[0]
    prediction = int(np.argmax(logits))
    
    return prediction == 1  # 1 = turn complete


def transcribe_audio(model, processor, device, audio_data):
    """Transcribe audio chunk using Parakeet model"""
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    audio_data = np.ascontiguousarray(audio_data.flatten())
    
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    if max_val < 0.01:
        return ""
    
    inputs = processor(
        audio=audio_data,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )
    
    model_dtype = next(model.parameters()).dtype
    
    if hasattr(inputs, 'input_features') and inputs.input_features is not None:
        input_tensor = inputs.input_features.to(device=device, dtype=model_dtype)
    elif hasattr(inputs, 'input_values') and inputs.input_values is not None:
        input_tensor = inputs.input_values.to(device=device, dtype=model_dtype)
    else:
        input_tensor = inputs["input_features"].to(device=device, dtype=model_dtype)
    
    with torch.no_grad():
        logits = model(input_tensor).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    transcription = transcription.replace("<pad>", "").replace("  ", " ").strip()
    
    return transcription


def listen_for_utterance(stt_model, stt_processor, turnsense_session, turnsense_tokenizer, device):
    """Listen for a complete utterance, using TurnSense to detect end of speech"""
    global audio_queue
    
    audio_queue = queue.Queue()  # Reset queue
    audio_buffer = np.array([], dtype=np.float32)
    full_text = ""
    last_text = ""
    last_speech_time = time.time()
    
    print("\n🎙️  Listening...\n")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=audio_callback
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    
                    chunk_flat = chunk.flatten().astype(np.float32)
                    audio_buffer = np.concatenate([audio_buffer, chunk_flat])
                    
                    # Check for speech/silence
                    chunk_energy = np.abs(chunk_flat).max()
                    if chunk_energy >= 0.01:
                        last_speech_time = time.time()
                    
                    silence_duration = time.time() - last_speech_time
                    
                    # Transcribe when we have enough audio
                    if len(audio_buffer) >= CHUNK_SIZE:
                        text = transcribe_audio(stt_model, stt_processor, device, audio_buffer)
                        
                        if text and len(text) > 1:
                            if text != last_text:
                                full_text = text
                                print(f"\r📝 {full_text}                              ", end="", flush=True)
                                last_text = text
                        
                        # Keep overlap
                        overlap = int(SAMPLE_RATE * 0.3)
                        audio_buffer = audio_buffer[-overlap:]
                    
                    # After 1 second of silence, check TurnSense
                    if full_text and silence_duration >= 1.0:
                        is_complete = check_turn_complete(
                            turnsense_session, turnsense_tokenizer, full_text
                        )
                        
                        if is_complete:
                            print(f"\n\n✅ TurnSense: End of utterance detected")
                            return full_text
                        
                        # If not complete but 2.5s silence, stop anyway
                        if silence_duration >= 2.5:
                            print(f"\n\n✅ Silence timeout")
                            return full_text
                    
                except queue.Empty:
                    silence_duration = time.time() - last_speech_time
                    if full_text and silence_duration >= 2.5:
                        print(f"\n\n✅ Silence timeout")
                        return full_text
                    continue
                    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return ""


def main():
    """Main voice assistant loop"""
    print("=" * 60)
    print("🤖 Voice Assistant")
    print("    Parakeet STT (GPU) + TurnSense + LM Studio + TTS")
    print("=" * 60)
    
    # Get device
    device, dtype = get_device()
    print(f"\n🖥️  Using device: {device.upper()}\n")
    
    # Load models
    try:
        stt_model, stt_processor = load_stt_model(device, dtype)
        turnsense_session, turnsense_tokenizer = load_turnsense_model()
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test LM Studio connection
    print("\n🔗 Testing LM Studio connection...")
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running")
        else:
            print("⚠️  LM Studio responded but may have issues")
    except:
        print("⚠️  Could not connect to LM Studio. Make sure it's running on port 1234.")
    
    # Test TTS
    print("\n🔊 Testing TTS...")
    speak("Voice assistant is ready. Just start speaking.")
    
    print("\n" + "=" * 60)
    print("Ready! Just speak - I'll listen automatically.")
    print("Say 'goodbye' or 'exit' to quit.")
    print("Press Ctrl+C to force quit.")
    print("=" * 60)
    
    try:
        while True:
            # Listen for user speech
            user_text = listen_for_utterance(
                stt_model, stt_processor, 
                turnsense_session, turnsense_tokenizer, 
                device
            )
            
            if not user_text:
                print("(No speech detected, listening again...)")
                continue
            
            # Skip very short inputs
            if len(user_text.strip()) < 2:
                print("(Too short, listening again...)")
                continue
            
            print(f"📜 You said: {user_text}")
            
            # Check for exit commands
            if any(word in user_text.lower() for word in ['exit', 'quit', 'goodbye', 'bye', 'stop']):
                speak("Goodbye! Have a great day!")
                break
            
            # Query LLM
            print("\n💭 Thinking...")
            response = query_llm(user_text)
            
            # Speak the response
            speak(response)
            
            # Small pause before listening again
            time.sleep(0.3)
            
            # Loop back to listening automatically
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("👋 Goodbye!")
        print("=" * 60)


if __name__ == "__main__":
    main()
