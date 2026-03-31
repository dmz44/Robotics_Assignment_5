#!/usr/bin/env python3

import os
import sys
import select
import subprocess
from queue import Queue
from threading import Thread

from faster_whisper import WhisperModel
from llama_cpp import Llama

# Make sure this path is mounted into your container, or download the model inside it!
MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")
RECORD_SECONDS = 5

# ==========================================
# 1. Initialize AI Models (GPU Accelerated)
# ==========================================
print("🧠 Loading Faster-Whisper (small) onto GPU...")
# Utilizing the CUDA 12.4 setup from your Dockerfile
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")

print(f"🧠 Loading LLM from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=33, # Offloading layers to your RTX 5050 
)

# ==========================================
# 2. TTS Setup (espeak-ng Queue)
# ==========================================
_tts_queue = Queue()

def _tts_worker():
    while True:
        text = _tts_queue.get()
        if text is None:
            break      
        # Using espeak-ng installed via your Dockerfile
        subprocess.run(
            ["espeak-ng", "-s", "140", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _tts_queue.task_done()

# Start the worker thread as a daemon so it exits with the program
_tts_thread = Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    _tts_queue.put(text)

# ==========================================
# 3. Audio Recording & Transcription
# ==========================================
def record_and_save(duration=RECORD_SECONDS, fname="mic.wav"):
    print(f"\n🎤 Recording for {duration} seconds...")
    # Using ALSA's arecord instead of sounddevice
    subprocess.run(
        ["arecord", "-d", str(duration), "-f", "cd", fname],
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
    )
    print("✅ Recording finished.")
    return fname

def transcribe_whisper(fname):
    print("📝 Transcribing...")
    segments, _ = whisper_model.transcribe(fname, beam_size=5)
    
    # Faster-whisper returns an iterator of segments, we need to join them
    text = " ".join([segment.text for segment in segments])
    return text.strip()

# ==========================================
# 4. Main Interaction Logic
# ==========================================
def get_user_input():
    _tts_queue.join()
    sys.stdout.write("\nYou (type or 's'/wait→voice): ")   
    sys.stdout.flush()
    
    # Wait 5 seconds for keyboard input
    ready, _, _ = select.select([sys.stdin], [], [], 5)
    
    if ready:
        line = sys.stdin.readline().strip()
        if line.lower() in ("exit", "quit"):
            return None
        elif line.lower() == "s":
            speak("speak in voice now")
        else:
            return line
            
    # Timeout or explicit 's' triggers voice
    speak("Voice Input:")
    wav = record_and_save()
    speak("Processing:")
    txt = transcribe_whisper(wav)
    print(f"You (voice): {txt}")
    speak("Robot speaking.")
    return txt

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"\n❌ ERROR: LLM model not found at {MODEL_PATH}")
        print("Make sure you mounted the directory in docker-compose.yml or downloaded the file!")
        return

    print("\n✅ Systems Ready — type 'exit' to quit.")
    speak("Systems Ready")
    
    while True:
        user_text = get_user_input()
        if user_text is None:
            break

        print("Assistant: ", end="", flush=True)
        buf = ""
        
        # Stream the LLM response
        for chunk in llm(
            prompt=user_text, # Note: For Llama-2-chat, you may want to wrap this in [INST] [/INST] tags if responses get weird
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            buf += token
            print(token, end="", flush=True)
            
            # Speak sentence by sentence
            if token.endswith((" ", ".", "?", "!")):
                # Only queue it if there's actual text (prevents espeak from reading punctuation out loud weirdly)
                if buf.strip():
                    speak(buf.strip())
                buf = ""
                
        # Catch any leftover text
        if buf.strip():
            speak(buf.strip())
        print()

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
