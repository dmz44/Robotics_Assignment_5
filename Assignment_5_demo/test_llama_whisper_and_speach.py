#!/usr/bin/env python3

import os, sys, select, subprocess
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from llama_cpp import Llama
from queue import Queue
from threading import Thread
import subprocess

MODEL_PATH = os.path.expanduser("~/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf")
RECORD_SECONDS = 5

# Resource management - using CPU for whisper.
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
# Select Whisper Model You can try base or small
whisper_model = whisper.load_model("small", device=device)

# LLama CPP
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=33, 
    #use_mlock=True,
)

# Create a FIFO queue for TTS texts
_tts_queue = Queue()

def _tts_worker():
  
    while True:
        text = _tts_queue.get()
        if text is None:
            break      
        # This blocks until this sentence is done speaking
        subprocess.run(
            ["espeak", "-s", "140", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _tts_queue.task_done()

# Start the worker thread as a daemon so it exits with the program
_tts_thread = Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    _tts_queue.put(text)

def record_and_save(duration=RECORD_SECONDS, fs=16000, fname="mic.wav"):
    print("start recording")
    print(duration)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    write(fname, fs, audio)
    print("end recording")
    return fname

def transcribe_whisper(fname):
    print("whisper start")
    res = whisper_model.transcribe(fname, temperature=0.0)
    print(res)   
    print("whisper end")
    return res["text"].strip()

def get_user_input():
    _tts_queue.join()
    sys.stdout.write("\nYou (type or 's'/wait→voice): ")   
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], 5)
    if ready:
        line = sys.stdin.readline().strip()
        if   line.lower() in ("exit","quit"):
            return None
        elif line.lower() == "s":
            speak("speak in voice now")
        else:
            return line
    # timeout or explicit 's'
    speak("Voice Input:")
    wav = record_and_save()
    speak("Processing:")
    txt = transcribe_whisper(wav)
    print(f"You (voice): {txt}")
    speak("Robot speaking.")
    return txt

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    print("LLM Ready — type 'exit' to quit.")
    speak("LLM Ready")
    while True:
        user_text = get_user_input()
        if user_text is None:
            break

        print("Assistant: ", end="", flush=True)
        buf = ""
        for chunk in llm(
            prompt=user_text,
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            buf += token
            print(token, end="", flush=True)
            if token.endswith((" ", ".", "?", "!")):
                speak(buf)
                buf = ""
        if buf.strip():
            speak(buf)
        print()

    print("\n Goodbye!")

if __name__ == "__main__":
    main()

