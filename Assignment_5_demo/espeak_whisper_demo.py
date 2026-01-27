#!/usr/bin/env python3
import os, sys, select, subprocess
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from llama_cpp import Llama


RECORD_SECONDS = 5

# Change here to test cpu only whisper.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Change here to test different model sizes or english only models.
whisper_model = whisper.load_model("small", device=device)


from queue import Queue
from threading import Thread

# Create a FIFO queue for TTS texts
_tts_queue = Queue()

def _tts_worker():

    while True:
        text = _tts_queue.get()
        if text is None:
            break      
        # this blocks until this sentence is done speaking
        subprocess.run(
            ["espeak", "-s", "140", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        _tts_queue.task_done()

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

    print("Whisper Ready — type 'exit' to quit.")
    speak("Whisper Ready")
    while True:
        user_text = get_user_input()
        if user_text is None:
            break

        # stream llama + TTS
        print("This is what you said: ", end="", flush=True)
        print(user_text)
        speak("This is what you said")
        speak(user_text)
        print()

    print("\n Goodbye!")

if __name__ == "__main__":
    main()

