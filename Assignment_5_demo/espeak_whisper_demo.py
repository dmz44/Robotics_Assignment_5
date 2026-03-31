import subprocess
import os
from faster_whisper import WhisperModel

def record_audio(filename="test_audio.wav", duration=5):
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    subprocess.run(["arecord", "-d", str(duration), "-f", "cd", filename])
    print("✅ Recording finished.\n")

def transcribe_audio(filename, model):
    print("📝 Transcribing...")
    segments, _ = model.transcribe(filename, beam_size=5)
    
    full_text = ""
    for segment in segments:
        full_text += segment.text + " "
        
    return full_text.strip()

def speak(text):
    print(f"🔊 Speaking: '{text}'")
    subprocess.run(["espeak-ng", text])

if __name__ == "__main__":
    audio_file = "echo_audio.wav"
    
    print("🧠 Initializing AI...")
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    print("✅ AI Ready.")
    
    record_audio(audio_file, duration=5)
    
    if os.path.exists(audio_file):
        transcribed_text = transcribe_audio(audio_file, whisper_model)
        
        if transcribed_text:
            print(f"\n🗣️ You said: {transcribed_text}\n")
            speak(transcribed_text)
        else:
            print("\n🤷 I didn't hear any words.")
            speak("I did not hear what you said.")
    else:
        print("❌ Error: Audio file missing.")
