import subprocess

def speak(text):
    print(f"Speaking: '{text}'")
    # Using espeak-ng which is installed in your Dockerfile
    subprocess.run(["espeak-ng", text])

if __name__ == "__main__":
    test_text = "Hello! I am speaking to you from inside the ROS 2 Humble Docker container."
    speak(test_text)
