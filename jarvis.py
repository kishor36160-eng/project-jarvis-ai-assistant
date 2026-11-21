import sounddevice as sd
import numpy as np
import pyttsx3
import requests
import subprocess
import time
import os
from faster_whisper import WhisperModel

# ---------------- SETTINGS ----------------

OLLAMA_URL = "http://localhost:11434/api/chat"
AI_MODEL = "qwen2.5:3b"

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

# Multilingual, lightweight speech recognition
print("Loading Jarvis...")

whisper = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

# Voice engine
engine = pyttsx3.init()
engine.setProperty("rate", 190)
engine.setProperty("volume", 1.0)

conversation = []
active = False

# ---------------- SPEAK ----------------

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()

# ---------------- LISTEN ----------------

def listen():
    try:
        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        audio = np.squeeze(audio)

        if np.max(np.abs(audio)) < 0.01:
            return ""

        segments, info = whisper.transcribe(
            audio,
            language=None,
            beam_size=1,
            vad_filter=True
        )

        # Language name is deliberately not displayed
        text = " ".join(
            segment.text.strip() for segment in segments
        )

        return text.strip().lower()

    except Exception as error:
        print("Mic error:", error)
        return ""

# ---------------- APP CONTROL ----------------

def open_app(command):

    apps = {
        "spotify": "spotify",
        "chrome": "chrome",
        "calculator": "calc",
        "notepad": "notepad",
        "paint": "mspaint",
        "file explorer": "explorer",
        "explorer": "explorer"
    }

    for name, executable in apps.items():
        if name in command:
            try:
                subprocess.Popen(executable, shell=True)
                speak("Opening " + name)
                return True
            except Exception:
                speak("App could not be opened")
                return True

    folders = {
        "downloads": os.path.expanduser("~/Downloads"),
        "documents": os.path.expanduser("~/Documents"),
        "desktop": os.path.expanduser("~/Desktop")
    }

    for name, path in folders.items():
        if name in command:
            os.startfile(path)
            speak("Opening " + name)
            return True

    return False

# ---------------- AI BRAIN ----------------

def ask_ai(question):

    conversation.append({
        "role": "user",
        "content": question
    })

    messages = [
        {
            "role": "system",
            "content": (
                "You are Jarvis, a helpful multilingual AI assistant. "
                "Understand and respond in the same language used by "
                "the user. Support Tamil, English, Hindi, Tanglish and "
                "other languages. Do not announce the detected language. "
                "Keep answers concise and natural. Never claim to have "
                "performed a computer action unless it was actually done."
            )
        }
    ] + conversation[-10:]

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": AI_MODEL,
                "messages": messages,
                "stream": False
            },
            timeout=90
        )

        response.raise_for_status()

        answer = response.json()["message"]["content"]

        conversation.append({
            "role": "assistant",
            "content": answer
        })

        return answer

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please open Ollama."

    except Exception as error:
        print("AI error:", error)
        return "Sorry, I could not process that."

# ---------------- MAIN ----------------

def main():

    global active

    speak("Jarvis is ready.")

    while True:

        command = listen()

        if not command:
            continue

        print("You:", command)

        if not active:

            if "hey jarvis" in command or "hi jarvis" in command:
                active = True
                speak("Yes, how can I help?")

            continue

        if "goodbye jarvis" in command or "bye jarvis" in command:
            active = False
            conversation.clear()
            speak("Going to standby.")
            continue

        if "stop jarvis" in command:
            speak("Jarvis stopped.")
            break

        if open_app(command):
            continue

        answer = ask_ai(command)
        speak(answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJarvis stopped.")