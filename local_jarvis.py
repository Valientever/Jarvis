#!/usr/bin/env python3
"""
JARVIS Local Wake Word Client

Run this on your LOCAL machine (not the remote server).
It listens for "Hey JARVIS" or "Jarvis", then records your command
and sends it to the remote JARVIS server for processing.

Usage:
    python local_jarvis.py --server http://<remote-ip>:5000
"""

import argparse
import pyaudio
import wave
import tempfile
import requests
import base64
import pyttsx3
import whisper
import numpy as np
import time

# Configuration
SAMPLE_RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 500
WAKE_WORDS = ["jarvis", "hey jarvis", "hey travis", "travis", "service"]

print("Loading Whisper model for wake word detection...")
whisper_model = whisper.load_model("tiny")  # Smaller model for faster wake word detection
print("Model loaded!")

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    """Speak text using local TTS"""
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=5):
    """Record audio from microphone"""
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"Recording for {duration} seconds...")
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wf = wave.open(f.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return f.name

def listen_for_wake_word():
    """Continuously listen for wake word"""
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("Listening for 'Hey JARVIS'...")
    frames = []
    recording_time = 2  # Record 2 seconds at a time for wake word detection

    while True:
        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK * recording_time)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Check audio level
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > SILENCE_THRESHOLD:
            # Save and transcribe
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wf = wave.open(f.name, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                result = whisper_model.transcribe(f.name)
                text = result["text"].lower().strip()

                import os
                os.unlink(f.name)

                # Check for wake word
                for wake_word in WAKE_WORDS:
                    if wake_word in text:
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        return True

    return False

def send_to_server(server_url, audio_file):
    """Send audio to remote JARVIS server"""
    with open(audio_file, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    try:
        response = requests.post(
            f"{server_url}/voice",
            json={"audio": audio_data, "user_id": "local_user"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'No response')
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {e}"

def main():
    parser = argparse.ArgumentParser(description='JARVIS Local Wake Word Client')
    parser.add_argument('--server', default='http://192.168.1.219:5000',
                        help='Remote JARVIS server URL')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  JARVIS Local Wake Word Client")
    print("="*50)
    print(f"\nServer: {args.server}")
    print("Say 'Hey JARVIS' to activate")
    print("Press Ctrl+C to quit\n")

    speak("JARVIS local client ready. Say Hey JARVIS to activate.")

    while True:
        try:
            # Wait for wake word
            if listen_for_wake_word():
                speak("Yes?")

                # Record command
                print("Listening for your command...")
                audio_file = record_audio(duration=5)

                print("Sending to server...")
                response = send_to_server(args.server, audio_file)

                import os
                os.unlink(audio_file)

                speak(response)
                print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
