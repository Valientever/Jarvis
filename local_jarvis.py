#!/usr/bin/env python3
"""
JARVIS Local Wake Word Client

Run this on your LOCAL machine (not the remote server).
Say "Hey JARVIS" to start a conversation, then keep talking naturally.
Say "goodbye" or stay silent to end the conversation.

Usage:
    python local_jarvis.py --server http://<remote-ip>:5000
"""

import argparse
import pyaudio
import wave
import requests
import base64
import pyttsx3
import whisper
import numpy as np
import time
import os

# Configuration
SAMPLE_RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 100
CONVERSATION_TIMEOUT = 10  # Seconds of silence before ending conversation
WAKE_WORDS = ["jarvis", "hey jarvis", "hey travis", "travis"]
EXIT_WORDS = ["goodbye", "bye", "stop", "exit", "quit", "that's all", "thank you goodbye"]
DEVICE_INDEX = 1  # Jabra Evolve2 65 - change if needed

print("Loading Whisper model...")
whisper_model = whisper.load_model("tiny")
print("Model loaded!")

engine = pyttsx3.init()
engine.setProperty('rate', 175)

def speak(text):
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def save_audio(frames, p):
    filepath = os.path.join(os.environ.get('TEMP', '.'), 'jarvis_temp.wav')
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filepath

def record_until_silence(max_duration=10, silence_duration=1.5):
    """Record audio until user stops speaking"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                    input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK)

    print("Listening... (speak now)")
    frames = []
    silent_chunks = 0
    has_speech = False
    max_chunks = int(SAMPLE_RATE / CHUNK * max_duration)
    silence_chunks_threshold = int(SAMPLE_RATE / CHUNK * silence_duration)

    for _ in range(max_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > SILENCE_THRESHOLD:
            has_speech = True
            silent_chunks = 0
        else:
            silent_chunks += 1

        # Stop if we had speech and then silence
        if has_speech and silent_chunks > silence_chunks_threshold:
            break

    stream.stop_stream()
    stream.close()

    if not has_speech:
        p.terminate()
        return None

    filepath = save_audio(frames, p)
    p.terminate()
    return filepath

def listen_for_wake_word():
    """Listen for wake word"""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                    input_device_index=DEVICE_INDEX, frames_per_buffer=CHUNK)
    print("\nListening for 'Hey JARVIS'...")

    while True:
        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK * 2)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > 50:
            print(f"  [Volume: {volume:.0f}]", end="\r")

        if volume > SILENCE_THRESHOLD:
            filepath = save_audio(frames, p)
            try:
                result = whisper_model.transcribe(filepath)
                text = result["text"].lower().strip()
                print(f"  Heard: '{text}'")
            except:
                continue

            for wake_word in WAKE_WORDS:
                if wake_word in text:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return True

    return False

def transcribe_audio(filepath):
    """Transcribe audio file"""
    try:
        result = whisper_model.transcribe(filepath)
        return result["text"].strip()
    except:
        return ""

def send_to_server(server_url, audio_file):
    with open(audio_file, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    try:
        response = requests.post(f"{server_url}/voice", json={"audio": audio_data, "user_id": "local_user"}, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {e}"

def conversation_mode(server_url):
    """Stay in conversation until user says goodbye or is silent"""
    speak("Yes? I'm listening.")

    silence_count = 0
    max_silence = 3  # End conversation after 3 silent attempts

    while True:
        audio_file = record_until_silence(max_duration=10, silence_duration=1.5)

        if audio_file is None:
            silence_count += 1
            if silence_count >= max_silence:
                speak("I'll be here if you need me.")
                return
            print(f"  (No speech detected, {max_silence - silence_count} more tries)")
            continue

        silence_count = 0

        # Transcribe locally to check for exit words
        text = transcribe_audio(audio_file)
        print(f"  You: {text}")

        # Check for exit words
        text_lower = text.lower()
        for exit_word in EXIT_WORDS:
            if exit_word in text_lower:
                speak("Goodbye! Let me know if you need anything.")
                os.unlink(audio_file)
                return

        # Send to server for response
        print("  Processing...")
        response = send_to_server(server_url, audio_file)
        os.unlink(audio_file)

        speak(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://192.168.1.219:5000')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("  JARVIS Local Wake Word Client")
    print("="*50)
    print(f"\nServer: {args.server}")
    print(f"Microphone: Device {DEVICE_INDEX}")
    print("\nSay 'Hey JARVIS' to start a conversation")
    print("Say 'goodbye' to end the conversation")
    print("Press Ctrl+C to quit\n")

    speak("JARVIS ready. Say Hey JARVIS to start a conversation.")

    while True:
        try:
            if listen_for_wake_word():
                conversation_mode(args.server)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
