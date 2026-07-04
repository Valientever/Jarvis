import os
import tempfile
import wave
import pyaudio
import pyttsx3
import whisper
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are JARVIS, a helpful AI voice assistant. You are friendly, concise, and helpful.
Keep responses brief and conversational since they will be spoken aloud. Use a professional but warm tone.
Avoid using markdown, bullet points, or special formatting - just speak naturally."""

conversation_history = []

print("Loading Whisper model (this may take a moment)...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded!")

engine = pyttsx3.init()
engine.setProperty('rate', 175)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    p = pyaudio.PyAudio()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )

    print(f"Recording for {duration} seconds...")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording complete!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wf = wave.open(f.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return f.name

def transcribe_audio(audio_file):
    """Convert speech to text using Whisper"""
    result = whisper_model.transcribe(audio_file)
    os.unlink(audio_file)
    return result["text"].strip()

def speak(text):
    """Convert text to speech"""
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()

def get_ai_response(message: str) -> str:
    """Get response from Groq"""
    conversation_history.append({"role": "user", "content": message})

    if len(conversation_history) > 20:
        conversation_history[:] = conversation_history[-20:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=256,
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message

def main():
    print("\n" + "="*50)
    print("  JARVIS Voice Assistant")
    print("="*50)
    print("\nCommands:")
    print("  Press ENTER to start recording (5 seconds)")
    print("  Type 'quit' to exit")
    print("  Type anything else to send as text\n")

    speak("Hello! I'm JARVIS, your voice assistant. How can I help you today?")

    while True:
        user_input = input("\n[Press ENTER to speak, or type a message]: ").strip()

        if user_input.lower() == 'quit':
            speak("Goodbye!")
            break
        elif user_input == "":
            try:
                audio_file = record_audio(duration=5)
                print("Transcribing...")
                text = transcribe_audio(audio_file)
                if text:
                    print(f"You said: {text}")
                    response = get_ai_response(text)
                    speak(response)
                else:
                    print("Could not understand audio. Please try again.")
            except Exception as e:
                print(f"Error recording: {e}")
        else:
            response = get_ai_response(user_input)
            speak(response)

if __name__ == "__main__":
    main()
