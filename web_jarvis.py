import os
import tempfile
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq
import whisper
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are JARVIS, a helpful AI voice assistant. You are friendly, concise, and helpful.
Keep responses brief and conversational since they will be spoken aloud. Use a professional but warm tone.
Avoid using markdown, bullet points, or special formatting - just speak naturally."""

conversation_history = []

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded!")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle text chat"""
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    response = get_ai_response(message)
    return jsonify({'response': response})

@app.route('/voice', methods=['POST'])
def voice():
    """Handle voice input"""
    data = request.json
    audio_data = data.get('audio', '')

    if not audio_data:
        return jsonify({'error': 'No audio provided'}), 400

    try:
        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)

        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        result = whisper_model.transcribe(temp_path)
        os.unlink(temp_path)

        text = result["text"].strip()
        if not text:
            return jsonify({'error': 'Could not understand audio'}), 400

        response = get_ai_response(text)
        return jsonify({
            'transcription': text,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'cleared'})

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

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  JARVIS Web Interface")
    print("="*50)
    print("\nOpen in your browser: http://<server-ip>:5000")
    print("Or locally: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
