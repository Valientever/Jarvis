import os
import tempfile
import base64
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from groq import Groq
import whisper
from dotenv import load_dotenv

load_dotenv()

# Initialize AI
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are JARVIS, a helpful AI assistant created by the user. You are friendly, concise, and helpful.
Keep responses brief and conversational. Use a professional but warm tone.
Avoid using markdown, bullet points, or special formatting - just speak naturally.

You are currently integrated with Slack and a web voice interface. When users message you, just respond helpfully to their questions or requests. Do not say you cannot access things - just answer their questions directly.

Never say things like "I can't access your Slack" or "I don't have access to" - you are JARVIS, a capable assistant. If you don't know something, just say so simply."""

# Shared conversation history (keyed by user_id)
conversation_history = {}

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded!")

# ============== Shared AI Function ==============

def get_ai_response(user_id: str, message: str) -> str:
    """Get response from Groq - shared between Slack and Web"""
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": "user", "content": message})

    if len(conversation_history[user_id]) > 20:
        conversation_history[user_id] = conversation_history[user_id][-20:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[user_id]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=256,
    )

    assistant_message = response.choices[0].message.content
    conversation_history[user_id].append({"role": "assistant", "content": assistant_message})

    return assistant_message

# ============== Flask Web Server ==============

flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@flask_app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id', 'web_user')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    response = get_ai_response(user_id, message)
    return jsonify({'response': response})

@flask_app.route('/voice', methods=['POST'])
def voice():
    data = request.json
    audio_data = data.get('audio', '')
    user_id = data.get('user_id', 'web_user')

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

        response = get_ai_response(user_id, text)
        return jsonify({
            'transcription': text,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/clear', methods=['POST'])
def clear():
    data = request.json or {}
    user_id = data.get('user_id', 'web_user')
    if user_id in conversation_history:
        conversation_history[user_id] = []
    return jsonify({'status': 'cleared'})

# ============== Slack Bot ==============

slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])

@slack_app.event("app_mention")
def handle_mention(event, say):
    user_id = f"slack_{event['user']}"
    text = event["text"].split(">", 1)[-1].strip()

    if not text:
        say("Hello! How can I help you?")
        return

    try:
        response = get_ai_response(user_id, text)
        say(response)
    except Exception as e:
        say(f"Sorry, I encountered an error: {e}")

@slack_app.event("message")
def handle_message(event, say):
    if event.get("channel_type") != "im":
        return
    if "bot_id" in event:
        return

    user_id = f"slack_{event['user']}"
    text = event.get("text", "")

    if not text:
        return

    try:
        response = get_ai_response(user_id, text)
        say(response)
    except Exception as e:
        say(f"Sorry, I encountered an error: {e}")

# ============== Main ==============

def run_flask():
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def run_slack():
    handler = SocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  JARVIS Unified Assistant")
    print("="*50)
    print("\n  Web Interface: http://192.168.1.219:5000")
    print("  Slack: Message @JARVIS in your workspace")
    print("\n  Both share the same AI brain!")
    print("="*50 + "\n")

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Web server started!")

    # Start Slack (blocks main thread)
    print("Slack bot starting...")
    run_slack()
