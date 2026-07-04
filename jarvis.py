import os
import logging
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from groq import Groq

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

print(f"Bot token starts with: {os.environ.get('SLACK_BOT_TOKEN', 'NOT SET')[:20]}...")
print(f"App token starts with: {os.environ.get('SLACK_APP_TOKEN', 'NOT SET')[:20]}...")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = App(token=os.environ["SLACK_BOT_TOKEN"])

SYSTEM_PROMPT = """You are JARVIS, a helpful AI assistant. You are friendly, concise, and helpful.
Keep responses brief unless asked for detail. Use a professional but warm tone."""

conversation_history = {}

def get_ai_response(user_id: str, message: str) -> str:
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": "user", "content": message})

    if len(conversation_history[user_id]) > 20:
        conversation_history[user_id] = conversation_history[user_id][-20:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[user_id]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
    )

    assistant_message = response.choices[0].message.content
    conversation_history[user_id].append({"role": "assistant", "content": assistant_message})

    return assistant_message

@app.event("app_mention")
def handle_mention(event, say):
    print(f"📩 Received mention: {event}")
    user_id = event["user"]
    text = event["text"].split(">", 1)[-1].strip()

    if not text:
        say("Hello! How can I help you?")
        return

    try:
        response = get_ai_response(user_id, text)
        say(response)
    except Exception as e:
        print(f"❌ Error: {e}")
        say(f"Sorry, I encountered an error: {e}")

@app.event("message")
def handle_message(event, say):
    print(f"📩 Received message: {event}")
    if event.get("channel_type") != "im":
        return
    if "bot_id" in event:
        return

    user_id = event["user"]
    text = event.get("text", "")

    if not text:
        return

    try:
        response = get_ai_response(user_id, text)
        say(response)
    except Exception as e:
        print(f"❌ Error: {e}")
        say(f"Sorry, I encountered an error: {e}")

if __name__ == "__main__":
    print("🤖 JARVIS is starting...")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    print("🟢 JARVIS is online! Message me in Slack.")
    handler.start()
