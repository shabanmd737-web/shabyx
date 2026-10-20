import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")
if not KEY:
    raise SystemExit("GEMINI_API_KEY nahi mili. .env file me GEMINI_API_KEY=apni_key likho.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
SYSTEM = (
    "You are SHABYX, a friendly and helpful AI assistant. "
    "LANGUAGE RULE: reply in the same language the user is speaking, judged by meaning, not by script. "
    "If the user speaks Hindi, even with English words mixed in (for example 'mujhe html sikhao' or "
    "'मुझे HTML सिखाओ'), reply fully in Hindi. If the user speaks English (for example 'I want to learn html'), "
    "reply fully in English. When the user types Hindi in Roman letters (Hinglish), reply in Hinglish (Hindi in Roman letters). "
    "When the user writes Hindi in Devanagari, reply in Devanagari Hindi. "
    "Keep answers clear and useful. "
    "Never say which company or model powers you; you are just SHABYX."
)
VOICE_NOTE = (
    " The reply will be read aloud, so keep it short and conversational, with no markdown, tables or emojis. "
    "The user's message comes from speech recognition, so English speech may be written in Devanagari letters "
    "(for example 'आई वांट टू लर्न एचटीएमएल' is English, and you must answer it in English using normal English letters). "
    "Judge the language actually spoken. If it is Hindi (even with some English words), answer fully in Devanagari Hindi, "
    "writing English or technical words in Devanagari too (for example पाइथन, कंप्यूटर), with no Roman letters."
)

client = genai.Client(api_key=KEY)
app = Flask(__name__, static_folder=None)  # sirf public/ ki files bahar jaati hain, .env nahi


@app.get("/")
def home():
    return send_from_directory("public", "index.html")


@app.get("/manifest.json")
def manifest():
    return send_from_directory("public", "manifest.json")


@app.get("/icon.svg")
def icon():
    return send_from_directory("public", "icon.svg")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    msgs = data.get("messages", [])[-20:]
    system = SYSTEM + (VOICE_NOTE if data.get("voice") else "")
    contents = [
        types.Content(
            role="model" if m.get("role") == "assistant" else "user",
            parts=[types.Part(text=str(m.get("content", ""))[:4000])],
        )
        for m in msgs
    ]
    if not contents or contents[-1].role != "user":
        return jsonify(error="Message khali hai."), 400
    try:
        r = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return jsonify(reply=r.text or "Mujhe is par jawab nahi mila. Dusre tareeke se pucho.")
    except Exception as e:
        print("Gemini error:", e)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return jsonify(error="Abhi bahut zyada log use kar rahe hain. Thodi der baad try karo."), 429
        return jsonify(error="AI se jawab nahi mila. Dobara try karo."), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
