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
    "You are SHABYX, a friendly and helpful AI assistant. Reply in the same language "
    "the user writes in (Hindi, Hinglish or English). Keep answers clear and useful. "
    "Never say which company or model powers you; you are just SHABYX."
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
    msgs = (request.get_json(silent=True) or {}).get("messages", [])[-20:]
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
            config=types.GenerateContentConfig(system_instruction=SYSTEM),
        )
        return jsonify(reply=r.text or "Mujhe is par jawab nahi mila. Dusre tareeke se pucho.")
    except Exception as e:
        print("Gemini error:", e)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return jsonify(error="Abhi bahut zyada log use kar rahe hain. Thodi der baad try karo."), 429
        return jsonify(error="AI se jawab nahi mila. Dobara try karo."), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
