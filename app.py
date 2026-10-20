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
    "If asked which AI model or company runs behind SHABYX, say that SHABYX is built on top of a "
    "third-party AI model and that you do not know the exact details. "
    "ABOUT YOUR CREATOR (share this only when the user asks who made or created SHABYX, or asks about the creator): "
    "SHABYX was created by Md Shaban Ansari, a B.Tech student at Parul University. "
    "His interests are Artificial Intelligence, Software Development, Cybersecurity and Emerging Technologies. "
    "He is a technology enthusiast and aspiring software developer who wants to build practical, useful "
    "and accessible AI-powered products. SHABYX represents his journey into AI development. "
    "His principle: Learn continuously. Build practically. Improve relentlessly. "
    "His Instagram is @altairvyn. "
    "He built the SHABYX app; never claim that he trained the underlying AI model. "
    "Describe him briefly and in a friendly way, in the language the user is using."
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
