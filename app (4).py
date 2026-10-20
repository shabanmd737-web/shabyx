import base64
import os
import re
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")
if not KEY:
    raise SystemExit("GEMINI_API_KEY nahi mili. .env file me GEMINI_API_KEY=apni_key likho.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

SYSTEM = (
    "You are SHABYX, a friendly and helpful AI assistant. "
    "LANGUAGE RULE: reply in the same language the user is speaking, judged by meaning, not by script. "
    "If the user speaks Hindi, even with English words mixed in (for example 'mujhe html sikhao' or "
    "'मुझे HTML सिखाओ'), reply fully in Hindi. If the user speaks English (for example 'I want to learn html'), "
    "reply fully in English. When the user types Hindi in Roman letters (Hinglish), reply in Hinglish (Hindi in Roman letters). "
    "When the user writes Hindi in Devanagari, reply in Devanagari Hindi. "
    "You can see and understand images that the user attaches. "
    "You cannot create images in normal chat: if the user asks you to create, draw or generate an image or photo, "
    "tell them to tap the 🎨 button to turn on image mode and then describe the image they want. "
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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # photo ke liye 16 MB tak

_hits = {}


def too_many(limit=10, per=3600):
    """Ek IP se ek ghante me max `limit` image requests (free quota bachane ke liye)."""
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "?").split(",")[0].strip()
    now = time.time()
    q = [x for x in _hits.get(ip, []) if now - x < per]
    if len(q) >= limit:
        _hits[ip] = q
        return True
    q.append(now)
    _hits[ip] = q
    return False


def parse_img(s):
    """'data:image/jpeg;base64,....' ko Gemini Part me badalta hai. Galat ho to None."""
    m = re.match(r"^data:(image/(?:jpeg|png|webp));base64,(.+)$", s or "", re.S)
    if not m:
        return None
    try:
        raw = base64.b64decode(m.group(2), validate=True)
    except Exception:
        return None
    if len(raw) > 6 * 1024 * 1024:
        return None
    return types.Part.from_bytes(data=raw, mime_type=m.group(1))


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
    contents = []
    for m in msgs:
        role = "model" if m.get("role") == "assistant" else "user"
        parts = []
        if role == "user":
            p = parse_img(m.get("image"))
            if p:
                parts.append(p)
        txt = str(m.get("content", ""))[:4000]
        if txt.strip():
            parts.append(types.Part(text=txt))
        elif parts:
            parts.append(types.Part(text="Is image ko dekho aur batao isme kya hai."))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
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


@app.post("/api/image")
def image():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()[:1000]
    if not prompt:
        return jsonify(error="Pehle likho ki kaisi image banani hai."), 400
    if too_many():
        return jsonify(error="Ek ghante me bahut image ban chuki hain. Thodi der baad try karo."), 429
    parts = []
    src = parse_img(data.get("image"))  # image di ho to wo edit hogi
    if src:
        parts.append(src)
    parts.append(types.Part(text=prompt))
    try:
        r = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except Exception as e:
        print("Image error:", e)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return jsonify(error="Image ka limit abhi khatam hai. Thodi der baad try karo."), 429
        return jsonify(error="Image nahi ban payi. Dobara try karo."), 502
    text, img = "", None
    for c in r.candidates or []:
        out = c.content.parts if c.content and c.content.parts else []
        for p in out:
            if getattr(p, "inline_data", None) and p.inline_data.data and not img:
                mime = p.inline_data.mime_type or "image/png"
                img = "data:%s;base64,%s" % (mime, base64.b64encode(p.inline_data.data).decode())
            elif getattr(p, "text", None):
                text += p.text
    if not img:
        return jsonify(error="Image nahi ban payi (ho sakta hai ye prompt allowed nahi). Alag tarike se likho."), 422
    return jsonify(image=img, text=text.strip()[:500])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
