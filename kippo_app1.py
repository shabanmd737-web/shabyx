import os
import json
import time
import base64

from flask import Flask, Response, request, send_from_directory, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# Main model first, backups after it. Change in Render > Environment if needed.
MODELS = [m.strip() for m in os.environ.get(
    "SHABYX_MODELS", "gemini-3.1-flash-lite,gemini-3.5-flash-lite"
).split(",") if m.strip()]

SYSTEM_PROMPT = """You are SHABYX, a smart, warm and friendly AI assistant. You are female.

Language rules:
- Reply in the same language the user writes in. If the user writes in Hindi, reply in Hindi, and use female grammar for yourself (for example "main karti hoon" style, in proper Hindi). If the user writes in English, reply in English.
- Never mix Hindi and English into Hinglish.

About your creator:
- You were created by Md Shaban Ansari, a B.Tech Computer Science Engineering student at Parul University.
- His interests: AI, software development, cybersecurity and emerging technologies.
- His principle: "Learn continuously. Build practically. Improve relentlessly."
- His Instagram: altairvyn
- If anyone asks who made you, say this clearly. Do not claim to be made by Google or any other company.

Style: clear, helpful and to the point. Use short paragraphs. Be honest when you are not sure.
You cannot generate images right now. If asked, say image generation is coming soon."""


def build_contents(messages):
    contents = []
    for m in messages[-20:]:
        role = "model" if m.get("role") == "assistant" else "user"
        parts = []
        text = (m.get("text") or "").strip()
        if text:
            parts.append(types.Part.from_text(text=text))
        for img in m.get("images") or []:
            try:
                data = base64.b64decode(img["data"])
                parts.append(types.Part.from_bytes(data=data, mime_type=img.get("mime", "image/jpeg")))
            except Exception:
                pass
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents


def stream_reply(contents):
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    sent_any = False
    for model in MODELS:
        for attempt in range(2):
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    text = getattr(chunk, "text", None)
                    if text:
                        sent_any = True
                        yield text
                if sent_any:
                    return
            except Exception as e:
                print(f"[SHABYX] model={model} attempt={attempt} error={e}", flush=True)
                if sent_any:
                    # Reply already started; do not restart it, just stop here.
                    return
                time.sleep(1.2 * (attempt + 1))
    yield "SHABYX is very busy right now. Please send your message again in a few seconds."


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    contents = build_contents(data.get("messages") or [])
    if not contents:
        return Response("Please type a message.", mimetype="text/plain")
    resp = Response(stream_reply(contents), mimetype="text/plain; charset=utf-8")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/health")
def health():
    return "ok", 200


@app.route("/public/<path:name>")
def public_files(name):
    return send_from_directory(os.path.join(BASE_DIR, "public"), name)


@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "kippo_chat1.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__BASE__", request.url_root.rstrip("/"))
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
