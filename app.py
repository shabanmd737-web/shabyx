import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai
from google.genai import types
from creator import SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__, static_folder="public", static_url_path="/static")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
try:
    client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=50000))
except Exception:
    client = genai.Client(api_key=API_KEY)
# gemini-2.5-flash was shut down on 16 Oct 2026, so we use the current Flash alias
# and fall back to other free models if one is unavailable.
MODELS = [os.getenv("GEMINI_MODEL"), "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3-flash-preview"]
MODELS = [m for m in MODELS if m]


@app.route("/")
def home():
    return send_from_directory(app.root_path, "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        history = data.get("history", [])[-20:]
        web = data.get("web", False)

        contents = []
        for m in history:
            parts = []
            if m.get("image"):
                parts.append(types.Part.from_bytes(
                    data=base64.b64decode(m["image"]["data"]),
                    mime_type=m["image"]["mime"],
                ))
            parts.append(types.Part(text=m["text"]))
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())] if web else None,
        )
        last_error = None
        for model in MODELS:
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                return jsonify({"reply": response.text or "No response received. Please try again."})
            except Exception as e:
                last_error = e
                print("Model failed:", model, "->", e)
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break
                if not ("404" in msg or "NOT_FOUND" in msg or "no longer available" in msg):
                    break  # only try the next model if this one is missing
        raise last_error
    except Exception as e:
        print("Error:", e)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return jsonify({"reply": "SHABYX is busy right now. Please wait a minute and try again."}), 200
        return jsonify({"reply": "Something went wrong (" + type(e).__name__ + "). Please try again in a moment."}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
