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
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"


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
            parts.append(types.Part.from_text(text=m["text"]))
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())] if web else None,
        )
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        return jsonify({"reply": response.text or "No response received. Please try again."})
    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": "Something went wrong. Please try again in a moment."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
