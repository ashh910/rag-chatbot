from flask import Flask, render_template, request, jsonify
from agent_invoking import get_response
import requests
import os
from dotenv import load_dotenv
load_dotenv()

GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
ALEMLLM_API_KEY = os.getenv("ALEMLLM_API_KEY")
KAZLLM_API_KEY = os.getenv("KAZLLM_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

URL = os.getenv("URL")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["message"]
    model_choice = request.json.get("model", "gpt-oss")

    match model_choice: 
        case "gpt-oss":
            api_key = GPT_OSS_API_KEY
        case "alemllm":
            api_key = ALEMLLM_API_KEY
        case "kazllm":
            api_key = KAZLLM_API_KEY
        case "gemma4":
            api_key = GEMMA_API_KEY
        case "qwen3-6":
            api_key = QWEN_API_KEY
        case _:
            raise ValueError(f"Invalid model choice: {model_choice}")

    try:
        reply = get_response(question, model_choice, api_key)
    except Exception as e:
        reply = str(e)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=8000, debug=True)