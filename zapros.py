from flask import Flask, render_template, request, jsonify, Request # RequestEntityTooLarge
from agent_invoking import agent_response
from manual_agent_invoking import manual_agent_response, kazllm_history, alemllm_history
from rag import (is_allowed_file_format, add_uploaded_documents_to_vectorstore)
import os
from dotenv import load_dotenv
load_dotenv()

GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
ALEMLLM_API_KEY = os.getenv("ALEMLLM_API_KEY")
KAZLLM_API_KEY = os.getenv("KAZLLM_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

URL = os.getenv("URL")

upload_folder = "RAG Files/uploaded files"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16*1000*1000

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    is_file_uploaded = False

    if request.is_json:
        question = request.json.get("message", "")
        model_choice = request.json.get("model", "gpt-oss")
    else:
        question = request.form.get("message", "")
        model_choice = request.form.get("model", "gpt-oss")

    try:
        files = request.files.getlist('files')
    except:
        raise RequestEntityTooLarge("Uploaded document is too large.")

    '''
        Connection Reset Issue

            When using the local development server, you may get a connection 
            reset error instead of a 413 response. You will get the correct 
            status response when running the app with a production WSGI server.

            https://flask.palletsprojects.com/en/stable/patterns/fileuploads/
    '''

    manual_agent = False
    history_log = None
 
    match model_choice: 
        case "gpt-oss":
            api_key = GPT_OSS_API_KEY
            manual_agent = False
        case "alemllm":
            api_key = ALEMLLM_API_KEY
            manual_agent = True
            history_log = alemllm_history
        case "kazllm":
            api_key = KAZLLM_API_KEY
            manual_agent = True 
            history_log = kazllm_history
        case "gemma4":
            api_key = GEMMA_API_KEY
            manual_agent = False
        case "qwen3-6":
            api_key = QWEN_API_KEY
            manual_agent = False
        case _:
            raise ValueError(f"Invalid model choice: {model_choice}")

    if files is None and not question:
        return jsonify({"error": "No input."}), 400
    
    file_status = []

    if files:
        accepted_files = []

        for file in files:
            if not is_allowed_file_format(file):
                file_status.append({"filename": file.filename, "accepted": False})
            else:
                accepted_files.append(file)
                file_status.append({"filename": file.filename, "accepted": True})
        
        if accepted_files is not None:
            add_uploaded_documents_to_vectorstore(accepted_files)

        is_file_uploaded = True

        try:
            if manual_agent:
                reply = manual_agent_response(api_key, model_choice, question, accepted_files)
            else:
                reply = agent_response(api_key, model_choice, question, accepted_files)
        except Exception as e:
            reply = str(e)
    else:
        try:
            if manual_agent:
                reply = manual_agent_response(api_key, model_choice, question)
            else:
                reply = agent_response(api_key, model_choice, question)
        except Exception as e:
            reply = str(e)
    
    if files:
        print({"reply": reply, "file_status": file_status})
        return jsonify({"reply": reply, "file_status": file_status})
    else:
        print({"reply": reply})
        return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=8000, debug=True)