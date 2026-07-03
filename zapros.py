import requests
from flask import Flask, render_template, request, jsonify
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
GEMINI_API = os.getenv("GEMINI_API")
URL = os.getenv("URL")

resource_folder = "RAG Files"
resources = []

for filename in os.listdir(resource_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(resource_folder, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        resources.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    separators = [" "],
    chunk_size = 300,
    chunk_overlap = 50,
    length_function = len
)
chunks = text_splitter.split_documents(resources)

embedding = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2-preview",
    google_api_key = GEMINI_API,
)

persist_directory = "RAG Files/chrome/"
vectordb = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory = persist_directory
)


app = Flask(__name__)

payload = {
    "model": "gpt-oss",
    "temperature": 0,
    "messages": [
        {
            "role": "system",
            "content": "Ты краткий ассистент. Если ты не знаешь какой-то информации или не можешь точно её проверить, не упоминай её.",
        }
    ],
}


def ask_gpt(question):
    #relevant info with diversity
    relevant_chunks = vectordb.max_marginal_relevance_search(
        question,
        k=2,
        fetch_k=3
    )

    payload["messages"].append({
        "role": "user",
        "content": question
    })

    payload["messages"].append({
        "role": "user",
        "content": f"the most relevant information extracted from documents: {relevant_chunks}"
    })

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    reply = data["choices"][0]["message"]["content"]

    payload["messages"].append({
        "role": "assistant",
        "content": reply
    })

    return reply


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    try:
        reply = ask_gpt(user_message)
    except Exception as e:
        reply = str(e)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)