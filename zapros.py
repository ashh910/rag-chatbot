from flask import Flask, render_template, request, jsonify
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv
import custom_llm
import agent_invoking


load_dotenv()

API_KEY = os.getenv("API_KEY")
GEMINI_API = os.getenv("GEMINI_API")
URL = os.getenv("URL")


resource_folder = "RAG Files"


@tool
def search_documents(question, resource_folder = resource_folder):
    ''' 
    This function searches for the most relevant information in 
    the documents stored in the resource folder based on the 
    user's question. It loads all text files, splits them into 
    chunks, creates embeddings, and performs a search to find 
    the most relevant chunks. The relevant information is then 
    appended to the payload for the LLM to use in generating 
    a response.
    '''

    resources = []

    #load all text files from the resource folder
    for filename in os.listdir(resource_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(resource_folder, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            resources.extend(loader.load())

    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = [" "],
        chunk_size = 300,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_documents(resources)

    #create embeddings and store them in a vector database
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

    #search for the most relevant chunks based on the question
    relevant_chunks = vectordb.max_marginal_relevance_search(
        question,
        k=2,
        fetch_k=3
    )

    #append the relevant chunks to the payload for the LLM
    custom_llm.payload["messages"].append({
        "role": "user",
        "content": f"the most relevant information extracted from documents: {relevant_chunks}"
    })


my_chatbot = custom_llm.CustomLLM(API_KEY, URL)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    reply = agent_invoking.agent_responses()

    try:
        custom_llm.payload["messages"].append({
            "role": "assistant",
            "content": reply
        })

    except Exception as e:
        reply = str(e)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(port=5000, debug=True)