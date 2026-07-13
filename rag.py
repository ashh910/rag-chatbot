from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from functools import lru_cache
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API = os.getenv("GEMINI_API")
EMBEDDER = os.getenv("EMBEDDER")
EMBED_URL = os.getenv("EMBED_URL")
RERANKER_API = os.getenv("RERANKER_API")
RERANKER_URL = os.getenv("RERANKER_URL")

resource_folder = "RAG Files"
persist_directory = "RAG Files/chrome/"

embedding = OpenAIEmbeddings(
    model = "text-1024",
    api_key = EMBEDDER,
    base_url = EMBED_URL
)

chunks_json = os.path.join(resource_folder, "splitted_chunks.json")

def save_chunks(chunks, path=chunks_json):
    chunks_list = []
    for chunk in chunks:
        chunks_list.append({"metadata": chunk.metadata, 
                        "page_content": chunk.page_content})
    with open(path, "w") as saving_chunks:
        json.dump(chunks_list, saving_chunks)

def load_chunks(path=chunks_json):
    with open(path, "r") as reading_chunks:
        datas = json.load(reading_chunks)
    result = []
    for data in datas:
        result.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
    return result

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

    Run this once (or whenever documents change) — not on every query.
    '''
    chunks = load_chunks()
    documents_text = []
    for chunk in chunks:
        documents_text.append(chunk.page_content)

    #search for the most relevant chunks based on the question
    reranker = {
        "model": "reranker",
        "query": question,
        "documents": documents_text,
        "top_n": 3,
    }

    try:
        resp = requests.post(
            RERANKER_URL,
            headers={
                "Authorization": f"Bearer {RERANKER_API}",
                "Content-Type": "application/json",
            },
            json=reranker,
            timeout=60,
        )
        resp.raise_for_status()
        relevant_chunks = resp.json()
        return relevant_chunks["results"]
    except requests.RequestException as e:
        print(f"Request failed: {e}\nBody: {resp.text if 'resp' in locals() else ''}")