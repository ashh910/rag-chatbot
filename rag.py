from langchain_community.document_loaders import (TextLoader, PyPDFLoader, 
                                                    WebBaseLoader)
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from functools import lru_cache
import requests, json, os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API = os.getenv("GEMINI_API")
EMBEDDER = os.getenv("EMBEDDER")
EMBED_URL = os.getenv("EMBED_URL")
RERANKER_API = os.getenv("RERANKER_API")
RERANKER_URL = os.getenv("RERANKER_URL")

preupload_persist_directory = "RAG Files/preuploaded files/chrome/"
preupload_folder = "RAG Files/preuploaded files/"

upload_folder = "RAG Files/uploaded files/"
for document in os.listdir(upload_folder):
    path = os.path.join(upload_folder, document)
    os.remove(path)


allowed_file_formats = [".txt", ".pdf"]


embedding = OpenAIEmbeddings(
    model = "text-1024",
    api_key = EMBEDDER,
    base_url = EMBED_URL
)


upload_vectordb = InMemoryVectorStore(embedding = embedding)
web_vectordb = InMemoryVectorStore(embedding = embedding)


def is_allowed_file_format(document):
    try:
        document_name = document.filename
    except AttributeError:
        document_name = document
    for file_format in allowed_file_formats:
        if document_name.endswith(file_format):
            return True


text_splitter = RecursiveCharacterTextSplitter(
    separators = [" ", "\n", "\n\n", "\t", ",", ".", "!", "?", ";", ":"],
    chunk_size = 750,
    chunk_overlap = 150,
    length_function = len
)


#vectorstore for pre uploaded content for a model (uploaded by me, not the user)
def create_static_vectorstore():
    global preupload_vectordb

    temp = []

    for document in os.listdir(preupload_folder):
        if is_allowed_file_format(document):
            path = os.path.join(preupload_folder, document)
            splitted_document = document.split(".")
            ending = splitted_document[-1]
            match ending:
                case "txt":
                    loader = TextLoader(path, encoding = "utf-8")
                case "pdf":
                    loader = PyPDFLoader(path)
            temp.extend(loader.load())

    chunks = text_splitter.split_documents(temp)

    preupload_vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embedding,
        persist_directory = preupload_persist_directory
    )


def add_uploaded_documents_to_vectorstore(uploaded_documents_list):
    global upload_vectordb
    temp = [] 

    for document in uploaded_documents_list:
        print(type(document))
        path = os.path.join(upload_folder, document.filename)
        document.save(path)
        document_name = document.filename
        splitted_document = document_name.split(".")
        ending = splitted_document[-1]
        match ending:
            case "txt":
                loader = TextLoader(path, encoding = "utf-8")
            case "pdf":
                loader = PyPDFLoader(path)
        temp.extend(loader.load())

    chunks = text_splitter.split_documents(temp)
    upload_vectordb.add_documents(documents = chunks)


if not os.path.exists(preupload_persist_directory) or not os.listdir(preupload_persist_directory):
    create_static_vectorstore()
else:
    preupload_vectordb = Chroma(
        persist_directory=preupload_persist_directory,
        embedding_function=embedding
    )


@tool
@lru_cache(maxsize=10)
def web_search(question, link):
    
    ''' 
    You have access to a web_search tool that searches web links and returns relevant excerpts.

    Only call web_search when the user's question requires looking up 
    specific information from provided link. For greetings, small talk, 
    or questions you can already answer directly, respond in plain text 
    without calling any tool.

    web_search takes two arguments:
    - question: the user's question, as a plain string.
    - link: the link provided by the user.
    '''

    print(f"[DEBUG]: [tool call] Information from a website was extracted for this call: {link}.")

    temp = []
    try:
        loader = WebBaseLoader(link)
        temp.extend(loader.load())
        chunks = text_splitter.split_documents(temp)
        web_vectordb.add_documents(documents = chunks)
    except Exception as e:
        print(f"Somethign went wrong: {e}")

    similar_chunks_pool = web_vectordb.similarity_search(question, k=20)
    chunks_content = []
    for chunk in similar_chunks_pool:
        chunks_content.append(chunk.page_content)

    reranker = {
        "model": "reranker",
        "query": question,
        "documents": chunks_content,
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


@tool 
@lru_cache(maxsize=10) 
def search_documents(question, is_uploaded_document=False):

    ''' 
    You have access to a search_documents tool that searches documents and returns relevant excerpts.

    Only call search_documents when the user's question requires looking up 
    specific information from documents. For greetings, small talk, 
    or questions you can already answer directly, respond in plain text 
    without calling any tool.

    search_documents takes two arguments:
    - question: the user's question, as a plain string.
    - is_uploaded_document: true if the user is asking about a file they personally 
      uploaded in this conversation in current or preivous messages, 
      false if asking about general/reference documents. This value determines if the
      model should refer to vectorbase for file data uploaded by the user.
    '''

    if is_uploaded_document:
        vectordb = upload_vectordb
        print(f"[DEBUG]: [tool call] User's file data was used to search documents.")
    else:
        vectordb = preupload_vectordb
        print(f"[DEBUG]: [tool call] Preuploaded file data was used to search documents.")

    similar_chunks_pool = vectordb.similarity_search(question, k=25)
    chunks_content = []
    for chunk in similar_chunks_pool:
        chunks_content.append(chunk.page_content)

    #search for the most relevant chunks based on the question
    reranker = {
        "model": "reranker",
        "query": question,
        "documents": chunks_content,
        "top_n": 5,
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