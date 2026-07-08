from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API = os.getenv("GEMINI_API")

resource_folder = "RAG Files"
persist_directory = "RAG Files/chrome/"

embedding = GoogleGenerativeAIEmbeddings(
        model = "gemini-embedding-2-preview",
        google_api_key = GEMINI_API
    )


#create a vectorstore from the documents in the resource folder
def create_vectorstore(resource_folder=resource_folder, persist_directory=persist_directory):
    resources = []

    #load all text files from the resource folder
    for filename in os.listdir(resource_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(resource_folder, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            resources.extend(loader.load())

    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = [" ", "\n", "\n\n", "\t", ",", ".", "!", "?", ";", ":"],
        chunk_size = 300,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_documents(resources)

    #create a vectorstore from the chunks and store it in the persist directory
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embedding,
        persist_directory = persist_directory
    )


if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    create_vectorstore()
else:
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )


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

    #search for the most relevant chunks based on the question
    relevant_chunks = vectordb.max_marginal_relevance_search(
        question,
        k=2,
        fetch_k=3
    )

    #return the relevant chunks to be used by the LLM for generating a response
    return relevant_chunks