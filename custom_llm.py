import requests
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
URL = os.getenv("URL")

my_chatbot = ChatOpenAI(
    model="gpt-oss",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("URL"),
    temperature=0,
)