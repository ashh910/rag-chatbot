import requests, os, base64
from openai import OpenAI
from langchain.tools import tool
from langchain_core.documents import Document
from functools import lru_cache
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

DEEPSEEK_OCR_URL = os.getenv("EMBED_URL")
DEEPSEEK_OCR_API_KEY = os.getenv("DEEPSEEK_OCR_API_KEY")

client = OpenAI(
    api_key = DEEPSEEK_OCR_API_KEY,
    base_url = DEEPSEEK_OCR_URL,
    timeout = 3600
)

@tool
@lru_cache(maxsize=50)
def extract_image_file_text(image):
    
    ''' 
        You have access to a extract_image_text tool that extracts text in 
        the image from a provided image link and returns relevant excerpts.

        Only call extract_image_text when the user's question requires looking up 
        specific text information from an image. For greetings, small talk, 
        or questions you can already answer directly, respond in plain text 
        without calling any tool.

        extract_image_text takes one argument:
        - image: document uploaded by the user (accepted_files) IF it ends with 
        ".png" or ".jpeg"
    '''
    
    upload_folder = "RAG Files/uploaded files"

    if type(image) != str:
        image_name = image.filename
    else:
        image_name = image

    path = os.path.join(upload_folder, image_name)
    with open(path, "rb") as f:
        file = f.read()
        b64_code = base64.b64encode(file).decode("utf-8")
        ext = image_name.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        b64_link = f"data:{mime};base64,{b64_code}"

  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {"url": b64_link}
                },
                {
                    "type": "text",
                    "text": "Free OCR.",
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model = "deepseek-ocr",
        messages=messages,
        max_tokens=2048,
        extra_body={
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
            },
        },
    )

    return(response.choices[0].message.content)


def extract_image_file_text_for_tool(image):
    
    ''' 
        You have access to a extract_image_text tool that extracts text in 
        the image from a provided image link and returns relevant excerpts.

        Only call extract_image_text when the user's question requires looking up 
        specific text information from an image. For greetings, small talk, 
        or questions you can already answer directly, respond in plain text 
        without calling any tool.

        extract_image_text takes one argument:
        - image: document uploaded by the user (accepted_files) IF it ends with 
        ".png" or ".jpeg"
    '''
    
    upload_folder = "RAG Files/uploaded files"

    if type(image) != str:
        image_name = image.filename
    else:
        image_name = image

    path = os.path.join(upload_folder, image_name)
    with open(path, "rb") as f:
        file = f.read()
        b64_code = base64.b64encode(file).decode("utf-8")
        ext = image_name.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        b64_link = f"data:{mime};base64,{b64_code}"
        
  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {"url": b64_link}
                },
                {
                    "type": "text",
                    "text": "Free OCR.",
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model = "deepseek-ocr",
        messages=messages,
        max_tokens=2048,
        extra_body={
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
            },
        },
    )

    return(response.choices[0].message.content)