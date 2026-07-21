import requests
import base64
import os
import random
import string
from deepseek_ocr import extract_image_text
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("TEXT_TO_IMAGE_API_KEY")
URL = os.getenv("TEXT_TO_IMAGE_URL")

def get_image(question, files=None):

    if files is not None:
        context = extract_image_text(files)
        question = f"Context: {context}, Question: {question}"

    print(question)
    payload = {
        "model": "text-to-image",
        "prompt": f"{question}",
        "size": "200x200"
    }

    try:
        resp = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        base64_string = data["data"][0].get("b64_json")
        return base64_string
    except (requests.RequestException, KeyError) as e:
        print(f"Request failed: {e}\nBody: {resp.text if 'resp' in locals() else ''}")