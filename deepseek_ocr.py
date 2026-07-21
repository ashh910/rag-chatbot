import requests
import os
from openai import OpenAI
from langchain.tools import tool
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_OCR_URL = os.getenv("EMBED_URL")
DEEPSEEK_OCR_API_KEY = os.getenv("DEEPSEEK_OCR_API_KEY")

client = OpenAI(
    api_key = DEEPSEEK_OCR_API_KEY,
    base_url = DEEPSEEK_OCR_URL,
    timeout = 3600
)

#IMAGE_URL = "https://i.imgur.com/x67a6us.jpeg"  # any plain text image

@tool
@lru_cache(maxsize=10)
def extract_image_text(image_link):

    ''' 
        You have access to a extract_image_text tool that extracts text in 
        the image from a provided image link and returns relevant excerpts.

        Only call extract_image_text when the user's question requires looking up 
        specific text information from an image. For greetings, small talk, 
        or questions you can already answer directly, respond in plain text 
        without calling any tool.

        extract_image_text takes one argument:
        - image_link: the link provided by the user that leads to the image with text.
    '''

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {"url": image_link}
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

    return(f'''
        This is the text extracted from an image without visual information.
        If some details are unclear, do not generate information for a user
        without concrete evidence. Simply notify the user that you only have
        access to text typed or written in the image.

        Extracted text:
        {(repr(response.choices[0].message.content))}
    ''')