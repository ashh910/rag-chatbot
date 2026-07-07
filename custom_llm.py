import requests

payload = {
            "model": "gpt-oss",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты краткий ассистент. Если ты не знаешь какой-то информации или не можешь точно её проверить, не упоминай её.",
                },
            ],
        }

class CustomLLM:
    def __init__(self, api_key, url, document_information = None):
        self.api_key = api_key
        self.url = url
        self.document_information = document_information


    def _call(self, question, document_information = None):

        payload["messages"].append({
            "role": "user",
            "content": question
        })

        if document_information is not None:
            payload["messages"].append({
                "role": "user",
                "content": f"the most relevant information extracted from documents: {document_information}"
            })
        
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        return reply
    