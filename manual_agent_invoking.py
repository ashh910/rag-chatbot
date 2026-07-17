from openai import OpenAI
from rag import search_documents
import json, re, os

from dotenv import load_dotenv
load_dotenv()
URL = os.getenv("URL")

history_initial_template = [
    {
        "role": "system", 
        "content": """
            You can use tools to answer questions. Available tools:
            - search_documents(question, is_uploaded_document) searches documents and returns relevant excerpts.
                - question: the user's question, as a plain string.
                - is_uploaded_document: true if the user is asking about a file they personally uploaded 
                in this conversation, false if asking about general/reference documents. Defaults to false 
                if not specified.

            Only use search_documents when the user's question requires looking up 
            specific information from documents. For greetings, small talk, 
            or questions you can already answer directly, respond in plain text immediately 
            without using any tool.

            When you need a tool, respond with ONLY this JSON on its own, nothing else:
            {"tool": "tool_name", "args": {"question": "...", "is_uploaded_document": true or false}}

            When you have the final answer, respond normally in plain text (no JSON).
        """
    },
    {
        "role": "system", 
        "content": "---------- History was cleared. ----------"
    }
]

kazllm_history = history_initial_template
alemllm_history = history_initial_template

def extract_tool_call(text: str):
    match = re.search(r'\{.*"tool".*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

def manual_agent_response(api_key, model, question, files_list = None):

    AVAILABLE_TOOLS = {"search_documents": search_documents}

    match model:
        case "alemllm":
            history_log = alemllm_history
        case "kazllm":
            history_log = kazllm_history
        case _:
            raise ValueError(f"Invalid model choice: {model}")
    
    if files_list is not None:
        history_log.append({"role": "system", "content": f"user uploaded following files: {files_list}" })

    history_log.append({"role": "user", "content": question})

    client = OpenAI(
        api_key=api_key,
        base_url=URL
    )

    MAX_ITERATIONS = 5
    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            messages=history_log,
            temperature=0.2
        )
        content = response.choices[0].message.content
        call = extract_tool_call(content)
        print(f"[DEBUG] model content: {content!r}")
        print(f"[DEBUG] extracted call: {call!r}")

        if not call:
            history_log.append({"role": "assistant", "content": content})
            return content

        tool_name = call["tool"]
        args = call.get("args", {})

        if tool_name not in AVAILABLE_TOOLS:
            history_log.append({"role": "assistant", "content": content})
            return content  # model hallucinated a tool that doesn't exist — bail out

        result = AVAILABLE_TOOLS[tool_name].invoke(args)

        history_log.append({"role": "assistant", "content": content})
        history_log.append({
            "role": "system",
            "content": f"You have used the tool provided by the system. Tool result: {result}"
        })

        if len(history_log) > 10:
            del history_log[1:3]