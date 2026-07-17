from pydantic import BaseModel
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from rag import search_documents
from langchain_openai import ChatOpenAI
import uuid, os
from dotenv import load_dotenv
load_dotenv()

GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
URL = os.getenv("URL")

Models_supporting_tools = {"gpt-oss", "gemma4", "qwen3-6"}

class Answer(BaseModel):
    summary: str
    confidence: float

gpt_oss_checkpointer = InMemorySaver()
qwen3_6_checkpointer = InMemorySaver()
gemma4_checkpointer = InMemorySaver()

plain_text_chat_history = []
plain_text_current_model = None

thread_config = { 
    "configurable":
    {
        "thread_id": "67",
    }
}

def get_chatbot(model_choice="gpt-oss", api_key=GPT_OSS_API_KEY, URL=URL):
    return ChatOpenAI(
            model=model_choice,
            api_key=api_key,
            base_url=URL,
            temperature=0.2,
        )


def agent_response(api_key, model_choice, question, file=None):
    match model_choice: 
        case "gpt-oss":
            checkpointer = gpt_oss_checkpointer
        case "gemma4":
            checkpointer = gemma4_checkpointer
        case "qwen3-6":
            checkpointer = qwen3_6_checkpointer
        case _:
            raise ValueError(f"Invalid model choice: {model_choice}")

    my_chatbot = get_chatbot(model_choice=model_choice, api_key=api_key)
    
    agent = create_agent(
        model=my_chatbot,
        tools=[search_documents],
        checkpointer=checkpointer
    )

    result = agent.invoke({"messages": [
            {
                "role": "user", 
                "content": question
            }
        ]}, thread_config,)

    prior_state = agent.get_state(thread_config)
    prior_messages = prior_state.values.get("messages", []) if prior_state.values else []
    prior_count = len(prior_messages)

    new_messages = result["messages"][prior_count:]

    real_tool_called = any(
        getattr(m, "tool_calls", None) and
        any(tc["name"] != "Answer" for tc in m.tool_calls)
        for m in new_messages
    )

    if real_tool_called and result.get("structured_response"):
        return result["structured_response"].summary
    else:
        return result["messages"][-1].content