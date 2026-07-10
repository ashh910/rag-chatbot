from pydantic import BaseModel
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from rag import search_documents
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()

GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
URL = os.getenv("URL")

Models_supporting_tools = {"gpt-oss", "gemma4", "qwen3-6"}

class Answer(BaseModel):
    summary: str
    confidence: float

checkpointer = InMemorySaver()

plain_text_chat_history = []
plain_text_current_model = None

thread_config = { 
    "configurable":
    {
        "thread_id": "1",
    }
}

def get_chatbot(model_choice="gpt-oss", api_key=GPT_OSS_API_KEY, URL=URL):
    return ChatOpenAI(
            model=model_choice,
            api_key=api_key,
            base_url=URL,
            temperature=0,
        )



def get_response(question, model_choice, api_key):
    supports_tools = model_choice in Models_supporting_tools

    if supports_tools:
        return agent_response(question, model_choice, api_key)
    else:
        return avarage_response(question, model_choice, api_key)


def agent_response(question, model_choice, api_key):
    
    my_chatbot = get_chatbot(model_choice=model_choice, api_key=api_key)
    
    agent = create_agent(
        model=my_chatbot,
        tools=[search_documents],
        response_format=Answer,
        checkpointer = checkpointer,
    )

    result = agent.invoke({"messages": [
            {
                "role": "user", 
                "content": question
            }
        ]}, thread_config,)

    if "structured_response" in result:
        #return the structured response from the model
        return result["structured_response"].summary 
    else:
        #model replied in plain text instead of using the structured tool
        return result["messages"][-1].content


def avarage_response(question, model_choice, api_key):
    global plain_text_current_model
    my_chatbot = get_chatbot(model_choice=model_choice, api_key=api_key)

    if plain_text_current_model != model_choice:
        plain_text_chat_history.clear()
        plain_text_current_model = model_choice

    plain_text_chat_history.append(HumanMessage(content=question))
    result = my_chatbot.invoke(plain_text_chat_history)
    return result.content
    plain_text_chat_history.append(AIMessage(content=result.content))