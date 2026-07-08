from pydantic import BaseModel
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from rag import search_documents
from custom_llm import my_chatbot
from langchain.agents.structured_output import ToolStrategy

class Answer(BaseModel):
    summary: str
    confidence: float

agent = create_agent(
    model=my_chatbot,
    tools=[search_documents],
    response_format=Answer,
    checkpointer = InMemorySaver(),
)

thread_config = { 
    "configurable":
    {
        "thread_id": "1",
    }
}

def agent_responses(question):
    
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