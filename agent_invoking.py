from langchain.agents import create_agent
import zapros
import custom_llm

agent = create_agent(model = zapros.my_chatbot, 
                     tools = [zapros.search_documents],
                     system_prompt = custom_llm.payload["messages"][0]["content"])

def agent_responses():
    reply = agent.invoke({"messages": custom_llm.payload["messages"]})
    return reply