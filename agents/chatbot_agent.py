from crewai import Agent
from tools.build_rag_chatbot import build_rag_chatbot_tool

chatbot_agent = Agent(
    role="Chatbot Agent",
    goal="Enable an interactive Q&A interface over the simplified research paper using RAG.",
    backstory="You're an intelligent assistant that uses retrieval-augmented generation to answer any question about the paper.",
    tools=[build_rag_chatbot_tool],
    verbose=True
)
