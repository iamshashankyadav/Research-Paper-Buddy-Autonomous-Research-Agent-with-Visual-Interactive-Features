"""
chains/chatbot_chain.py

Conversational RAG chatbot with LangChain + Groq LLM.
"""

from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from utils.config import GROQ_API_KEY


def build_chatbot(retriever):
    """
    Build a conversational retrieval chain using Groq LLM.

    Args:
        retriever: LangChain retriever object (e.g., from ChromaDB).

    Returns:
        LangChain ConversationalRetrievalChain instance.
    """
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
