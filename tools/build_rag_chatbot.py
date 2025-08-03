from crewai.tools import Tool
from typing import Dict, List
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Replace with Groq/Together wrapper if needed
import os
import shutil
from dotenv import load_dotenv
# --- Step 1: Prepare Embeddings & Vector Store ---
def build_vector_store(sections: Dict[str, str], persist_dir="rag_db"):
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    docs = []
    for section_title, content in sections.items():
        docs.append(Document(page_content=content, metadata={"section": section_title}))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Build Chroma store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore

# --- Step 2: Build QA Chain ---
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)

    # Use Groq LLM (e.g., LLaMA3-8B or Mistral)
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=2,
        model_name="llama3-70b-8192",  # You can also try: llama3-70b-8192 or gemma-7b-it
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain


# --- Step 3: Full tool wrapper ---
def rag_chatbot_function(content: Dict) -> str:
    """
    Sets up RAG pipeline and returns a chat-ready QA chain.
    content = {
        "sections": {"Intro": "...", "Methods": "..."}
    }
    """
    print("[✓] Building chatbot RAG pipeline...")
    sections = content.get("sections", {})
    if not sections:
        return "❌ No sections provided."

    vectorstore = build_vector_store(sections)
    qa_chain = build_qa_chain(vectorstore)

    # Interactive terminal test (optional)
    print("\n🧠 Ask questions about the paper! Type 'exit' to stop.\n")
    while True:
        query = input("📚 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.run(query)
        print("🤖 Bot:", result)
    
    return "✅ Chatbot setup complete. Use qa_chain.run(question) to query."

# --- Export Tool ---
build_rag_chatbot_tool = Tool(
    name="RAG Chatbot Builder",
    description="Builds a chatbot interface over the simplified paper using RAG with embeddings and Chroma.",
    func=rag_chatbot_function
)
