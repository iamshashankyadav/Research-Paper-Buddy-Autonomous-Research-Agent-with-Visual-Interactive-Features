import os
import io
import zipfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF reading

from langchain_groq import ChatGroq
from langchain_huggingface.embeddings.huggingface_endpoint import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.config import OUTPUT_DIR
from chains.booklet_chain import generate_booklet_from_pdf

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Autonomous RAG with Groq + HuggingFace", layout="wide")

# =========================
# Load API Keys
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set in .env")
if not HF_API_KEY:
    st.error("❌ HF_API_KEY not set in .env")

# =========================
# Helper Functions
# =========================
def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file path."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def build_vectorstore(texts, persist_directory="vectorstore", index_name="rag_index"):
    """Builds Chroma vectorstore using HuggingFace Inference API embeddings."""
    embedder = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_API_KEY
    )
    vectordb = Chroma.from_texts(
        texts,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_name=index_name
    )
    
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def run_rag_query(query, retriever):
    """Runs a RAG query using Groq LLM + Chroma retriever."""
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    return response.content

# =========================
# Streamlit UI
# =========================
st.title("📄 Autonomous RAG App")
st.write("Upload a PDF, index it with free HuggingFace embeddings, chat with it using Groq LLM, and generate a LaTeX booklet.")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file:
    # Save uploaded PDF
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = uploads_dir / uploaded_file.name
    with open(saved_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📄 Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(saved_pdf_path)

    with st.spinner("✂️ Splitting text..."):
        chunks = split_text(pdf_text)

    with st.spinner("📦 Building vectorstore (this may take a few seconds)..."):
        retriever = build_vectorstore(chunks)

    st.success("✅ PDF processed and indexed!")

    # =========================
    # Booklet Generation Button
    # =========================
    if st.button("Generate Simplified Booklet (.tex + images)"):
        with st.spinner("Generating booklet (LLM + citations + visuals)..."):
            try:
                tex_path, image_paths = generate_booklet_from_pdf(
                    str(saved_pdf_path),
                    out_dir=str(OUTPUT_DIR)
                )
                st.success("Booklet generated (LaTeX).")

                # Provide .tex download
                with open(tex_path, "rb") as f:
                    st.download_button(
                        "Download booklet.tex",
                        f,
                        file_name=Path(tex_path).name
                    )

                # Also make a zip of .tex + images
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    zf.write(tex_path, arcname=Path(tex_path).name)
                    for img in image_paths:
                        zf.write(img, arcname=Path("diagrams") / Path(img).name)
                buf.seek(0)
                st.download_button(
                    "Download package (.zip)",
                    buf,
                    file_name=f"{Path(tex_path).stem}_package.zip"
                )
            except Exception as e:
                st.error(f"Booklet generation failed: {e}")

    # =========================
    # Q&A Interface
    # =========================
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("🤖 Generating answer..."):
            answer = run_rag_query(query, retriever)
        st.markdown("### Answer:")
        st.write(answer)
