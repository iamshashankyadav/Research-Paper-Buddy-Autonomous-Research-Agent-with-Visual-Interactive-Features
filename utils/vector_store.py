# utils/vector_store.py
from typing import List, Tuple
from pathlib import Path
from config import HF_API_KEY, EMBEDDING_MODEL
from langchain_huggingface.embeddings.huggingface_endpoint import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

import os


def build_vectorstore(texts, persist_directory, index_name):
    embedder = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_API_KEY
    )
    vectordb = Chroma.from_texts(
        texts, embedding=embedder,
        persist_directory=persist_directory,
        collection_name=index_name
    )
   
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
