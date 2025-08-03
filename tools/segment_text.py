from crewai_tools import tool
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load transformer model once globally
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@tool("Segment Text")
def segment_text(sections: Dict[str, str], max_tokens_per_chunk: int = 200) -> Dict[str, List[str]]:
    """
    Breaks long section texts into semantically coherent chunks using sentence embeddings and KMeans.

    Args:
        sections (Dict[str, str]): A dictionary of section titles and text.
        max_tokens_per_chunk (int): Maximum tokens (approx.) per chunk.

    Returns:
        Dict[str, List[str]]: Section name → List of chunked texts.
    """
    segmented = {}

    for title, text in sections.items():
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            segmented[title] = [text.strip()]
            continue

        # Embed all sentences
        embeddings = embedding_model.encode(sentences)

        # Estimate number of chunks
        est_chunks = max(1, int(len(sentences) * 15 / max_tokens_per_chunk))
        if est_chunks >= len(sentences):
            segmented[title] = [text.strip()]
            continue

        # Cluster sentences into groups
        kmeans = KMeans(n_clusters=est_chunks, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # Group sentences by cluster label
        chunks = {}
        for label, sent in zip(labels, sentences):
            chunks.setdefault(label, []).append(sent)

        # Join and store chunks
        chunk_texts = [" ".join(group).strip() for group in chunks.values()]
        segmented[title] = chunk_texts

    return segmented
