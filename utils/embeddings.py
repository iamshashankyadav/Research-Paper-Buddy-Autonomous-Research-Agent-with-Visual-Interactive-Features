"""
utils/embeddings.py

Handles generating vector embeddings for text chunks.
Supports:
- Local SentenceTransformers (default)
- Groq API (placeholder for when Groq adds embeddings endpoint)
"""

from typing import List, Literal, Optional
import numpy as np


def embed_with_sentence_transformer(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True
) -> List[List[float]]:
    """
    Embed text using a SentenceTransformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("SentenceTransformers not installed. `pip install sentence-transformers`")

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, show_progress_bar=False)

    if normalize:
        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

    return vectors.tolist()


def embed_with_groq(
    texts: List[str],
    api_key: str,
    model_name: str = "groq-embedding-model",
    normalize: bool = True
) -> List[List[float]]:
    """
    Placeholder for Groq embeddings.
    Replace with actual API calls when available.
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # TODO: Replace this with real Groq embedding API request
    dim = 1024
    rng = np.random.default_rng()
    vectors = rng.normal(size=(len(texts), dim))

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

    return vectors.tolist()


def get_embeddings(
    texts: List[str],
    backend: Literal["sentence_transformer", "groq"] = "sentence_transformer",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    normalize: bool = True
) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts: List of text strings.
        backend: Which embedding backend to use ("sentence_transformer" or "groq").
        api_key: Required for Groq backend.
        model_name: Model name for backend.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        List of embedding vectors (list of floats).
    """
    if backend == "sentence_transformer":
        if not model_name:
            model_name = "all-MiniLM-L6-v2"
        return embed_with_sentence_transformer(texts, model_name=model_name, normalize=normalize)

    elif backend == "groq":
        return embed_with_groq(texts, api_key=api_key, model_name=model_name or "groq-embedding-model", normalize=normalize)

    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


if __name__ == "__main__":
    sample = ["This is a test.", "Embeddings are cool."]
    vecs = get_embeddings(sample, backend="sentence_transformer")
    print(f"Generated {len(vecs)} vectors, each of length {len(vecs[0])}")
