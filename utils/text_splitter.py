"""
utils/text_splitter.py

Chunk text using LangChain's built-in text splitters.
Default: RecursiveCharacterTextSplitter for general text,
and TokenTextSplitter for token-accurate splitting.
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter


def split_text_recursive(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.
    Good for general documents with paragraphs and headings.

    Args:
        text: Input text.
        chunk_size: Target chunk size (characters).
        chunk_overlap: Overlap between chunks (characters).
        separators: Optional list of separators for splitting hierarchy.

    Returns:
        List of chunk strings.
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return splitter.split_text(text)


def split_text_tokens(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    model_name: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Split text by token count using TokenTextSplitter.
    Useful for strict token limits.

    Args:
        text: Input text.
        chunk_size: Target chunk size (tokens).
        chunk_overlap: Overlap between chunks (tokens).
        model_name: Model encoding name for tiktoken.

    Returns:
        List of chunk strings.
    """
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=model_name  # LangChain handles tiktoken usage internally
    )
    return splitter.split_text(text)
