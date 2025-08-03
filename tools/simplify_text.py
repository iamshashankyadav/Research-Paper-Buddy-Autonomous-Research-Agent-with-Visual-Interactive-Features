from crewai_tools import tool
from typing import Dict, List
import requests

@tool("Simplify Section")
def simplify_text(chunks_by_section: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Simplifies each chunk of scientific text using LLaMA 3.1 via Ollama local server.

    Args:
        chunks_by_section (Dict[str, List[str]]): Section name → list of text chunks

    Returns:
        Dict[str, List[str]]: Section name → list of simplified chunk strings
    """
    simplified_output = {}

    for section, chunks in chunks_by_section.items():
        simplified_output[section] = []
        for chunk in chunks:
            try:
                prompt = (
                    "You are a helpful assistant that explains technical research papers in simple terms for undergraduate students.\n"
                    f"Simplify this text:\n{chunk}"
                )
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3:8b", "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                content = response.json().get("response", "").strip()
                simplified_output[section].append(content)
            except Exception as e:
                simplified_output[section].append(f"[Error simplifying chunk: {str(e)}]")

    return simplified_output
