from crewai_tools import tool
from typing import Dict, List
import requests
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

@tool("Citation Lookup")
def fetch_citations(chunks_by_section: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Enhances simplified text chunks by fetching relevant citation info or definitions from Semantic Scholar.

    Args:
        chunks_by_section (Dict[str, List[str]]): Section name → list of simplified chunks

    Returns:
        Dict[str, List[str]]: Section name → list of annotated text with citations (where possible)
    """
    annotated_output = {}
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    stop_words = set(stopwords.words("english"))

    def extract_keywords(text: str, top_k: int = 6) -> str:
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]
        tagged = pos_tag(tokens)
        # Focus on nouns and proper nouns
        candidates = [word for word, tag in tagged if tag.startswith("NN")]
        freq = Counter(candidates)
        top_keywords = [word for word, _ in freq.most_common(top_k)]
        return " ".join(top_keywords)

    for section, chunks in chunks_by_section.items():
        annotated_output[section] = []
        for chunk in chunks:
            query = extract_keywords(chunk)

            try:
                response = requests.get(
                    base_url,
                    params={"query": query, "fields": "title,authors,url", "limit": 1}
                )
                response.raise_for_status()
                data = response.json()
                if data.get("data"):
                    paper = data["data"][0]
                    author_name = paper['authors'][0]['name'] if paper['authors'] else "Unknown Author"
                    citation = f"[{paper['title']} by {author_name}]({paper['url']})"
                    chunk += f"\n\nSource: {citation}"
            except Exception as e:
                chunk += f"\n\n[No citation found — {str(e)}]"

            annotated_output[section].append(chunk)

    return annotated_output
