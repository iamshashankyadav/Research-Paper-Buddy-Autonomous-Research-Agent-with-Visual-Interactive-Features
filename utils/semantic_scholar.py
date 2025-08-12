"""
utils/semantic_scholar.py

Fetches paper metadata from Semantic Scholar API and formats citations.
"""

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_paper_by_title(title: str, limit: int = 1):
    """
    Search for a paper by title using Semantic Scholar API.
    """
    params = {
        "query": title,
        "limit": limit,
        "fields": "title,authors,year,url"
    }
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def format_citation(paper_data: dict) -> str:
    """
    Format a paper dict into APA-style citation.
    """
    authors = ", ".join([a["name"] for a in paper_data.get("authors", [])])
    year = paper_data.get("year", "n.d.")
    title = paper_data.get("title", "")
    url = paper_data.get("url", "")
    return f"{authors} ({year}). {title}. {url}"
import random

def _enrich_with_citation(topic: str) -> str:
    """
    Placeholder: Return a mock citation link for the given topic.
    Later, connect to Semantic Scholar API to fetch real references.
    """
    fake_links = [
        f"https://www.semanticscholar.org/search?q={topic.replace(' ', '%20')}",
        f"https://scholar.google.com/scholar?q={topic.replace(' ', '+')}",
        f"https://arxiv.org/search/?query={topic.replace(' ', '+')}&searchtype=all"
    ]
    return random.choice(fake_links)
