"""
chains/booklet_chain.py

Pipeline for generating a research booklet from a PDF:
1. Extract & clean text
2. Split into sections
3. Summarize each section with Groq LLM
4. Enrich with Semantic Scholar citations
5. Generate rule-based visuals
6. Compile final PDF via LaTeX
"""

from pathlib import Path
from typing import Optional, List, Dict

from utils.pdf_loader import extract_text, split_into_sections
from utils.semantic_scholar import search_paper_by_title, format_citation
from utils.visualization import flow_diagram, bar_chart
from utils.latex_generator import generate_booklet_pdf
from utils.config import GROQ_API_KEY

from langchain_groq import ChatGroq


def _summarize_section(text: str, llm) -> str:
    """
    Use Groq LLM to summarize a given section.
    """
    prompt = f"""Summarize the following research paper section into simple,
clear language for a student audience. Keep technical terms but explain them if needed.

Section:
{text}
"""
    response = llm.invoke(prompt)
    return response.content.strip()



def _enrich_with_citation(section_heading: str) -> Optional[str]:
    """
    Look up related paper via Semantic Scholar and return formatted citation.
    """
    result = search_paper_by_title(section_heading)
    if result and "data" in result and len(result["data"]) > 0:
        return format_citation(result["data"][0])
    return None


def _generate_visual(section_text: str) -> Optional[str]:
    """
    Generate a diagram or chart based on section content (rule-based).
    """
    lines = section_text.split("\n")
    # Simple detection for methodology steps
    steps = [line.strip("-•0123456789. ") for line in lines if line.strip().startswith(("-", "•", "1", "2", "3"))]
    if len(steps) >= 3:
        return flow_diagram(steps)

    # Simple detection for numeric results
    numeric_lines = [(l.split(":")[0], l.split(":")[1]) for l in lines if ":" in l and any(c.isdigit() for c in l)]
    if 1 <= len(numeric_lines) <= 6:
        labels, vals = [], []
        for label, val in numeric_lines:
            try:
                labels.append(label.strip())
                vals.append(float(val.strip().replace("%", "")))
            except ValueError:
                continue
        if labels and vals:
            return bar_chart(labels, vals, title="Results")
    return None

from pathlib import Path
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
import os

from utils.latex_generator import generate_booklet_pdf
from utils.pdf_loader import extract_text, split_into_sections
from utils.visualization import _generate_visual
from utils.semantic_scholar import _enrich_with_citation

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_booklet_from_pdf(
    pdf_path: str,
    out_dir: Optional[str] = None
) -> tuple:
    """
    Full pipeline: PDF → booklet LaTeX + images.

    Returns:
        (tex_path, image_paths)
    """
    if out_dir is None:
        out_dir = "outputs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )

    # Step 1: Extract text & split into sections
    raw_text = extract_text(pdf_path)
    sections_raw = split_into_sections(raw_text)

    sections_processed: List[Dict[str, str]] = []
    images: List[str] = []

    for sec in sections_raw:
        # Step 2: Summarize
        summary = llm.invoke(
            f"Summarize the following section in clear, simple terms:\n\n{sec['text']}"
        ).content

        # Step 3: Add citation (optional)
        citation = _enrich_with_citation(sec["heading"])
        if citation:
            summary += f"\n\nFurther reading: {citation}"

        # Step 4: Generate visual (optional)
        img_path = _generate_visual(sec["text"], out_dir)
        if img_path:
            images.append(str(img_path))

        sections_processed.append({
            "heading": sec["heading"],
            "text": summary
        })

    # Step 5: Generate LaTeX file
    title = Path(pdf_path).stem + " — Research Booklet"
    tex_path = generate_booklet_pdf(title, sections_processed, images, out_dir=out_dir)
    return str(tex_path), images



