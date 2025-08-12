"""
utils/latex_generator.py

Generates LaTeX code and compiles it to PDF using pdflatex.
"""


from pathlib import Path
from typing import List, Dict
from utils.config import OUTPUT_DIR
"""
utils/latex_generator.py

Generates LaTeX code for the booklet.
"""

def generate_booklet_pdf(
    title: str,
    sections: List[Dict[str, str]],
    images: List[str],
    out_dir: Path = OUTPUT_DIR
) -> str:
    """
    Generate LaTeX from sections and save as .tex file only.

    Args:
        title (str): The booklet title
        sections (List[Dict]): List of {"heading":..., "text":...}
        images (List[str]): List of image paths to include
        out_dir (Path): Where to save the .tex file

    Returns:
        str: Path to the generated .tex file
    """
    tex_path = Path(out_dir) / "booklet.tex"
    latex_content = _build_latex(title, sections, images)
    tex_path.write_text(latex_content, encoding="utf-8")
    return str(tex_path)


def _build_latex(title: str, sections: List[Dict[str, str]], images: List[str]) -> str:
    """
    Build LaTeX content for the booklet.
    """
    latex = [
        r"\documentclass[12pt,a4paper]{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{hyperref}",
        r"\title{" + title + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    for i, sec in enumerate(sections):
        latex.append(r"\section{" + sec["heading"] + "}")
        latex.append(sec["text"])
        if i < len(images):
            latex.append(r"\begin{figure}[h]")
            latex.append(r"\centering")
            latex.append(r"\includegraphics[width=0.8\textwidth]{" + images[i] + "}")
            latex.append(r"\end{figure}")

    latex.append(r"\end{document}")
    return "\n".join(latex)
