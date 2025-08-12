"""
utils/visualization.py

Generates basic flow diagrams and bar charts using Matplotlib.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

OUTPUT_DIR = Path("outputs/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from pathlib import Path
import uuid

def _generate_visual(section_text: str, out_dir: str = "outputs/diagrams") -> str:
    """
    Placeholder: Generate a simple bar chart showing word frequency in the section.
    Later, replace with more relevant visualizations.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    words = section_text.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Take top 5 frequent words
    top_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    labels, values = zip(*top_items) if top_items else ([], [])

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Top Words in Section")
    plt.tight_layout()

    img_path = Path(out_dir) / f"visual_{uuid.uuid4().hex}.png"
    plt.savefig(img_path)
    plt.close()

    return str(img_path)

def flow_diagram(steps: List[str], out_path: Optional[str] = None) -> str:
    """
    Generate a simple vertical flow diagram from a list of steps.
    """
    fig, ax = plt.subplots(figsize=(5, len(steps) * 0.8))
    ax.axis("off")

    for i, step in enumerate(steps):
        ax.text(0.5, 1 - i * 0.15, step,
                ha="center", va="center",
                fontsize=12, bbox=dict(boxstyle="round", facecolor="lightblue"))

    if not out_path:
        out_path = OUTPUT_DIR / "flow_diagram.png"
    else:
        out_path = Path(out_path)

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def bar_chart(labels: List[str], values: List[float], title: str = "", out_path: Optional[str] = None) -> str:
    """
    Generate a basic bar chart from labels and values.
    """
    fig, ax = plt.subplots()
    ax.bar(labels, values, color="skyblue")
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if not out_path:
        out_path = OUTPUT_DIR / "bar_chart.png"
    else:
        out_path = Path(out_path)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return str(out_path)
