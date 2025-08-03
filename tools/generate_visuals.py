from crewai_tools import tool
from typing import Dict, List
import tempfile
import os
import graphviz

@tool("Generate Visuals")
def generate_visuals(chunks_by_section: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Generates simple flowchart-style diagrams from method or process descriptions.
    Diagrams are saved as image files and file paths are returned alongside simplified content.

    Args:
        chunks_by_section (Dict[str, List[str]]): Section → list of simplified chunks

    Returns:
        Dict[str, List[str]]: Section → list of file paths to .png diagrams (one per chunk if applicable)
    """
    visual_output = {}

    for section, chunks in chunks_by_section.items():
        visual_output[section] = []
        if section.lower() not in ["methods", "methodology", "approach", "experiments"]:
            # Only create visuals for relevant sections
            visual_output[section] = ["[Visual generation skipped for non-method section]"] * len(chunks)
            continue

        for idx, chunk in enumerate(chunks):
            try:
                # Extract steps as nodes heuristically by splitting on linebreaks or periods
                steps = [step.strip() for step in chunk.split('.') if step.strip()]
                if len(steps) < 2:
                    visual_output[section].append("[Not enough steps for a diagram]")
                    continue

                dot = graphviz.Digraph(format='png')
                dot.attr(dpi='150')
                for i, step in enumerate(steps):
                    dot.node(str(i), step[:80])  # trim if too long
                    if i > 0:
                        dot.edge(str(i-1), str(i))

                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = os.path.join(tmpdir, f"{section}_{idx}.png")
                    dot.render(filename=out_path, cleanup=True)
                    visual_output[section].append(out_path + ".png")
            except Exception as e:
                visual_output[section].append(f"[Diagram error — {str(e)}]")

    return visual_output
