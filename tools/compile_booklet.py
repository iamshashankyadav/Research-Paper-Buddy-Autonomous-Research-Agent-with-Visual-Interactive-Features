from crewai.tools import Tool
import os
from fpdf import FPDF
from typing import Dict, List

class BookletPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, self.title, ln=True, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"{title}", ln=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, body)
        self.ln()

    def add_image(self, image_path):
        if os.path.exists(image_path):
            self.image(image_path, w=170)
            self.ln(10)

def compile_booklet_function(content: Dict) -> str:
    """
    Generates a research booklet PDF from provided section content and visuals.
    content = {
        "title": "Paper Title",
        "sections": {"Intro": "....", "Methods": "..."},
        "visuals": ["flowchart1.png", "conceptmap.png"]
    }
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "booklet.pdf")

    pdf = BookletPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_title(content.get("title", "Research Booklet"))
    pdf.title = content.get("title", "Research Booklet")

    # Write each section
    for section, body in content.get("sections", {}).items():
        pdf.chapter_title(section)
        pdf.chapter_body(body)

    # Add visuals
    visuals: List[str] = content.get("visuals", [])
    if visuals:
        pdf.chapter_title("Visual Diagrams")
        for image_path in visuals:
            pdf.add_image(image_path)

    pdf.output(pdf_path)
    print(f"[✓] Booklet generated at: {pdf_path}")
    return pdf_path

compile_booklet_tool = Tool(
    name="Booklet Compiler",
    description="Generates a formatted PDF booklet from simplified sections and visuals",
    func=compile_booklet_function
)
