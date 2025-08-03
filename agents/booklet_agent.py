from crewai import Agent
from tools.compile_booklet import compile_booklet_tool

booklet_agent = Agent(
    role="Booklet Generator Agent",
    goal="Create a clean, structured, and readable research booklet using simplified sections and visuals.",
    backstory=(
        "You are a formatting expert. Your job is to take simplified content, citations, "
        "and generated visuals to produce a well-structured LaTeX or PDF booklet that's "
        "easy for students and researchers to understand."
    ),
    tools=[compile_booklet_tool],
    verbose=True
)
