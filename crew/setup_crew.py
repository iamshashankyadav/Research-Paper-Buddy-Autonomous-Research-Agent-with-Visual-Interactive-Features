from crewai import Crew
from agents.reader_agent import reader_agent
from agents.segmenter_agent import segmenter_agent
from agents.explainer_agent import explainer_agent
from agents.citation_agent import citation_agent
from agents.visualizer_agent import visualizer_agent
from agents.booklet_agent import booklet_agent
from agents.chatbot_agent import chatbot_agent

def initialize_crew():
    return Crew(
        agents=[
            reader_agent,
            segmenter_agent,
            explainer_agent,
            citation_agent,
            visualizer_agent,
            booklet_agent,
            chatbot_agent
        ],
        process="sequential",
        verbose=True
    )
