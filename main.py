# main.py

from crew.setup_crew import initialize_crew

if __name__ == "__main__":
    crew = initialize_crew()
    crew.kickoff()  # Starts the execution chain
