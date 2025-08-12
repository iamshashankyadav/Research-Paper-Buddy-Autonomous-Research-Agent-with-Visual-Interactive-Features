# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === API KEYS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")   # HuggingFace Inference API key

if not GROQ_API_KEY:
    # keep non-blocking - some devs run without GROQ during unit tests
    print("Warning: GROQ_API_KEY not set. Set it in .env for real LLM calls.")

if not HF_API_KEY:
    print("Warning: HF_API_KEY not set. You will need it to compute embeddings via HuggingFace Inference API.")

# === Model Settings ===
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
