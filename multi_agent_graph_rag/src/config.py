import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "graph_rag")

COORDINATOR_MODEL = "gpt-4o-mini"
SYNTHESIS_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
TOP_K_NODES = 3


def validate_config():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
