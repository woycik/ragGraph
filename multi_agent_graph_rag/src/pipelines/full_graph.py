from openai import OpenAI

from src.config import SYNTHESIS_MODEL
from src.graph_store import GraphStore
from src.token_tracker import TokenTracker

PIPELINE_NAME = "2. Full-Graph RAG"

SYSTEM = (
    "You are a knowledge graph expert. You have been given a complete knowledge graph. "
    "Answer the question precisely, citing specific relationships where relevant."
)


def run(query: str, store: GraphStore, tracker: TokenTracker) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{store.get_full_graph_text()}\n\nQuestion: {query}"},
        ],
    )

    tracker.record(PIPELINE_NAME, "synthesis", SYNTHESIS_MODEL, response.usage)
    return response.choices[0].message.content
