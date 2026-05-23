from openai import OpenAI

from src.config import SYNTHESIS_MODEL
from src.graph_store import GraphStore
from src.token_tracker import TokenTracker

PIPELINE_NAME = "1. Standard RAG"

SYSTEM = "You are a helpful assistant. Answer the question based strictly on the provided context."


def run(query: str, store: GraphStore, tracker: TokenTracker) -> str:
    matches = store.search_entities(query, top_k=5)
    context = "\n\n".join(
        f"[{store.nodes[m['node_id']]['type']}] {store.nodes[m['node_id']]['name']}: "
        f"{store.nodes[m['node_id']]['description']}"
        for m in matches
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )

    tracker.record(PIPELINE_NAME, "synthesis", SYNTHESIS_MODEL, response.usage)
    return response.choices[0].message.content
