import asyncio
import json
from typing import Annotated

from openai import OpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.config import COORDINATOR_MODEL, SYNTHESIS_MODEL
from src.graph_store import GraphStore
from src.token_tracker import TokenTracker
from src.tools import query_qdrant_subgraph

PIPELINE_NAME = "3. Multi-Agent GraphRAG"

COORDINATOR_PROMPT = """Break down the following question into 2-4 focused sub-queries,
each targeting a specific entity or concept in a knowledge graph about the AI industry.

Return ONLY valid JSON:
{{
  "sub_queries": ["sub-query 1", "sub-query 2", "sub-query 3"],
  "reasoning": "brief explanation"
}}

Each sub-query should cover a different angle: organisations, people, models, funding.
Keep sub-queries short and in English.

Question: {query}"""

SYNTHESIS_SYSTEM = (
    "You are an AI analyst. You have received a condensed knowledge graph context "
    "gathered by parallel research agents. Answer the question precisely, citing "
    "specific entities and relationships. Be thorough but concise."
)


def _list_merge(a: list, b: list) -> list:
    return a + b


class State(TypedDict):
    query: str
    sub_queries: list[str]
    worker_results: Annotated[list[dict], _list_merge]
    condensed_context: str
    answer: str
    token_log: Annotated[list[dict], _list_merge]


def coordinator(state: State, tracker: TokenTracker) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model=COORDINATOR_MODEL,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": COORDINATOR_PROMPT.format(query=state["query"])}],
    )

    tracker.record(PIPELINE_NAME, "coordinator", COORDINATOR_MODEL, response.usage)

    try:
        parsed = json.loads(response.choices[0].message.content)
        sub_queries = parsed.get("sub_queries", [state["query"]])
    except json.JSONDecodeError:
        sub_queries = [state["query"]]

    print(f"\n[Coordinator] {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")

    return {
        "sub_queries": sub_queries,
        "token_log": [{"stage": "coordinator", "model": COORDINATOR_MODEL,
                       "input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens}],
    }


async def _worker(sub_query: str, worker_id: int) -> dict:
    print(f"  [Worker {worker_id}] Searching: '{sub_query}'")
    raw = await asyncio.to_thread(query_qdrant_subgraph.invoke, {"sub_query": sub_query})
    result = json.loads(raw) if isinstance(raw, str) else {"sub_query": sub_query, "facts": []}
    print(f"  [Worker {worker_id}] Found {len(result.get('facts', []))} entities")
    return result


async def _run_workers(sub_queries: list[str]) -> list[dict]:
    return list(await asyncio.gather(
        *[_worker(sq, i + 1) for i, sq in enumerate(sub_queries)]
    ))


def workers(state: State) -> dict:
    print(f"\n[Workers] Running {len(state['sub_queries'])} searches in parallel...")
    return {"worker_results": asyncio.run(_run_workers(state["sub_queries"]))}


def aggregator(state: State) -> dict:
    entities: dict[str, dict] = {}

    for result in state["worker_results"]:
        for fact in result.get("facts", []):
            eid = fact.get("entity_id", "")
            if eid not in entities:
                entities[eid] = fact
            else:
                known = {r["relation"] for r in entities[eid].get("relations", [])}
                for rel in fact.get("relations", []):
                    if rel["relation"] not in known:
                        entities[eid].setdefault("relations", []).append(rel)
                        known.add(rel["relation"])

    lines = [f"=== CONTEXT ({len(state['sub_queries'])} parallel searches) ===\n"]
    for entity in entities.values():
        lines.append(f"[{entity['type']}] {entity['name']}")
        lines.append(f"  {entity['description']}")
        for rel in entity.get("relations", [])[:6]:
            lines.append(f"  -> {rel['relation']}: {rel['name']} -- {rel['detail'][:100]}")
        lines.append("")

    total_raw = sum(len(r.get("facts", [])) for r in state["worker_results"])
    unique = len(entities)
    print(f"\n[Aggregator] {total_raw} facts -> {unique} unique entities ({total_raw - unique} duplicates removed)")

    return {"condensed_context": "\n".join(lines)}


def synthesizer(state: State, tracker: TokenTracker) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": f"{state['condensed_context']}\n\nQuestion: {state['query']}"},
        ],
    )

    tracker.record(PIPELINE_NAME, "synthesizer", SYNTHESIS_MODEL, response.usage)
    print(f"\n[Synthesizer] {response.usage.prompt_tokens} input tokens from condensed context")

    return {
        "answer": response.choices[0].message.content,
        "token_log": [{"stage": "synthesizer", "model": SYNTHESIS_MODEL,
                       "input": response.usage.prompt_tokens,
                       "output": response.usage.completion_tokens}],
    }


def build_graph(tracker: TokenTracker):
    builder = StateGraph(State)
    builder.add_node("coordinator", lambda s: coordinator(s, tracker))
    builder.add_node("workers", workers)
    builder.add_node("aggregator", aggregator)
    builder.add_node("synthesizer", lambda s: synthesizer(s, tracker))

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "workers")
    builder.add_edge("workers", "aggregator")
    builder.add_edge("aggregator", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def run(query: str, store: GraphStore, tracker: TokenTracker) -> str:
    graph = build_graph(tracker)
    final = graph.invoke({
        "query": query,
        "sub_queries": [],
        "worker_results": [],
        "condensed_context": "",
        "answer": "",
        "token_log": [],
    })
    return final["answer"]
