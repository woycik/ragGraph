import json

from langchain_core.tools import tool

_graph_store = None


def set_graph_store(gs) -> None:
    global _graph_store
    _graph_store = gs


@tool
def query_qdrant_subgraph(sub_query: str) -> str:
    if _graph_store is None:
        return json.dumps({"error": "GraphStore not initialised"})

    matches = _graph_store.search_entities(sub_query, top_k=3)
    seen: set[str] = set()
    facts = []

    for match in matches:
        nid = match["node_id"]
        if nid in seen:
            continue
        seen.add(nid)

        sg = _graph_store.get_subgraph(nid)
        facts.append({
            "entity_id": nid,
            "name": sg["name"],
            "type": sg["type"],
            "description": sg["description"],
            "relations": [
                {
                    "name": nb["name"],
                    "type": nb["type"],
                    "relation": nb["relation"],
                    "detail": nb["description"],
                }
                for nb in sg["neighbors"]
                if not nb["relation"].startswith("REVERSE_")
            ],
            "relevance_score": match["score"],
        })

    return json.dumps({"sub_query": sub_query, "facts": facts}, ensure_ascii=False, indent=2)


@tool
def query_entity_by_name(entity_name: str) -> str:
    if _graph_store is None:
        return json.dumps({"error": "GraphStore not initialised"})

    nid = entity_name.lower().replace(" ", "_")
    if nid not in _graph_store.nodes:
        matches = _graph_store.search_entities(entity_name, top_k=1)
        if not matches:
            return json.dumps({"error": f"Entity '{entity_name}' not found"})
        nid = matches[0]["node_id"]

    return json.dumps(_graph_store.get_subgraph(nid), ensure_ascii=False, indent=2)


WORKER_TOOLS = [query_qdrant_subgraph, query_entity_by_name]
