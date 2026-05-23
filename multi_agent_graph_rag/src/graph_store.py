import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.config import EMBEDDING_MODEL, QDRANT_COLLECTION, TOP_K_NODES, VECTOR_SIZE


class GraphStore:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.nodes: dict[str, dict] = {}
        self.adjacency: dict[str, list[dict]] = {}

    def load_and_ingest(self, json_path: str | Path) -> None:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.nodes = {n["id"]: n for n in data["nodes"]}

        for edge in data["edges"]:
            src, tgt = edge["source"], edge["target"]
            self.adjacency.setdefault(src, []).append({
                "target": tgt,
                "relation": edge["relation"],
                "description": edge["description"],
            })
            self.adjacency.setdefault(tgt, []).append({
                "target": src,
                "relation": f"REVERSE_{edge['relation']}",
                "description": edge["description"],
            })

        self.client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        node_ids = list(self.nodes.keys())
        texts = [
            f"{self.nodes[nid]['name']} ({self.nodes[nid]['type']}): {self.nodes[nid]['description']}"
            for nid in node_ids
        ]
        embeddings = list(self.embedder.embed(texts))

        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=i,
                    vector=emb.tolist(),
                    payload={
                        "node_id": node_ids[i],
                        "name": self.nodes[node_ids[i]]["name"],
                        "type": self.nodes[node_ids[i]]["type"],
                    },
                )
                for i, emb in enumerate(embeddings)
            ],
        )

        print(f"[GraphStore] Loaded {len(self.nodes)} nodes, {len(data['edges'])} edges.")

    def search_entities(self, query: str, top_k: int = TOP_K_NODES) -> list[dict]:
        (vec,) = list(self.embedder.embed([query]))
        hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec.tolist(),
            limit=top_k,
        ).points
        return [
            {
                "node_id": h.payload["node_id"],
                "name": h.payload["name"],
                "type": h.payload["type"],
                "score": round(h.score, 4),
            }
            for h in hits
        ]

    def get_subgraph(self, entity_id: str) -> dict:
        if entity_id not in self.nodes:
            return {"error": f"Entity '{entity_id}' not found"}

        node = self.nodes[entity_id]
        neighbors = [
            {
                "entity_id": nb["target"],
                "name": self.nodes.get(nb["target"], {}).get("name", nb["target"]),
                "type": self.nodes.get(nb["target"], {}).get("type", "Unknown"),
                "relation": nb["relation"],
                "description": nb["description"],
            }
            for nb in self.adjacency.get(entity_id, [])
        ]

        return {
            "entity_id": entity_id,
            "name": node["name"],
            "type": node["type"],
            "description": node["description"],
            "neighbors": neighbors,
        }

    def get_full_graph_text(self) -> str:
        lines = ["=== KNOWLEDGE GRAPH ===\n", "--- ENTITIES ---"]
        for node in self.nodes.values():
            lines.append(f"[{node['type']}] {node['name']} (id:{node['id']})\n  {node['description']}")

        lines.append("\n--- RELATIONSHIPS ---")
        for src_id, edges in self.adjacency.items():
            src_name = self.nodes.get(src_id, {}).get("name", src_id)
            for e in edges:
                if e["relation"].startswith("REVERSE_"):
                    continue
                tgt_name = self.nodes.get(e["target"], {}).get("name", e["target"])
                lines.append(f"{src_name} --[{e['relation']}]--> {tgt_name}")
                lines.append(f"  {e['description']}")

        return "\n".join(lines)
