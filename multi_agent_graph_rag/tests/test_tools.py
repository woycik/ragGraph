import sys
import warnings
import json
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.graph_store import GraphStore
from src.tools import set_graph_store, query_qdrant_subgraph, query_entity_by_name

store = GraphStore()
store.load_and_ingest("data/knowledge_graph.json")
set_graph_store(store)

result = query_qdrant_subgraph.invoke({"sub_query": "Who are the investors of Anthropic?"})
data = json.loads(result)
print(f"Sub-query: {data['sub_query']}")
print(f"Facts: {len(data['facts'])}")
for f in data["facts"]:
    print(f"  {f['name']} ({f['type']}) — {len(f['relations'])} relations")

result2 = query_entity_by_name.invoke({"entity_name": "openai"})
data2 = json.loads(result2)
print(f"\nOpenAI neighbors: {len(data2['neighbors'])}")
print("OK")
