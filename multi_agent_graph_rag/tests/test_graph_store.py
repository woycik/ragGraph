import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.graph_store import GraphStore

store = GraphStore()
store.load_and_ingest("data/knowledge_graph.json")

results = store.search_entities("founders of Anthropic")
print("Search results:")
for r in results:
    print(f"  {r['name']} ({r['type']}) score={r['score']}")

sub = store.get_subgraph("anthropic")
print(f"\nAnthropic neighbors ({len(sub['neighbors'])}):")
for n in sub["neighbors"][:5]:
    print(f"  {n['relation']} -> {n['name']}")

print("\nOK")
