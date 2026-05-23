import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.token_tracker import TokenTracker, ModelCall
from src.visualization import generate_all_charts, plot_knowledge_graph
from src.graph_store import GraphStore

tracker = TokenTracker()
tracker.calls = [
    ModelCall("1. Standard RAG",        "synthesis",   "gpt-4o-mini", 480,  95),
    ModelCall("2. Full-Graph RAG",      "synthesis",   "gpt-4o-mini", 7850, 140),
    ModelCall("3. Multi-Agent GraphRAG","coordinator", "gpt-4o-mini", 210,  85),
    ModelCall("3. Multi-Agent GraphRAG","synthesizer", "gpt-4o-mini", 980,  130),
]

tracker.print_report()

store = GraphStore()
store.load_and_ingest("data/knowledge_graph.json")

paths = generate_all_charts(tracker, store=store)
print(f"Generated {len(paths)} charts")
for p in paths:
    print(f"  {p.name}")

print("OK")
