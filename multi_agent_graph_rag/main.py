#!/usr/bin/env python3
import argparse
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

from src.config import validate_config
from src.graph_store import GraphStore
from src.token_tracker import TokenTracker
from src.tools import set_graph_store
from src.visualization import generate_all_charts, plot_knowledge_graph
import src.pipelines.standard_rag as rag
import src.pipelines.full_graph as full_graph
import src.pipelines.multi_agent as multi_agent

DATA_PATH = Path(__file__).parent / "data" / "knowledge_graph.json"

DEMO_QUESTIONS = [
    "Who founded Anthropic and what were they doing before? What models has Anthropic built?",
    "How is Microsoft connected to AI model development, and which AI products benefit from this?",
    "Which AI companies have received investment from Google, and what models do they develop?",
]


def print_answer(name: str, text: str):
    print(f"\n{'-' * 65}")
    print(f" {name}")
    print(f"{'-' * 65}")
    print(text)


def run_query(query: str, store: GraphStore, tracker: TokenTracker, pipelines: list[str]):
    print(f"\n[Query] {query}\n")

    if "rag" in pipelines:
        print("Running Standard RAG...")
        print_answer("Standard RAG", rag.run(query, store, tracker))

    if "full" in pipelines:
        print("\nRunning Full-Graph RAG...")
        print_answer("Full-Graph RAG", full_graph.run(query, store, tracker))

    if "multi" in pipelines:
        print("\nRunning Multi-Agent GraphRAG...")
        print_answer("Multi-Agent GraphRAG", multi_agent.run(query, store, tracker))


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Graph-RAG Explorer")
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--pipeline", "-p", choices=["rag", "full", "multi", "all"], default="all")
    parser.add_argument("--questions", action="store_true", help="Run all 3 demo questions")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--graph-only", action="store_true", help="Only render the knowledge graph")
    args = parser.parse_args()

    validate_config()

    print("\n" + "=" * 65)
    print("   MULTI-AGENT PARALLEL GRAPH-RAG EXPLORER")
    print("   Powered by LangGraph + Qdrant + GPT-4o-mini")
    print("=" * 65 + "\n")

    print("Loading knowledge graph...")
    store = GraphStore()
    store.load_and_ingest(DATA_PATH)
    set_graph_store(store)

    if args.graph_only:
        plot_knowledge_graph(store)
        return

    tracker = TokenTracker()
    pipelines = ["rag", "full", "multi"] if args.pipeline == "all" else [args.pipeline]
    questions = (
        [args.query] if args.query
        else DEMO_QUESTIONS if args.questions
        else [DEMO_QUESTIONS[0]]
    )

    for q in questions:
        run_query(q, store, tracker, pipelines)

    tracker.print_report()

    if not args.no_charts:
        generate_all_charts(tracker, store=store)
        print("Charts saved to: charts/")


if __name__ == "__main__":
    main()
