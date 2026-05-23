import json
import sys
import warnings
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


def mock_response(text, prompt_tokens=100, completion_tokens=50):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = text
    r.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return r


COORDINATOR_JSON = json.dumps({
    "sub_queries": ["Who founded Anthropic?", "What models has Anthropic built?", "Who invested in Anthropic?"],
    "reasoning": "Split by people, products, and funding",
})
SYNTHESIS_TEXT = (
    "Anthropic was founded by Dario and Daniela Amodei. "
    "They developed Claude 3 Opus and Sonnet. Investors include Google and Amazon."
)


def test_all_pipelines():
    from src.graph_store import GraphStore
    from src.token_tracker import TokenTracker
    from src.tools import set_graph_store

    print("Loading graph...")
    store = GraphStore()
    store.load_and_ingest("data/knowledge_graph.json")
    set_graph_store(store)

    tracker = TokenTracker()
    query = "Who founded Anthropic and what models have they built?"

    print("\n[1] Standard RAG")
    with patch("openai.OpenAI") as Mock:
        Mock.return_value.chat.completions.create.return_value = mock_response(
            "Anthropic was founded by former OpenAI researchers.", 300, 40)
        import src.pipelines.standard_rag as rag
        answer = rag.run(query, store, tracker)
        assert len(answer) > 5
        print(f"   OK: {answer[:70]}...")

    print("\n[2] Full-Graph RAG")
    with patch("openai.OpenAI") as Mock:
        Mock.return_value.chat.completions.create.return_value = mock_response(
            "Anthropic was founded by Dario Amodei.", 5000, 50)
        import src.pipelines.full_graph as fg
        answer = fg.run(query, store, tracker)
        assert len(answer) > 10
        print(f"   OK: {answer[:70]}...")

    print("\n[3] Multi-Agent GraphRAG")
    calls = [0]

    def side_effect(*a, **kw):
        n = calls[0]; calls[0] += 1
        return mock_response(COORDINATOR_JSON, 150, 80) if n == 0 else mock_response(SYNTHESIS_TEXT, 800, 120)

    with patch("openai.OpenAI") as Mock:
        Mock.return_value.chat.completions.create.side_effect = side_effect
        import src.pipelines.multi_agent as ma
        answer = ma.run(query, store, tracker)
        assert "Anthropic" in answer
        assert calls[0] == 2
        print(f"   OK: {answer[:70]}...")
        print(f"   API calls: {calls[0]} (coordinator + synthesizer)")

    print("\n[4] Token report:")
    tracker.print_report()

    summary = tracker.summary_by_pipeline()
    assert len(summary) == 3
    assert summary["2. Full-Graph RAG"]["input_tokens"] > summary["1. Standard RAG"]["input_tokens"]
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_all_pipelines()
