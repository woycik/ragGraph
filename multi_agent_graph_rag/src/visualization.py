from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from src.token_tracker import TokenTracker

OUTPUT_DIR = Path(__file__).parent.parent / "charts"

COLORS = {
    "1. Standard RAG":        "#4C9BE8",
    "2. Full-Graph RAG":      "#E8714C",
    "3. Multi-Agent GraphRAG": "#4CE87A",
}
FALLBACK = ["#4C9BE8", "#E8714C", "#4CE87A", "#B04CE8"]

THEME = {
    "figure.facecolor": "#1E1E2E",
    "axes.facecolor":   "#2A2A3E",
    "axes.labelcolor":  "#CDD6F4",
    "axes.titlecolor":  "#CDD6F4",
    "xtick.color":      "#CDD6F4",
    "ytick.color":      "#CDD6F4",
    "text.color":       "#CDD6F4",
    "grid.color":       "#3E3E5E",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "font.size":        11,
}

NODE_COLORS = {
    "Organization": "#4C9BE8",
    "Person":       "#E8714C",
    "Model":        "#4CE87A",
    "Product":      "#FFD700",
    "Technology":   "#B04CE8",
}


def _setup():
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update(THEME)
    OUTPUT_DIR.mkdir(exist_ok=True)


def _labels(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + h * 0.02,
            fmt.format(h), ha="center", va="bottom",
            color="#CDD6F4", fontsize=9, fontweight="bold",
        )


def _pipeline_colors(names):
    return [COLORS.get(n, FALLBACK[i % len(FALLBACK)]) for i, n in enumerate(names)]


def _short(names):
    return [n.split(". ", 1)[1] if ". " in n else n for n in names]


def plot_token_comparison(tracker: TokenTracker) -> Path:
    _setup()
    summary = tracker.summary_by_pipeline()
    if not summary:
        return None

    names = list(summary.keys())
    colors = _pipeline_colors(names)
    x, w = np.arange(len(names)), 0.28

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#1E1E2E")

    b1 = ax.bar(x - w, [summary[n]["input_tokens"]  for n in names], w, color=[c + "CC" for c in colors], label="Input")
    b2 = ax.bar(x,      [summary[n]["output_tokens"] for n in names], w, color=[c + "88" for c in colors], label="Output")
    b3 = ax.bar(x + w,  [summary[n]["total_tokens"]  for n in names], w, color=colors, label="Total")

    for bars in (b1, b2, b3):
        _labels(ax, bars)

    ax.set_title("Token Usage Comparison", fontsize=14, fontweight="bold", pad=16)
    ax.set_ylabel("Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(_short(names), fontsize=10)
    ax.legend(facecolor="#2A2A3E", edgecolor="#3E3E5E", labelcolor="#CDD6F4")
    fig.tight_layout()

    out = OUTPUT_DIR / "01_token_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def plot_cost_comparison(tracker: TokenTracker) -> Path:
    _setup()
    summary = tracker.summary_by_pipeline()
    if not summary:
        return None

    names = list(summary.keys())
    costs = [summary[n]["cost_usd"] * 1000 for n in names]
    colors = _pipeline_colors(names)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1E1E2E")

    bars = ax.bar(_short(names), costs, color=colors, width=0.5, edgecolor="#1E1E2E", linewidth=1.5)
    _labels(ax, bars, fmt="{:.3f}")

    ax.set_title("Estimated Cost per Query (milli-USD)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Cost (milli-USD)")
    fig.tight_layout()

    out = OUTPUT_DIR / "02_cost_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def plot_stacked_tokens(tracker: TokenTracker) -> Path:
    _setup()
    summary = tracker.summary_by_pipeline()
    if not summary:
        return None

    names = list(summary.keys())
    inputs  = [summary[n]["input_tokens"]  for n in names]
    outputs = [summary[n]["output_tokens"] for n in names]
    colors  = _pipeline_colors(names)
    x, w = np.arange(len(names)), 0.45

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1E1E2E")

    ax.bar(x, inputs,  w, color=colors, label="Input")
    ax.bar(x, outputs, w, bottom=inputs, color=["#FFFFFF33"] * len(names), hatch="//", edgecolor="#FFFFFF44", label="Output")

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        total = inp + out
        ax.text(x[i], total + total * 0.01, f"{total:,}", ha="center", va="bottom",
                color="#CDD6F4", fontsize=9, fontweight="bold")

    ax.set_title("Input vs Output Tokens per Pipeline", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(_short(names), fontsize=10)
    ax.legend(handles=[
        mpatches.Patch(color="#888888", label="Input tokens"),
        mpatches.Patch(color="#FFFFFF55", hatch="//", label="Output tokens"),
    ], facecolor="#2A2A3E", edgecolor="#3E3E5E", labelcolor="#CDD6F4")
    fig.tight_layout()

    out = OUTPUT_DIR / "03_stacked_tokens.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def plot_api_calls(tracker: TokenTracker) -> Path:
    _setup()
    summary = tracker.summary_by_pipeline()
    if not summary:
        return None

    names = list(summary.keys())
    calls = [summary[n]["api_calls"] for n in names]
    colors = _pipeline_colors(names)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1E1E2E")

    bars = ax.bar(_short(names), calls, color=colors, width=0.4, edgecolor="#1E1E2E")
    ax.set_yticks(range(0, max(calls) + 2))
    _labels(ax, bars)

    ax.set_title("Number of API Calls per Pipeline", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("API calls")
    fig.tight_layout()

    out = OUTPUT_DIR / "04_api_calls.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def plot_knowledge_graph(store, max_nodes: int = 30) -> Path:
    plt.rcParams.update(THEME)
    OUTPUT_DIR.mkdir(exist_ok=True)

    G = nx.DiGraph()
    visible = {n["id"] for n in list(store.nodes.values())[:max_nodes]}

    for nid in visible:
        node = store.nodes[nid]
        G.add_node(nid, label=node["name"], ntype=node["type"])

    for src, edges in store.adjacency.items():
        if src not in visible:
            continue
        for e in edges:
            if not e["relation"].startswith("REVERSE_") and e["target"] in visible:
                G.add_edge(src, e["target"])

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor("#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    ax.axis("off")

    pos = nx.spring_layout(G, k=2.2, seed=42)
    node_colors = [NODE_COLORS.get(G.nodes[n].get("ntype", ""), "#888888") for n in G.nodes()]
    node_sizes  = [800 + G.degree(n) * 120 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#FFFFFF33", arrows=True,
                           arrowsize=14, width=0.8, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: G.nodes[n]["label"] for n in G.nodes()},
                            font_size=7.5, font_color="#CDD6F4", font_weight="bold")

    ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in NODE_COLORS.items()],
              loc="upper left", facecolor="#2A2A3E", edgecolor="#3E3E5E",
              labelcolor="#CDD6F4", fontsize=9)
    ax.set_title("Knowledge Graph — AI Ecosystem", fontsize=15, fontweight="bold", color="#CDD6F4", pad=14)
    fig.tight_layout()

    out = OUTPUT_DIR / "05_knowledge_graph.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1E1E2E")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def plot_summary_dashboard(tracker: TokenTracker) -> Path:
    _setup()
    summary = tracker.summary_by_pipeline()
    if not summary:
        return None

    names  = list(summary.keys())
    short  = _short(names)
    colors = _pipeline_colors(names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1E1E2E")
    fig.suptitle("RAG Pipeline Comparison", fontsize=15, fontweight="bold", color="#CDD6F4", y=1.02)

    panels = [
        ("Total Tokens",      [summary[n]["total_tokens"] for n in names], "tokens",      "{:.0f}"),
        ("Cost (milli-USD)",  [summary[n]["cost_usd"] * 1000 for n in names], "milli-USD", "{:.3f}"),
        ("API Calls",         [summary[n]["api_calls"] for n in names], "calls",       "{:.0f}"),
    ]

    for ax, (title, values, ylabel, fmt) in zip(axes, panels):
        ax.set_facecolor("#2A2A3E")
        bars = ax.bar(short, values, color=colors, width=0.5)
        _labels(ax, bars, fmt=fmt)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        if ylabel == "calls":
            ax.set_yticks(range(0, int(max(values)) + 2))

    fig.tight_layout()

    out = OUTPUT_DIR / "06_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1E1E2E")
    plt.close(fig)
    print(f"[Viz] {out.name}")
    return out


def generate_all_charts(tracker: TokenTracker, store=None) -> list[Path]:
    print("\n[Viz] Generating charts...")
    paths = [
        plot_token_comparison(tracker),
        plot_cost_comparison(tracker),
        plot_stacked_tokens(tracker),
        plot_api_calls(tracker),
    ]
    if store is not None:
        paths.append(plot_knowledge_graph(store))
    paths.append(plot_summary_dashboard(tracker))
    paths = [p for p in paths if p]
    print(f"[Viz] {len(paths)} charts saved to {OUTPUT_DIR}\n")
    return paths
