"""Generate visualization charts for benchmark findings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_retrieval_performance() -> None:
    """Plot grouped bar chart of Top-1 Acc, Top-5 Hit, and Pasal-Recall@5."""
    methods = [
        "dense-baseline",
        "hybrid-bm25",
        "tree-llm",
        "tree-jev",
        "tree-d3-base",
        "tree-d3 (tuned)",
    ]
    top1_acc = [60.00, 60.00, 66.67, 0.00, 60.00, 60.00]
    top5_hit = [60.00, 80.00, 66.67, 20.00, 80.00, 80.00]
    pasal_rec = [60.00, 60.00, 66.67, 40.00, 80.00, 80.00]

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 5.5), dpi=150)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    rects1 = ax.bar(
        x - width, top1_acc, width, label="Top-1 Accuracy (Hit@1)", color="#2563eb", zorder=3
    )
    rects2 = ax.bar(x, top5_hit, width, label="Top-5 Hit (Hit@5)", color="#10b981", zorder=3)
    rects3 = ax.bar(x + width, pasal_rec, width, label="Pasal-Recall@5", color="#f59e0b", zorder=3)

    ax.set_ylabel("Accuracy / Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Information Retrieval Performance by Architecture: Indonesian Legal RAG",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper left", fontsize=10)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "retrieval_performance.svg", format="svg")
    plt.savefig(ASSETS_DIR / "retrieval_performance.png", format="png", dpi=200)
    plt.close()
    print("Saved retrieval_performance.svg and .png")


def plot_latency_accuracy_tradeoff() -> None:
    """Plot latency vs Top-5 Hit accuracy frontier."""
    data = [
        {"name": "dense-baseline", "lat": 11.88, "hit": 60.00, "faith": 0.60, "type": "Vector"},
        {"name": "hybrid-bm25", "lat": 13.35, "hit": 80.00, "faith": 0.60, "type": "Vector"},
        {"name": "tree-llm", "lat": 23.51, "hit": 66.67, "faith": 0.50, "type": "Tree (LLM)"},
        {"name": "tree-jev", "lat": 10.65, "hit": 20.00, "faith": 0.30, "type": "Tree (Cloud)"},
        {
            "name": "tree-d3-base",
            "lat": 73.43,
            "hit": 80.00,
            "faith": 0.50,
            "type": "Tree (Local CPU)",
        },
        {"name": "tree-d3", "lat": 69.54, "hit": 80.00, "faith": 0.60, "type": "Tree (Local CPU)"},
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

    color_map = {
        "Vector": "#3b82f6",
        "Tree (LLM)": "#8b5cf6",
        "Tree (Cloud)": "#ef4444",
        "Tree (Local CPU)": "#10b981",
    }

    for d in data:
        c = color_map[d["type"]]
        ax.scatter(
            d["lat"],
            d["hit"],
            s=d["faith"] * 500,
            color=c,
            alpha=0.85,
            edgecolors="#1e293b",
            linewidth=1.5,
            zorder=3,
        )
        offset_y = 2.5 if d["name"] != "tree-d3-base" else -4.0
        offset_x = 0.5
        ax.annotate(
            f"{d['name']}\n(Faith: {d['faith']:.2f})",
            (d["lat"] + offset_x, d["hit"] + offset_y),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("p50 Query Latency (seconds, log-scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-5 Retrieval Hit Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Retrieval Accuracy vs Latency Trade-off (Bubble Size = Faithfulness)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xscale("log")
    ax.set_ylim(10, 95)
    ax.set_xlim(8, 90)

    # Note about CPU vs GPU
    ax.text(
        0.03,
        0.05,
        "* Note: tree-d3 latency is on CPU with reference DeltaNet loops;\n  GPU Triton linear attention drops decision readout to <1s.",
        transform=ax.transAxes,
        fontsize=8.5,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "latency_accuracy_tradeoff.svg", format="svg")
    plt.savefig(ASSETS_DIR / "latency_accuracy_tradeoff.png", format="png", dpi=200)
    plt.close()
    print("Saved latency_accuracy_tradeoff.svg and .png")


def plot_d3_decision_tuning() -> None:
    """Plot D3 foundation vs fine-tuned comparative metrics."""
    categories = [
        "Supervised Decision\nRouting Acc (%)",
        "Pasal Identification\nRecall@5 (%)",
        "End-to-End Answer\nFaithfulness (x100)",
        "Top-5 Retrieval\nHit Rate (%)",
    ]
    base_scores = [17.50, 80.00, 50.00, 80.00]
    tuned_scores = [50.00, 80.00, 60.00, 80.00]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    rects1 = ax.bar(
        x - width / 2,
        base_scores,
        width,
        label="Foundation (vllm-sr/d3-edge 0.6B)",
        color="#94a3b8",
        zorder=3,
    )
    rects2 = ax.bar(
        x + width / 2,
        tuned_scores,
        width,
        label="Fine-Tuned (ersaasyhar/d3-edge-indo-legal)",
        color="#6366f1",
        zorder=3,
    )

    ax.set_ylabel("Score / Percentage", fontsize=11, fontweight="bold")
    ax.set_title(
        "System 1 Decision Fine-Tuning Impact: Foundation vs Domain-Adapted",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper left", fontsize=10)

    for r in rects1:
        h = r.get_height()
        ax.annotate(
            f"{h:.1f}",
            (r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for r in rects2:
        h = r.get_height()
        ax.annotate(
            f"{h:.1f}",
            (r.get_x() + r.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#4338ca",
        )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "d3_decision_tuning.svg", format="svg")
    plt.savefig(ASSETS_DIR / "d3_decision_tuning.png", format="png", dpi=200)
    plt.close()
    print("Saved d3_decision_tuning.svg and .png")


def main() -> None:
    print(f"Generating benchmark visualization assets in {ASSETS_DIR}...")
    plot_retrieval_performance()
    plot_latency_accuracy_tradeoff()
    plot_d3_decision_tuning()
    print("All visualizations generated successfully.")


if __name__ == "__main__":
    main()
