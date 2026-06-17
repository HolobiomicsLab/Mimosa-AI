"""Generate visualizations for the evolution analysis report.

Outputs all PNGs into /tmp/evo_report/figs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

OUT = Path("/tmp/evo_report")
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

GOAL_ORDER = ["clintox", "shap_diffusion", "bulk_modulus", "elk_homerange", "dkpes"]
GOAL_COLORS = {
    "clintox": "#1f77b4",
    "shap_diffusion": "#ff7f0e",
    "bulk_modulus": "#2ca02c",
    "elk_homerange": "#d62728",
    "dkpes": "#9467bd",
}
KIND_MARKERS = {"seed": "s", "mutation": "o", "crossover": "D"}


def load_data():
    wf = pd.read_csv(OUT / "workflows.csv")
    wf = wf[wf["iteration"].notna()].copy()
    wf["iteration"] = wf["iteration"].astype(int)
    for col in ("overall_score", "overall_score_uncapped", "qd_score", "novelty_score",
                "cost_usd", "cumulative_cost_usd", "wall_time_s",
                "stagnation", "success_rate", "effective_boldness",
                "parent_score", "agent_budget", "genotype_lines", "gradient_chars"):
        if col in wf.columns:
            wf[col] = pd.to_numeric(wf[col], errors="coerce")
    wf = wf.sort_values(["goal_label", "iteration"]).reset_index(drop=True)

    qd = pd.read_csv(OUT / "qd_archive.csv")
    var = pd.read_csv(OUT / "variation_log.csv")
    return wf, qd, var


# ---------- 1. Reward trajectory per goal ----------
def fig_reward_trajectory(wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for label in GOAL_ORDER:
        g = wf[wf["goal_label"] == label].sort_values("iteration")
        if g.empty:
            continue
        color = GOAL_COLORS[label]
        ax.plot(g["iteration"], g["overall_score"], "-", color=color, alpha=0.5, lw=1.2)
        # marker by evolution kind
        for kind, marker in KIND_MARKERS.items():
            sub = g[g["evolution_kind"] == kind]
            if not sub.empty:
                ax.scatter(sub["iteration"], sub["overall_score"], marker=marker,
                           color=color, s=70, edgecolor="black", linewidth=0.6,
                           label=f"{label} ({kind})" if kind == "seed" else None)
        # best-so-far
        best = g["overall_score"].cummax()
        ax.plot(g["iteration"], best, ":", color=color, lw=2.0, alpha=0.9)

    # legend for kinds
    legend_kinds = [plt.Line2D([0], [0], marker=m, color="grey", linestyle="",
                               markersize=9, label=k) for k, m in KIND_MARKERS.items()]
    legend_goals = [plt.Line2D([0], [0], color=GOAL_COLORS[l], lw=3, label=l) for l in GOAL_ORDER]
    leg1 = ax.legend(handles=legend_goals, loc="lower right", title="Goal", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=legend_kinds, loc="upper left", title="Evolution kind", fontsize=9)

    ax.set_xlabel("iteration")
    ax.set_ylabel("overall_score (rubric, capped)")
    ax.set_title("Reward trajectory across 5 evolution lineages\n(dotted line = best-so-far)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "01_reward_trajectory.png", dpi=130)
    plt.close(fig)


# ---------- 2. Per-goal small-multiples score vs iter ----------
def fig_per_goal_small_multiples(wf: pd.DataFrame) -> None:
    goals = [g for g in GOAL_ORDER if (wf["goal_label"] == g).any()]
    n = len(goals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, goals):
        g = wf[wf["goal_label"] == label].sort_values("iteration")
        color = GOAL_COLORS[label]
        ax.plot(g["iteration"], g["overall_score"], "-o", color=color, label="capped score", lw=1.2, ms=5)
        if g["overall_score_uncapped"].notna().any():
            ax.plot(g["iteration"], g["overall_score_uncapped"], "--^", color="grey",
                    label="uncapped", lw=1, ms=4, alpha=0.7)
        best = g["overall_score"].cummax()
        ax.plot(g["iteration"], best, ":", color="black", lw=1.8, alpha=0.8, label="best so far")
        ax.set_title(f"{label}\n(n={len(g)} iters, best={g['overall_score'].max():.3f})")
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("overall_score")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-goal evolution trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "02_per_goal_trajectories.png", dpi=130)
    plt.close(fig)


# ---------- 3. QD score vs novelty scatter ----------
def fig_qd_novelty(wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in GOAL_ORDER:
        g = wf[wf["goal_label"] == label]
        if g.empty:
            continue
        ax.scatter(g["novelty_score"], g["qd_score"], color=GOAL_COLORS[label],
                   s=80, alpha=0.75, edgecolor="black", linewidth=0.5, label=label)
    ax.set_xlabel("novelty_score (k-NN distance, behaviour-fingerprint)")
    ax.set_ylabel("qd_score = 0.6·quality + 0.4·novelty")
    ax.set_title("QD landscape: quality vs novelty per workflow")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "03_qd_novelty_scatter.png", dpi=130)
    plt.close(fig)


# ---------- 4. Variation engine schedule (Rechenberg 1/5) ----------
def fig_variation_schedule(wf: pd.DataFrame) -> None:
    goals = [g for g in GOAL_ORDER if (wf["goal_label"] == g).any()]
    n = len(goals)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, goals):
        g = wf[wf["goal_label"] == label].sort_values("iteration")
        color = GOAL_COLORS[label]
        x = g["iteration"]
        ax.plot(x, g["stagnation"], "-o", color="firebrick", label="stagnation", ms=4)
        ax.plot(x, g["success_rate"], "-o", color="green", label="success_rate", ms=4)
        ax.plot(x, g["effective_boldness"], "-D", color=color, label="effective_boldness", ms=5, lw=2)
        ax.set_title(f"Variation engine schedule — {label}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.set_ylabel("[0..1]")
        # annotate scope band with text
        for _, row in g.iterrows():
            if isinstance(row.get("scope_band"), str):
                ax.annotate(row["scope_band"][:14], (row["iteration"], -0.02),
                            fontsize=6, ha="center", color="dimgrey", rotation=45)
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Rechenberg 1/5 schedule per goal: how 'boldness' tracks stagnation vs. success", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "04_variation_schedule.png", dpi=130)
    plt.close(fig)


# ---------- 5. Cost (cumulative) per goal ----------
def fig_cost_curves(wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in GOAL_ORDER:
        g = wf[wf["goal_label"] == label].sort_values("iteration")
        if g.empty:
            continue
        # cumulative cost per goal (sum cost_usd per iteration within goal — cumulative_cost is global)
        cum = g["cost_usd"].cumsum()
        ax.plot(g["iteration"], cum, "-o", color=GOAL_COLORS[label], label=f"{label}  (${cum.iloc[-1]:.2f})", ms=5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cumulative cost (USD, per-goal)")
    ax.set_title("Per-goal cumulative LLM cost across evolution iterations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "05_cost_curves.png", dpi=130)
    plt.close(fig)


# ---------- 6. Agent budget + genotype size growth ----------
def fig_agent_and_geno(wf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in GOAL_ORDER:
        g = wf[wf["goal_label"] == label].sort_values("iteration")
        if g.empty:
            continue
        color = GOAL_COLORS[label]
        axes[0].plot(g["iteration"], g["agent_budget"], "-o", color=color, label=label, ms=5)
        axes[1].plot(g["iteration"], g["genotype_lines"], "-o", color=color, label=label, ms=5)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("agent_budget")
    axes[0].set_title("Agent budget (max # agents the mutator may use)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("workflow_genotype line count")
    axes[1].set_title("Workflow code size (lines in workflow_genotype_*.py)")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "06_agent_and_geno.png", dpi=130)
    plt.close(fig)


# ---------- 7. Behaviour descriptor (qd_descriptor) heatmap per goal ----------
def fig_qd_descriptor_heatmap(wf: pd.DataFrame) -> None:
    goals = [g for g in GOAL_ORDER if (wf["goal_label"] == g).any()]
    n = len(goals)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    src_labels = ["literature", "goal", "narration", "math", "cs_practice", "statistical"]
    for ax, label in zip(axes, goals):
        g = wf[wf["goal_label"] == label].sort_values("iteration").reset_index(drop=True)
        mat = []
        for d in g["qd_descriptor"]:
            try:
                v = json.loads(d)
                if isinstance(v, list) and len(v) == 6:
                    mat.append(v)
                else:
                    mat.append([0] * 6)
            except Exception:
                mat.append([0] * 6)
        mat = np.array(mat, dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(6))
        ax.set_xticklabels(src_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{label}\n(iters {g['iteration'].min()}-{g['iteration'].max()})", fontsize=10)
        ax.set_ylabel("iteration" if ax is axes[0] else "")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="per-source pass rate")
    fig.suptitle("Behaviour descriptor (6-dim failure fingerprint) over iterations", fontsize=12)
    fig.savefig(FIGS / "07_qd_descriptor_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------- 8. Lineage / family tree per goal ----------
def fig_lineage_tree(wf: pd.DataFrame) -> None:
    goals = [g for g in GOAL_ORDER if (wf["goal_label"] == g).any()]
    n = len(goals)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, label in zip(axes, goals):
        g = wf[wf["goal_label"] == label].copy()
        G = nx.DiGraph()
        for _, row in g.iterrows():
            G.add_node(row["uuid"],
                       iteration=int(row["iteration"]),
                       score=float(row["overall_score"]) if pd.notna(row["overall_score"]) else 0.0,
                       kind=row["evolution_kind"])
            parents = str(row["parent_uuids"]).split(";") if isinstance(row["parent_uuids"], str) else []
            for p in parents:
                p = p.strip()
                if p:
                    G.add_edge(p, row["uuid"])
        # only keep edges where both endpoints exist
        edges = [(u, v) for u, v in G.edges if u in G.nodes and v in G.nodes]
        # layout: x = iteration (or topo gen), y = score
        pos = {}
        for node, d in G.nodes(data=True):
            if "iteration" in d:
                pos[node] = (d["iteration"], d["score"])
        # for orphan parents (no row), put at -1
        for node in list(G.nodes):
            if node not in pos:
                pos[node] = (-1, 0.0)

        node_colors = []
        node_sizes = []
        for node, d in G.nodes(data=True):
            s = d.get("score", 0)
            node_colors.append(s)
            node_sizes.append(120 + 280 * s)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, alpha=0.5, edge_color="grey",
                               arrowsize=10, width=1.0)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap="viridis",
                                       vmin=0, vmax=1, node_size=node_sizes,
                                       edgecolors="black", linewidths=0.6)
        for node, d in G.nodes(data=True):
            if d.get("kind") == "crossover":
                ax.scatter(*pos[node], marker="D", s=node_sizes[list(G.nodes).index(node)] * 1.2,
                           facecolors="none", edgecolors="red", linewidths=1.8)
        ax.set_title(f"{label}: lineage  (n={len(g)})", fontsize=11)
        ax.set_xlabel("iteration"); ax.set_ylabel("overall_score")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        # colorbar
        fig.colorbar(nodes, ax=ax, fraction=0.04, pad=0.02, label="score")
    for ax in axes[len(goals):]:
        ax.set_visible(False)
    fig.suptitle("Evolution lineages: parent→child edges, node color/size = overall_score (red diamond = crossover)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "08_lineage_trees.png", dpi=130)
    plt.close(fig)


# ---------- 9. Evolution kind distribution ----------
def fig_kind_distribution(wf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot = wf.pivot_table(index="goal_label", columns="evolution_kind",
                           values="uuid", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(GOAL_ORDER).dropna(how="all")
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color={"seed": "#bbbbbb", "mutation": "#1f77b4", "crossover": "#d62728"})
    for c in ax.containers:
        ax.bar_label(c, label_type="center", fontsize=8, color="white")
    ax.set_ylabel("# workflows")
    ax.set_title("Evolution operator usage per goal")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="kind")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "09_kind_distribution.png", dpi=130)
    plt.close(fig)


# ---------- 10. QD archive admission ----------
def fig_qd_admissions(qd: pd.DataFrame, wf: pd.DataFrame) -> None:
    # qd lacks goal label by default — join via uuid
    if qd.empty:
        return
    uuid_to_label = dict(zip(wf["uuid"], wf["goal_label"]))
    qd = qd.copy()
    qd["goal_label"] = qd["uuid"].map(uuid_to_label).fillna("unknown")
    qd = qd[qd["goal_label"] != "unknown"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for label in GOAL_ORDER:
        g = qd[qd["goal_label"] == label].sort_values("iteration")
        if g.empty:
            continue
        ax.scatter(g["iteration"], g["qd_score"], color=GOAL_COLORS[label],
                   s=90, edgecolor="black", linewidth=0.5, alpha=0.85, label=label)
        # mark evictions
        if "evicted_uuid" in g.columns:
            ev = g[g["evicted_uuid"].notna() & (g["evicted_uuid"] != "")]
            ax.scatter(ev["iteration"], ev["qd_score"], marker="x", s=160,
                       color="black", linewidth=1.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("qd_score (admission)")
    ax.set_title("QD archive admissions per goal  (× = also caused eviction)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "10_qd_admissions.png", dpi=130)
    plt.close(fig)


# ---------- 11. Empty / failed iterations timeline ----------
def fig_failure_timeline(wf_full: pd.DataFrame) -> None:
    # wf_full includes empties (without iteration). order by uuid (timestamp prefix)
    wf_full = wf_full.copy()
    wf_full["timestamp"] = wf_full["uuid"].str.slice(0, 15)
    wf_full = wf_full.sort_values("timestamp").reset_index(drop=True)
    wf_full["idx"] = range(len(wf_full))
    fig, ax = plt.subplots(figsize=(13, 3.5))
    colors = []
    for _, row in wf_full.iterrows():
        if row["goal_id"] == "unknown":
            colors.append("#cccccc")  # empty
        else:
            colors.append(GOAL_COLORS.get(row["goal_label"], "#999"))
    ax.bar(wf_full["idx"], [1] * len(wf_full), color=colors, edgecolor="none", width=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Wall-clock timeline of all 97 attempted iterations "
                 f"({(wf_full['goal_id']=='unknown').sum()} empty/failed, "
                 f"{(wf_full['goal_id']!='unknown').sum()} produced artifacts)")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=GOAL_COLORS[l]) for l in GOAL_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#cccccc"))
    ax.legend(handles, GOAL_ORDER + ["EMPTY (no artifacts)"], loc="upper right",
              ncol=6, fontsize=8, bbox_to_anchor=(1, 1.35))
    fig.tight_layout()
    fig.savefig(FIGS / "11_failure_timeline.png", dpi=130)
    plt.close(fig)


# ---------- 12. Summary stats CSV ----------
def write_summary(wf: pd.DataFrame) -> None:
    rows = []
    for label in GOAL_ORDER:
        g = wf[wf["goal_label"] == label]
        if g.empty:
            continue
        first = g.iloc[0]
        best = g.loc[g["overall_score"].idxmax()]
        last_iter = g["iteration"].max()
        rows.append({
            "goal": label,
            "n_iters": len(g),
            "iter_range": f"{g['iteration'].min()}-{last_iter}",
            "seed_score": float(first["overall_score"]),
            "best_score": float(best["overall_score"]),
            "best_iter": int(best["iteration"]),
            "delta_seed_to_best": float(best["overall_score"] - first["overall_score"]),
            "final_score": float(g.iloc[-1]["overall_score"]),
            "n_crossovers": int((g["evolution_kind"] == "crossover").sum()),
            "n_mutations": int((g["evolution_kind"] == "mutation").sum()),
            "total_cost_usd": float(g["cost_usd"].sum()),
            "total_wall_time_min": float(g["wall_time_s"].sum() / 60),
            "max_agent_budget": int(g["agent_budget"].max()) if g["agent_budget"].notna().any() else None,
            "max_geno_lines": int(g["genotype_lines"].max()) if g["genotype_lines"].notna().any() else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.to_string(index=False))


def main() -> None:
    wf_full = pd.read_csv(OUT / "workflows.csv")
    wf, qd, var = load_data()

    fig_reward_trajectory(wf)
    fig_per_goal_small_multiples(wf)
    fig_qd_novelty(wf)
    fig_variation_schedule(wf)
    fig_cost_curves(wf)
    fig_agent_and_geno(wf)
    fig_qd_descriptor_heatmap(wf)
    fig_lineage_tree(wf)
    fig_kind_distribution(wf)
    fig_qd_admissions(qd, wf)
    fig_failure_timeline(wf_full)
    write_summary(wf)
    print(f"\nwrote {len(list(FIGS.glob('*.png')))} figures to {FIGS}")


if __name__ == "__main__":
    main()
