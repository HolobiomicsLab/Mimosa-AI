"""Deep-investigation visualizations.

Outputs to /tmp/evo_report/figs/ with d_ prefix.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("/tmp/evo_report")
FIGS = OUT / "figs"

GOAL_ORDER = ["clintox", "shap_diffusion", "bulk_modulus", "elk_homerange", "dkpes"]
GOAL_COLORS = {
    "clintox": "#1f77b4",
    "shap_diffusion": "#ff7f0e",
    "bulk_modulus": "#2ca02c",
    "elk_homerange": "#d62728",
    "dkpes": "#9467bd",
}
NOISE_FLOOR = 0.08  # ±0.08 from agent 10's analysis (temperature=1.0 LLM judge)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    wf = pd.read_csv(OUT / "workflows.csv")
    wf = wf[wf["iteration"].notna()].copy()
    wf["iteration"] = wf["iteration"].astype(int)
    for col in ("overall_score", "qd_score", "novelty_score", "cost_usd",
                "stagnation", "success_rate", "effective_boldness"):
        if col in wf.columns:
            wf[col] = pd.to_numeric(wf[col], errors="coerce")
    ev = pd.read_csv(OUT / "eval_summary.csv")
    return wf, ev


# ---------- D1. Crossover outcomes vs parent scores ----------
def fig_crossover_outcomes(wf: pd.DataFrame) -> None:
    cx = wf[wf["evolution_kind"] == "crossover"].copy()
    uuid_to_score = dict(zip(wf["uuid"], wf["overall_score"]))
    parent_best, parent_worst, child = [], [], []
    labels = []
    for _, r in cx.iterrows():
        ps = str(r["parent_uuids"]).split(";") if isinstance(r["parent_uuids"], str) else []
        scores = [uuid_to_score.get(p) for p in ps]
        scores = [s for s in scores if s is not None and not pd.isna(s)]
        if len(scores) < 2:
            continue
        parent_best.append(max(scores))
        parent_worst.append(min(scores))
        child.append(r["overall_score"])
        labels.append(r["goal_label"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # left: child vs best parent
    for label in GOAL_ORDER:
        idx = [i for i, l in enumerate(labels) if l == label]
        if not idx:
            continue
        x = [parent_best[i] for i in idx]
        y = [child[i] for i in idx]
        axes[0].scatter(x, y, color=GOAL_COLORS[label], s=85,
                        edgecolor="black", linewidth=0.6, alpha=0.8, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].fill_between([0, 1], [-NOISE_FLOOR, 1 - NOISE_FLOOR], [NOISE_FLOOR, 1 + NOISE_FLOOR],
                         color="grey", alpha=0.15, label=f"±{NOISE_FLOOR} noise band")
    axes[0].set_xlabel("BEST parent score")
    axes[0].set_ylabel("child (crossover) score")
    axes[0].set_title("Crossover children vs their BEST parent\n(grey band = noise floor, below diagonal = regression)")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc="lower right", fontsize=8)

    # right: child vs worst parent
    for label in GOAL_ORDER:
        idx = [i for i, l in enumerate(labels) if l == label]
        if not idx:
            continue
        x = [parent_worst[i] for i in idx]
        y = [child[i] for i in idx]
        axes[1].scatter(x, y, color=GOAL_COLORS[label], s=85,
                        edgecolor="black", linewidth=0.6, alpha=0.8, label=label)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[1].set_xlabel("WORST parent score")
    axes[1].set_ylabel("child (crossover) score")
    axes[1].set_title("Crossover children vs their WORST parent")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # annotate stats
    parent_best = np.array(parent_best); parent_worst = np.array(parent_worst); child = np.array(child)
    pct_beat_best = (child > parent_best).mean() * 100
    pct_beat_worst = (child > parent_worst).mean() * 100
    pct_below_both = (child < parent_worst).mean() * 100
    fig.suptitle(
        f"Crossover outcomes (n={len(child)})  —  beat best: {pct_beat_best:.0f}%   "
        f"beat worst-only: {pct_beat_worst - pct_beat_best:.0f}%   below both: {pct_below_both:.0f}%",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "d1_crossover_outcomes.png", dpi=130)
    plt.close(fig)


# ---------- D2. Reward noise floor vs observed deltas ----------
def fig_noise_floor(wf: pd.DataFrame) -> None:
    # for each lineage, compute |child_score - parent_score| using the single-parent for mutations,
    # and |child_score - mean(parent_score)| for crossovers
    uuid_to_score = dict(zip(wf["uuid"], wf["overall_score"]))
    deltas = []
    abs_deltas = []
    labels_d = []
    for _, r in wf.iterrows():
        if r["evolution_kind"] == "seed":
            continue
        ps = str(r["parent_uuids"]).split(";") if isinstance(r["parent_uuids"], str) else []
        scores = [uuid_to_score.get(p) for p in ps]
        scores = [s for s in scores if s is not None and not pd.isna(s)]
        if not scores:
            continue
        ps_score = np.mean(scores)
        d = r["overall_score"] - ps_score
        deltas.append(d)
        abs_deltas.append(abs(d))
        labels_d.append(r["goal_label"])
    deltas = np.array(deltas); abs_deltas = np.array(abs_deltas)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(deltas, bins=30, color="#888", edgecolor="black")
    axes[0].axvspan(-NOISE_FLOOR, NOISE_FLOOR, color="red", alpha=0.18, label=f"noise floor ±{NOISE_FLOOR}")
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("child_score − parent_score (signed)")
    axes[0].set_ylabel("# offspring")
    axes[0].set_title("Distribution of parent→child score deltas\n(grey = below noise floor → indistinguishable from re-eval variance)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # right: per goal
    in_noise = (abs_deltas < NOISE_FLOOR).sum() / len(abs_deltas) * 100
    bar_data = []
    for label in GOAL_ORDER:
        idx = [i for i, l in enumerate(labels_d) if l == label]
        if not idx:
            continue
        within = sum(1 for i in idx if abs_deltas[i] < NOISE_FLOOR)
        bar_data.append((label, within, len(idx) - within))
    if bar_data:
        ls = [b[0] for b in bar_data]
        in_n = [b[1] for b in bar_data]
        out_n = [b[2] for b in bar_data]
        x = np.arange(len(ls))
        axes[1].bar(x, in_n, color="#cccccc", label="|Δ| < noise floor (indistinguishable)")
        axes[1].bar(x, out_n, bottom=in_n, color=[GOAL_COLORS[l] for l in ls],
                    label="|Δ| > noise floor (real signal)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(ls, rotation=20)
        axes[1].set_ylabel("# offspring")
        axes[1].set_title(f"Offspring with detectable improvement vs noise per goal\n(overall: {in_noise:.0f}% of offspring within noise band)")
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "d2_noise_floor.png", dpi=130)
    plt.close(fig)


# ---------- D3. Score components stacked ----------
def fig_score_components(wf: pd.DataFrame, ev: pd.DataFrame) -> None:
    merged = wf.merge(ev[["uuid", "base_mean", "info_bonus", "hard_fail_capped",
                           "n_error", "n_scored", "uncapped"]],
                       on="uuid", how="left")
    goals = [g for g in GOAL_ORDER if (merged["goal_label"] == g).any()]
    n = len(goals)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, goals):
        g = merged[merged["goal_label"] == label].sort_values("iteration")
        x = g["iteration"]
        # stacked: base_mean (bottom), info_bonus (top)
        ax.bar(x, g["base_mean"], color=GOAL_COLORS[label], alpha=0.7, label="base_mean")
        ax.bar(x, g["info_bonus"], bottom=g["base_mean"], color="gold", alpha=0.85, label="info_bonus")
        # mark hard_fail_capped iterations
        capped = g[g["hard_fail_capped"] == True]
        if not capped.empty:
            ax.scatter(capped["iteration"], capped["overall_score"] + 0.05, marker="v", s=90,
                       color="red", edgecolor="black", linewidth=0.5, label="hard_fail_capped")
        # mark uncapped vs overall divergence
        diff = g["uncapped"] - g["overall_score"]
        ax.plot(x, g["uncapped"], "o", mfc="none", mec="black", ms=8, lw=1, label="uncapped")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label}: score decomposition (bar = base + bonus; v = hard-fail cap; ○ = uncapped)", fontsize=10)
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("iteration")
    fig.suptitle("How the overall_score is composed: base_mean + info_bonus, with hard-fail caps", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "d3_score_components.png", dpi=130)
    plt.close(fig)


# ---------- D4. Verifier-claim ERROR rate ----------
def fig_verifier_error_rate(wf: pd.DataFrame, ev: pd.DataFrame) -> None:
    merged = wf.merge(ev[["uuid", "n_claims", "n_pass", "n_fail", "n_error", "n_scored"]],
                       on="uuid", how="left")
    merged["error_rate"] = merged["n_error"] / merged["n_claims"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for label in GOAL_ORDER:
        g = merged[merged["goal_label"] == label].sort_values("iteration")
        if g.empty:
            continue
        ax.plot(g["iteration"], g["error_rate"] * 100, "-o",
                color=GOAL_COLORS[label], label=f"{label} (mean {g['error_rate'].mean()*100:.0f}%)", ms=6)
    ax.axhline(30, color="black", ls="--", lw=1, alpha=0.5)
    ax.text(0.5, 31, "30% rubric broken", fontsize=8, color="black")
    ax.set_xlabel("iteration")
    ax.set_ylabel("% of verifier CLAIMS that ERRORED (verifier itself broke)")
    ax.set_title("Rubric reliability: % of claims that errored per iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig(FIGS / "d4_verifier_error_rate.png", dpi=130)
    plt.close(fig)


# ---------- D5. Root-cause flowchart (text) ----------
def fig_rootcause(_: pd.DataFrame, __: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axis("off")
    nodes = [
        # (x, y, text, color)
        (0.08, 0.85, "Judge LLM:\ntemperature=1.0\nno seed\n→ ±0.08 noise floor", "#ffd9d9"),
        (0.08, 0.50, "Verifier checks:\n30% ERROR\n(syntax bugs,\nopenpyxl/shapely\nmissing)", "#ffd9d9"),
        (0.08, 0.15, "Rubric stability:\nhard-fail cap flips\nlate (50% of iters)\ndkpes claim set\nchanges mid-run", "#ffd9d9"),

        (0.35, 0.85, "Textual gradient:\nrepeats same\ncomplaints 20×\nleaks rubric\nclaim names", "#fff2cc"),
        (0.35, 0.50, "Mutator prompt:\nno scalar gradient\nliteral 'Boldness 100%'\n'Clean-slate'\nat stagnation", "#fff2cc"),
        (0.35, 0.15, "No pre-eval gate:\nbroken Python\ngets full verify\n→ 39 empty folders\n59 wall-hours wasted", "#fff2cc"),

        (0.62, 0.85, "Crossover:\n0% high+high pairs\n80% low+low\n45% of iters\nare crossovers", "#d9e8ff"),
        (0.62, 0.50, "Selection:\narchive peak=8\nMAX_CHILDREN=2 bypassed\nselection ≈ random\nw.r.t. fitness", "#d9e8ff"),
        (0.62, 0.15, "Topology:\nnever reaches 7\nactually 3-5 agents\nsame template across\niters; +70-130% lines\n= prompt scaffolding", "#d9e8ff"),

        (0.88, 0.50, "Net effect:\n\nfinal < seed\nfor every goal\n\nbest gain over\nseed: +0.04\n(within noise!)", "#ffaaaa"),
    ]
    for x, y, text, color in nodes:
        ax.add_patch(plt.Rectangle((x - 0.06, y - 0.08), 0.12, 0.16,
                                    facecolor=color, edgecolor="black", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5)

    # arrows
    arrows = [
        # noise floor → low signal → all selections useless
        (0.14, 0.85, 0.82, 0.55, "swamps signal"),
        (0.14, 0.50, 0.82, 0.50, "weighted by errors"),
        (0.14, 0.15, 0.82, 0.45, "drift mid-lineage"),
        (0.41, 0.85, 0.55, 0.55, "no flip → repeat"),
        (0.41, 0.50, 0.55, 0.55, "radicalism w/o ground"),
        (0.41, 0.15, 0.55, 0.50, "40% of attempts wasted"),
        (0.68, 0.85, 0.82, 0.55, "35% below both"),
        (0.68, 0.50, 0.82, 0.50, "uniform random sel."),
        (0.68, 0.15, 0.82, 0.45, "no real exploration"),
    ]
    for x1, y1, x2, y2, _label in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="grey", alpha=0.55, lw=1.2))

    # column headers
    ax.text(0.08, 0.99, "VERIFIER LAYER", ha="center", fontsize=11, weight="bold", color="#a40000")
    ax.text(0.35, 0.99, "MUTATOR LAYER", ha="center", fontsize=11, weight="bold", color="#7f6c00")
    ax.text(0.62, 0.99, "SELECTION LAYER", ha="center", fontsize=11, weight="bold", color="#003a8c")
    ax.text(0.88, 0.99, "OUTCOME", ha="center", fontsize=11, weight="bold", color="#7a0000")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Root-cause map: 9 compounding defects across 3 layers", fontsize=13, pad=20)
    fig.savefig(FIGS / "d5_rootcause.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    wf, ev = load()
    fig_crossover_outcomes(wf)
    fig_noise_floor(wf)
    fig_score_components(wf, ev)
    fig_verifier_error_rate(wf, ev)
    fig_rootcause(wf, ev)
    print("deep figures written")


if __name__ == "__main__":
    main()
