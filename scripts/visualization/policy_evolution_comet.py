#!/usr/bin/env python3
"""Render a clean paper-style policy evolution comet plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def render_static(df: pd.DataFrame, output_png: Path, output_svg: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 180,
            "savefig.dpi": 360,
            "axes.labelsize": 13,
            "axes.titlesize": 18,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    df = df.sort_values("step")
    step = df["step"].to_numpy()
    cost = df["cost_total_tokens_mean"].to_numpy() / 1000.0
    acc = df["correctness"].to_numpy()
    llm_usage = df["llm_usage_rate"].to_numpy()
    mean_llm = df["num_llm_mean"].fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(13.6, 8.0), facecolor="#050712", constrained_layout=True)
    ax.set_facecolor("#09101e")

    x_min, x_max = np.nanmin(cost), np.nanmax(cost)
    y_min, y_max = np.nanmin(acc), np.nanmax(acc)
    x_pad = max(4.0, 0.08 * (x_max - x_min))
    y_pad = max(0.035, 0.18 * (y_max - y_min))
    ax.set_xlim(max(0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0, y_min - y_pad), min(1.0, y_max + y_pad))

    # Efficient frontier background: upper-left is brighter.
    gx, gy = np.meshgrid(
        np.linspace(*ax.get_xlim(), 320),
        np.linspace(*ax.get_ylim(), 260),
    )
    frontier = _norm(gy) - 0.82 * _norm(gx)
    ax.imshow(
        frontier,
        extent=[*ax.get_xlim(), *ax.get_ylim()],
        origin="lower",
        cmap="viridis",
        alpha=0.13,
        aspect="auto",
        zorder=0,
    )
    ax.contour(gx, gy, frontier, levels=9, colors="white", linewidths=0.45, alpha=0.10, zorder=1)

    # Raw trajectory plus a smoothed comet spine.
    pts = np.array([cost, acc]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    progress = _norm(step[:-1])

    for lw, alpha in [(16, 0.04), (9, 0.075), (5, 0.12)]:
        glow = LineCollection(segs, cmap="winter", norm=plt.Normalize(0, 1), linewidth=lw, alpha=alpha, zorder=2)
        glow.set_array(progress)
        ax.add_collection(glow)

    lc = LineCollection(segs, cmap="winter", norm=plt.Normalize(0, 1), linewidth=2.6, alpha=0.80, zorder=3)
    lc.set_array(progress)
    ax.add_collection(lc)

    cost_s = _smooth(cost, 7)
    acc_s = _smooth(acc, 7)
    ax.plot(cost_s, acc_s, color="#eaf3ff", lw=2.0, alpha=0.30, zorder=4)

    sizes = 46 + np.clip(mean_llm, 0, np.nanpercentile(mean_llm, 95)) * 9
    ax.scatter(cost, acc, s=sizes * 2.7, c=llm_usage, cmap="plasma", alpha=0.10, linewidth=0, zorder=5)
    sc = ax.scatter(
        cost,
        acc,
        s=sizes,
        c=llm_usage,
        cmap="plasma",
        vmin=0,
        vmax=1,
        edgecolors="#f3f8ff",
        linewidths=0.55,
        alpha=0.94,
        zorder=6,
    )

    # Sparse arrows make time direction legible without turning the plot into spaghetti.
    for i in range(10, len(cost) - 2, 18):
        ax.annotate(
            "",
            xy=(cost[i + 2], acc[i + 2]),
            xytext=(cost[i - 2], acc[i - 2]),
            arrowprops=dict(arrowstyle="-|>", color="#dcecff", lw=1.2, alpha=0.55),
            zorder=7,
        )

    start_idx = int(np.argmin(step))
    best_idx = int(np.argmax(acc))
    final_idx = int(np.argmax(step))
    labels = [
        (start_idx, "start", "#76e4f7", (8, 8)),
        (best_idx, "best", "#50fa7b", (9, 10)),
        (final_idx, "final", "#ffb86c", (-52, -18)),
    ]
    for idx, label, color, offset in labels:
        ax.scatter([cost[idx]], [acc[idx]], s=230, color=color, edgecolor="white", linewidth=1.2, zorder=9)
        ax.annotate(
            f"{label}\nstep {int(step[idx])}",
            (cost[idx], acc[idx]),
            textcoords="offset points",
            xytext=offset,
            color=color,
            fontsize=10,
            weight="bold",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.75),
            zorder=10,
        )

    ax.annotate(
        "higher accuracy\nlower cost",
        xy=(0.11, 0.86),
        xycoords="axes fraction",
        xytext=(0.34, 0.20),
        textcoords="axes fraction",
        color="#c4d7f7",
        fontsize=12,
        ha="center",
        arrowprops=dict(arrowstyle="simple", color="#58a6ff", alpha=0.23),
    )

    ax.set_title("Policy Evolution Comet", color="#f5f7ff", loc="left", pad=14, fontweight="bold")
    ax.text(
        0.0,
        1.015,
        "r4/a8 LoRA, LR 1e-4. Each point is a training checkpoint; color is Gemini subcall usage and size is mean subcall count.",
        transform=ax.transAxes,
        color="#a9b5cd",
        fontsize=10.5,
        ha="left",
    )
    ax.set_xlabel("Mean rollout cost (k tokens)", color="#dfe8f8")
    ax.set_ylabel("Batch correctness", color="#dfe8f8")
    ax.grid(color="#34405c", alpha=0.22, linewidth=0.7)
    ax.tick_params(colors="#b8c4db")
    for spine in ax.spines.values():
        spine.set_color("#394763")

    cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.036)
    cbar.set_label("LLM subcall usage rate", color="#dfe8f8")
    cbar.ax.tick_params(colors="#b8c4db")
    cbar.outline.set_edgecolor("#394763")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(output_svg, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_interactive(df: pd.DataFrame, output_html: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    df = df.sort_values("step")
    cost_k = df["cost_total_tokens_mean"] / 1000.0
    mean_llm = df["num_llm_mean"].fillna(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cost_k,
            y=df["correctness"],
            mode="lines",
            line=dict(color="rgba(225,238,255,0.35)", width=4),
            hoverinfo="skip",
            name="trajectory",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cost_k,
            y=df["correctness"],
            mode="markers",
            marker=dict(
                size=8 + np.clip(mean_llm, 0, 14) * 1.55,
                color=df["llm_usage_rate"],
                colorscale="Plasma",
                cmin=0,
                cmax=1,
                opacity=0.94,
                line=dict(color="white", width=0.65),
                colorbar=dict(title="LLM usage"),
            ),
            customdata=np.stack(
                [
                    df["step"],
                    df["reward_mean"].fillna(np.nan),
                    mean_llm,
                    df["duration_s_mean"].fillna(np.nan),
                    df["attempts_finished"].fillna(np.nan),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "step=%{customdata[0]:.0f}<br>"
                "cost=%{x:.1f}k tokens<br>"
                "correctness=%{y:.3f}<br>"
                "LLM usage=%{marker.color:.2f}<br>"
                "mean #LLM=%{customdata[2]:.2f}<br>"
                "reward=%{customdata[1]:.3f}<br>"
                "duration=%{customdata[3]:.1f}s<br>"
                "attempts=%{customdata[4]:.0f}<extra></extra>"
            ),
            name="checkpoints",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Policy Evolution Comet",
        xaxis_title="Mean rollout cost (k tokens)",
        yaxis_title="Batch correctness",
        width=1050,
        height=720,
        margin=dict(l=72, r=40, t=76, b=70),
    )
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["correctness", "cost_total_tokens_mean", "llm_usage_rate"])
    df = df[(df["step"] >= 50) & (df["step"] < 150)].copy()
    if df.empty:
        raise SystemExit("No rows with correctness, cost, and LLM usage metrics.")

    render_static(
        df,
        args.out_dir / "policy_evolution_comet_r4_lr1e4.png",
        args.out_dir / "policy_evolution_comet_r4_lr1e4.svg",
    )
    render_interactive(df, args.out_dir / "policy_evolution_comet_r4_lr1e4.html")
    df.sort_values("step").to_csv(args.out_dir / "policy_evolution_comet_r4_lr1e4_plotted_points.csv", index=False)


if __name__ == "__main__":
    main()
