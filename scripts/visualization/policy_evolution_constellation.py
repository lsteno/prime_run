#!/usr/bin/env python3
"""Render a 3D policy-evolution constellation from aggregated RLM training metrics."""

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
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def render_static(df: pd.DataFrame, output_png: Path, output_svg: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )

    cost_k = df["cost_total_tokens_mean"].to_numpy() / 1000.0
    correctness = df["correctness"].to_numpy()
    steps = df["step"].to_numpy()
    llm_usage = df["llm_usage_rate"].to_numpy()
    num_llm = df["num_llm_mean"].fillna(0).to_numpy()

    fig = plt.figure(figsize=(11.5, 8.4), facecolor="#070a12")
    ax = fig.add_subplot(111, projection="3d", facecolor="#070a12")

    # Smooth trail gives a clean "orbit" while raw points still show individual checkpoints.
    order = np.argsort(steps)
    x_s = _smooth(cost_k[order], 5)
    y_s = _smooth(correctness[order], 5)
    z_s = _smooth(steps[order], 5)

    ax.plot(x_s, y_s, z_s, color="#e8edf7", alpha=0.38, linewidth=2.4, zorder=1)
    ax.plot(x_s, np.full_like(y_s, max(0.0, correctness.min() - 0.035)), z_s, color="#58a6ff", alpha=0.10, linewidth=1.5)
    ax.plot(np.full_like(x_s, cost_k.min() - 4), y_s, z_s, color="#ff6ad5", alpha=0.10, linewidth=1.5)

    sizes = 42 + np.clip(num_llm, 0, np.nanpercentile(num_llm, 95) if len(num_llm) else 1) * 9
    scatter = ax.scatter(
        cost_k,
        correctness,
        steps,
        c=llm_usage,
        cmap="magma",
        s=sizes,
        alpha=0.92,
        edgecolor="#f6f7fb",
        linewidth=0.45,
        depthshade=False,
        zorder=4,
    )

    # Dim projection shadows onto the floor make the volume easier to read.
    ax.scatter(cost_k, correctness, np.full_like(steps, steps.min()), c=llm_usage, cmap="magma", s=sizes * 0.35, alpha=0.10, depthshade=False)

    best_idx = int(np.nanargmax(correctness))
    final_idx = int(np.nanargmax(steps))
    start_idx = int(np.nanargmin(steps))
    markers = [
        (start_idx, "step 50", "#8be9fd"),
        (best_idx, f"best batch\\nstep {int(steps[best_idx])}", "#50fa7b"),
        (final_idx, "final", "#ffb86c"),
    ]
    for idx, label, color in markers:
        ax.scatter([cost_k[idx]], [correctness[idx]], [steps[idx]], s=210, color=color, edgecolor="white", linewidth=1.2, depthshade=False)
        ax.text(cost_k[idx], correctness[idx] + 0.018, steps[idx] + 1.5, label, color=color, fontsize=9, ha="center")

    ax.set_title("Policy Evolution Constellation\nLoRA r4/a8, LR 1e-4, depth-1 RLM RLVR", color="#f6f7fb", pad=16)
    ax.set_xlabel("Mean rollout cost (k tokens)", color="#dfe7f5", labelpad=10)
    ax.set_ylabel("Batch correctness", color="#dfe7f5", labelpad=10)
    ax.set_zlabel("Training step", color="#dfe7f5", labelpad=8)

    ax.set_xlim(max(0, np.nanmin(cost_k) - 4), np.nanmax(cost_k) + 8)
    ax.set_ylim(max(0, np.nanmin(correctness) - 0.05), min(1.0, np.nanmax(correctness) + 0.08))
    ax.set_zlim(np.nanmin(steps), np.nanmax(steps) + 3)
    ax.view_init(elev=24, azim=-58)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_tick_params(colors="#b8c3d9")
        axis.line.set_color("#566079")
    ax.xaxis.pane.set_facecolor((0.03, 0.04, 0.08, 0.25))
    ax.yaxis.pane.set_facecolor((0.03, 0.04, 0.08, 0.12))
    ax.zaxis.pane.set_facecolor((0.03, 0.04, 0.08, 0.08))
    ax.grid(True, color="#273149", alpha=0.18)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("LLM subcall usage rate", color="#dfe7f5")
    cbar.ax.yaxis.set_tick_params(color="#b8c3d9")
    plt.setp(cbar.ax.get_yticklabels(), color="#b8c3d9")
    cbar.outline.set_edgecolor("#566079")

    note = (
        "Each point is one training batch with observed attempt logs. "
        "Color shows fraction of rollouts using Gemini subcalls; marker size scales with mean subcall count."
    )
    fig.text(0.06, 0.035, note, color="#9aa7bd", fontsize=9)

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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["cost_total_tokens_mean"] / 1000.0,
            y=df["correctness"],
            z=df["step"],
            mode="lines",
            line=dict(color="rgba(220,230,255,0.35)", width=5),
            name="trajectory",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["cost_total_tokens_mean"] / 1000.0,
            y=df["correctness"],
            z=df["step"],
            mode="markers",
            marker=dict(
                size=5 + np.clip(df["num_llm_mean"].fillna(0), 0, 10),
                color=df["llm_usage_rate"],
                colorscale="Magma",
                cmin=0,
                cmax=1,
                opacity=0.92,
                line=dict(color="white", width=0.6),
                colorbar=dict(title="LLM usage"),
            ),
            customdata=np.stack(
                [
                    df["reward_mean"].fillna(np.nan),
                    df["num_llm_mean"].fillna(np.nan),
                    df["attempts_finished"].fillna(0),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "step=%{z}<br>"
                "cost=%{x:.1f}k tokens<br>"
                "correctness=%{y:.3f}<br>"
                "LLM usage=%{marker.color:.2f}<br>"
                "reward=%{customdata[0]:.3f}<br>"
                "mean #LLM=%{customdata[1]:.2f}<br>"
                "attempts=%{customdata[2]:.0f}<extra></extra>"
            ),
            name="checkpoints",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Policy Evolution Constellation — LoRA r4/a8 LR 1e-4",
        scene=dict(
            xaxis_title="Mean rollout cost (k tokens)",
            yaxis_title="Batch correctness",
            zaxis_title="Training step",
            camera=dict(eye=dict(x=1.65, y=-1.7, z=1.25)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["correctness", "cost_total_tokens_mean", "llm_usage_rate"])
    df = df[(df["step"] >= 50) & (df["step"] < 150)].copy()
    df = df.sort_values("step")
    if df.empty:
        raise SystemExit("No rows with correctness, cost, and LLM usage metrics.")

    render_static(
        df,
        args.out_dir / "policy_evolution_constellation_r4_lr1e4.png",
        args.out_dir / "policy_evolution_constellation_r4_lr1e4.svg",
    )
    render_interactive(df, args.out_dir / "policy_evolution_constellation_r4_lr1e4.html")
    df.to_csv(args.out_dir / "policy_evolution_constellation_r4_lr1e4_plotted_points.csv", index=False)


if __name__ == "__main__":
    main()
