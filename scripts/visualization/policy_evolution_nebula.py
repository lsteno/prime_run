#!/usr/bin/env python3
"""Render a polished 2D policy phase portrait from RLM training metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
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
            "savefig.dpi": 320,
            "axes.labelsize": 10,
            "axes.titlesize": 13,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.unicode_minus": False,
        }
    )

    df = df.sort_values("step").copy()
    step = df["step"].to_numpy()
    cost_k = df["cost_total_tokens_mean"].to_numpy() / 1000.0
    correctness = df["correctness"].to_numpy()
    llm_usage = df["llm_usage_rate"].to_numpy()
    mean_llm = df["num_llm_mean"].fillna(0).to_numpy()
    reward = df["reward_mean"].fillna(np.nan).to_numpy()
    duration = df["duration_s_mean"].fillna(np.nan).to_numpy()

    fig = plt.figure(figsize=(15.5, 9.0), facecolor="#060812")
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        width_ratios=[1.45, 1.45, 1.45, 0.95, 0.95],
        height_ratios=[0.7, 1.0, 1.0],
        wspace=0.34,
        hspace=0.36,
    )
    ax = fig.add_subplot(gs[:, :3], facecolor="#090d19")
    ax_usage = fig.add_subplot(gs[0, 3:], facecolor="#090d19")
    ax_vitals = fig.add_subplot(gs[1:, 3:], facecolor="#090d19")

    x_min, x_max = np.nanmin(cost_k), np.nanmax(cost_k)
    y_min, y_max = np.nanmin(correctness), np.nanmax(correctness)
    x_pad = max(3.0, (x_max - x_min) * 0.08)
    y_pad = max(0.025, (y_max - y_min) * 0.16)
    xlim = (max(0.0, x_min - x_pad), x_max + x_pad)
    ylim = (max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    # Subtle phase-space utility background: up-left is better.
    gx, gy = np.meshgrid(np.linspace(*xlim, 260), np.linspace(*ylim, 260))
    utility = _normalize(gy) - 0.72 * _normalize(gx)
    ax.contourf(gx, gy, utility, levels=18, cmap="mako" if "mako" in plt.colormaps() else "viridis", alpha=0.19)
    ax.contour(gx, gy, utility, levels=8, colors="#dbe7ff", linewidths=0.35, alpha=0.10)

    points = np.array([cost_k, correctness]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    step_norm = _normalize(step[:-1])

    # Draw glow under the actual trajectory.
    for width, alpha in [(18, 0.035), (11, 0.06), (6, 0.10)]:
        glow = LineCollection(segments, cmap="cool", norm=plt.Normalize(0, 1))
        glow.set_array(step_norm)
        glow.set_linewidth(width)
        glow.set_alpha(alpha)
        ax.add_collection(glow)

    line = LineCollection(segments, cmap="cool", norm=plt.Normalize(0, 1))
    line.set_array(step_norm)
    line.set_linewidth(2.5)
    line.set_alpha(0.92)
    ax.add_collection(line)

    sizes = 54 + np.clip(mean_llm, 0, np.nanpercentile(mean_llm, 95)) * 13
    # Outer halos encode subcall usage visually without making the path unreadable.
    ax.scatter(
        cost_k,
        correctness,
        s=sizes * (1.8 + llm_usage * 1.7),
        c=llm_usage,
        cmap="plasma",
        alpha=0.12,
        linewidths=0,
        zorder=3,
    )
    scatter = ax.scatter(
        cost_k,
        correctness,
        s=sizes,
        c=llm_usage,
        cmap="plasma",
        edgecolors="#f7fbff",
        linewidths=0.45,
        alpha=0.96,
        zorder=4,
    )

    # Direction arrows every ~14 checkpoints.
    for idx in range(8, len(cost_k) - 1, 14):
        dx = cost_k[idx + 1] - cost_k[idx - 1]
        dy = correctness[idx + 1] - correctness[idx - 1]
        ax.annotate(
            "",
            xy=(cost_k[idx] + dx * 0.45, correctness[idx] + dy * 0.45),
            xytext=(cost_k[idx] - dx * 0.45, correctness[idx] - dy * 0.45),
            arrowprops=dict(arrowstyle="->", color="#e9f3ff", alpha=0.55, lw=1.0),
            zorder=5,
        )

    start_idx = int(np.nanargmin(step))
    best_idx = int(np.nanargmax(correctness))
    final_idx = int(np.nanargmax(step))
    for idx, label, color, offset in [
        (start_idx, f"start\nstep {int(step[idx])}", "#8be9fd", (-7, -0.018)),
        (best_idx, f"best batch\nstep {int(step[idx])}", "#50fa7b", (4, 0.020)),
        (final_idx, f"final\nstep {int(step[idx])}", "#ffb86c", (4, -0.018)),
    ]:
        ax.scatter([cost_k[idx]], [correctness[idx]], s=260, color=color, edgecolors="white", linewidths=1.2, zorder=7)
        ax.annotate(
            label,
            xy=(cost_k[idx], correctness[idx]),
            xytext=(cost_k[idx] + offset[0], correctness[idx] + offset[1]),
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.75),
            zorder=8,
        )

    ax.annotate(
        "better policy direction",
        xy=(xlim[0] + (xlim[1] - xlim[0]) * 0.12, ylim[1] - (ylim[1] - ylim[0]) * 0.10),
        xytext=(xlim[0] + (xlim[1] - xlim[0]) * 0.42, ylim[0] + (ylim[1] - ylim[0]) * 0.18),
        color="#d8e8ff",
        fontsize=10,
        arrowprops=dict(arrowstyle="simple", color="#58a6ff", alpha=0.23),
        alpha=0.82,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Mean rollout cost (k tokens)")
    ax.set_ylabel("Batch correctness")
    ax.set_title("Policy Phase Portrait: learning to move up-left", color="#f4f7ff", loc="left", pad=14, fontsize=18, fontweight="bold")
    ax.text(
        0.0,
        1.015,
        "LoRA r4/a8, LR 1e-4, depth-1 RLM RLVR. Color/halo = Gemini subcall usage; size = mean subcall count.",
        transform=ax.transAxes,
        color="#9ba8c4",
        fontsize=9.5,
        ha="left",
    )
    ax.grid(color="#2f3a52", alpha=0.24, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#3b4660")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.018)
    cbar.set_label("LLM subcall usage rate", color="#dfe7f5")
    cbar.ax.tick_params(colors="#aebbd1")
    cbar.outline.set_edgecolor("#3b4660")

    # Top-right compact subcall heat strip.
    usage_img = np.vstack([llm_usage, _smooth(llm_usage, 7), _normalize(mean_llm)])
    ax_usage.imshow(usage_img, aspect="auto", cmap="plasma", interpolation="bicubic", extent=[step.min(), step.max(), 0, 3])
    ax_usage.set_yticks([0.5, 1.5, 2.5], ["mean #LLM", "usage smooth", "usage"])
    ax_usage.set_title("Subcall rhythm", color="#f4f7ff", loc="left", fontsize=12, fontweight="bold")
    ax_usage.set_xlabel("step")
    ax_usage.tick_params(colors="#aebbd1")
    for spine in ax_usage.spines.values():
        spine.set_color("#3b4660")

    # Right side temporal "vitals" panel.
    ax_vitals.plot(step, _smooth(correctness, 5), color="#50fa7b", lw=2.4, label="correctness")
    ax_vitals.plot(step, _normalize(cost_k) * (ylim[1] - ylim[0]) + ylim[0], color="#ffb86c", lw=1.7, alpha=0.92, label="cost, normalized")
    ax_vitals.plot(step, _normalize(duration) * (ylim[1] - ylim[0]) + ylim[0], color="#8be9fd", lw=1.3, alpha=0.82, label="duration, normalized")
    if np.isfinite(reward).any():
        ax_vitals.plot(step, _normalize(reward) * (ylim[1] - ylim[0]) + ylim[0], color="#bd93f9", lw=1.25, alpha=0.72, label="reward, normalized")
    ax_vitals.fill_between(step, ylim[0], _smooth(correctness, 5), color="#50fa7b", alpha=0.08)
    ax_vitals.set_xlim(step.min(), step.max())
    ax_vitals.set_ylim(*ylim)
    ax_vitals.set_xlabel("training step")
    ax_vitals.set_ylabel("correctness / normalized vitals")
    ax_vitals.set_title("Trajectory vitals", color="#f4f7ff", loc="left", fontsize=12, fontweight="bold")
    ax_vitals.grid(color="#2f3a52", alpha=0.24, linewidth=0.6)
    ax_vitals.legend(loc="lower right", frameon=False, fontsize=8, labelcolor="#dfe7f5")
    ax_vitals.tick_params(colors="#aebbd1")
    for spine in ax_vitals.spines.values():
        spine.set_color("#3b4660")

    for axis in [ax, ax_usage, ax_vitals]:
        axis.xaxis.label.set_color("#dfe7f5")
        axis.yaxis.label.set_color("#dfe7f5")
        axis.tick_params(colors="#aebbd1")

    fig.text(
        0.018,
        0.025,
        "Reading guide: a healthy run drifts up-left. Horizontal loops mean cost changes without accuracy gain; vertical moves mean accuracy changes at similar cost.",
        color="#8793ad",
        fontsize=8.7,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(output_svg, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_interactive(df: pd.DataFrame, output_html: Path) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    df = df.sort_values("step").copy()
    cost_k = df["cost_total_tokens_mean"] / 1000.0
    mean_llm = df["num_llm_mean"].fillna(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cost_k,
            y=df["correctness"],
            mode="lines",
            line=dict(color="rgba(230,238,255,0.28)", width=5),
            hoverinfo="skip",
            name="trajectory",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cost_k,
            y=df["correctness"],
            mode="markers+text",
            marker=dict(
                size=9 + np.clip(mean_llm, 0, 12) * 1.8,
                color=df["llm_usage_rate"],
                colorscale="Plasma",
                cmin=0,
                cmax=1,
                line=dict(color="white", width=0.7),
                colorbar=dict(title="LLM usage"),
            ),
            text=np.where(df["step"].isin([df["step"].min(), df["correctness"].idxmax(), df["step"].max()]), df["step"].astype(str), ""),
            textposition="top center",
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
        title="Policy Phase Portrait — cost vs correctness",
        xaxis_title="Mean rollout cost (k tokens)",
        yaxis_title="Batch correctness",
        width=1100,
        height=760,
        margin=dict(l=70, r=40, t=70, b=70),
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
        args.out_dir / "policy_evolution_nebula_r4_lr1e4.png",
        args.out_dir / "policy_evolution_nebula_r4_lr1e4.svg",
    )
    render_interactive(df, args.out_dir / "policy_evolution_nebula_r4_lr1e4.html")
    df.sort_values("step").to_csv(args.out_dir / "policy_evolution_nebula_r4_lr1e4_plotted_points.csv", index=False)


if __name__ == "__main__":
    main()
