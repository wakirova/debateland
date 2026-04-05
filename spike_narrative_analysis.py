"""
Attention spikes + narrative archetypes (from existing combined_data_cleaned.csv).

No extra scrapers required: tag rows, detect spike days, chart and export tables.
Integrity / controversy is one archetype among launches, comparisons, dev tooling, etc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "figure.dpi": 150,
    }
)

DATA = Path(__file__).resolve().parent / "combined_data_cleaned.csv"
OUT_DIR = Path(__file__).resolve().parent

# (column_name, keywords) — multi-label: a row can match several themes
NARRATIVE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "trust_integrity",
        (
            "leak",
            "leaked",
            "dumb down",
            "dumbed",
            "undercover",
            "frustration regex",
            "scandal",
            "ccleaks",
            "source code",
            "map file",
        ),
    ),
    (
        "launch_model",
        (
            "announced",
            "release",
            " sonnet",
            " opus",
            " haiku",
            "claude 4",
            "claude 3",
            "new model",
            "1m token",
            "context window",
            "system card",
        ),
    ),
    (
        "developer_tooling",
        (
            "claude code",
            "cowork",
            "vscode",
            "terminal agent",
            "npm registry",
        ),
    ),
    (
        "comparison",
        (
            "chatgpt",
            "openai",
            "gemini",
            "gpt-4",
            "gpt4",
            "gpt 4",
        ),
    ),
    (
        "pricing_ops",
        (
            "price",
            "pricing",
            "subscription",
            "rate limit",
            "billing",
            "api cost",
            "pro plan",
        ),
    ),
    (
        "workflow_agent",
        (
            "openclaw",
            "agent",
            "telegram",
            "automation",
            "workflow",
        ),
    ),
]


def tag_narratives(text: str) -> dict[str, bool]:
    t = (text or "").lower()
    out: dict[str, bool] = {}
    for col, kws in NARRATIVE_RULES:
        out[col] = any(kw in t for kw in kws)
    return out


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df["full_text"].notna() & (df["full_text"].str.len() > 20)].copy()
    print(f"Rows for spike/narrative analysis: {len(df)}")

    tags = df["full_text"].astype(str).apply(tag_narratives)
    tag_df = pd.DataFrame(list(tags))
    for c in tag_df.columns:
        df[c] = tag_df[c].values

    df["day"] = df["date"].dt.normalize()
    daily = (
        df.groupby("day", as_index=False)
        .agg(
            n_posts=("full_text", "count"),
            engagement_sum=("total_engagement", "sum"),
            engagement_mean=("total_engagement", "mean"),
        )
        .sort_values("day")
    )

    # Spike days: robust z-score on daily volume (fallback to top percentiles if low variance)
    n = daily["n_posts"].to_numpy(dtype=float)
    med = np.median(n)
    mad = np.median(np.abs(n - med)) + 1e-9
    robust_z = 0.6745 * (n - med) / mad
    daily["volume_robust_z"] = robust_z

    z_std = float(np.std(n, ddof=0)) or 1e-9
    daily["volume_z"] = (n - float(np.mean(n))) / z_std

    spike_mask = (daily["volume_robust_z"] >= 2.0) | (daily["volume_z"] >= 2.0)
    if spike_mask.sum() == 0 and len(daily) >= 3:
        thr = np.percentile(n, 85)
        spike_mask = daily["n_posts"] >= thr

    daily["is_spike"] = spike_mask
    spike_days = set(daily.loc[daily["is_spike"], "day"])

    # Stacked narrative share per day (rows can count in multiple themes)
    theme_cols = [c for c, _ in NARRATIVE_RULES]
    daily_themes = df.groupby("day")[theme_cols].sum().reset_index()
    daily_plot = daily.merge(daily_themes, on="day", how="left")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax0 = axes[0]
    ax0.bar(daily_plot["day"], daily_plot["n_posts"], color="#4c72b0", alpha=0.85, width=0.9, label="Items per day")
    for sd in sorted(spike_days):
        ax0.axvline(sd, color="crimson", alpha=0.35, linewidth=2)
    ax0.set_ylabel("Count")
    ax0.set_title("Daily discourse volume (red band = spike days)")
    ax0.legend(loc="upper left")

    ax1 = axes[1]
    bottom = np.zeros(len(daily_plot))
    colors = sns.color_palette("Set2", n_colors=len(theme_cols))
    for i, col in enumerate(theme_cols):
        vals = daily_plot[col].fillna(0).to_numpy()
        ax1.bar(daily_plot["day"], vals, bottom=bottom, label=col, color=colors[i], width=0.9, alpha=0.9)
        bottom = bottom + vals
    ax1.set_ylabel("Tagged rows (multi-label)")
    ax1.set_title("Narrative mix over time (keyword-tagged rows per day)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "spikes_and_narratives.png", bbox_inches="tight")
    plt.close()
    print(" Saved spikes_and_narratives.png")

    # Engagement by theme (among rows that hit that theme)
    rows_per_theme = {c: int(df[c].sum()) for c in theme_cols}
    mean_eng_by_theme = {
        c: float(df.loc[df[c], "total_engagement"].mean()) if df[c].any() else float("nan")
        for c in theme_cols
    }
    theme_summary = pd.DataFrame(
        {
            "theme": theme_cols,
            "row_count": [rows_per_theme[c] for c in theme_cols],
            "mean_total_engagement": [mean_eng_by_theme[c] for c in theme_cols],
        }
    ).sort_values("mean_total_engagement", ascending=False)
    theme_summary.to_csv(OUT_DIR / "narrative_theme_summary.csv", index=False)
    print(" Saved narrative_theme_summary.csv")

    daily_out = daily_plot[
        ["day", "n_posts", "engagement_sum", "is_spike", "volume_robust_z", "volume_z"] + theme_cols
    ].copy()
    daily_out.to_csv(OUT_DIR / "daily_spike_metrics.csv", index=False)
    print(" Saved daily_spike_metrics.csv")

    # Top URLs on spike days (high engagement first)
    spike_df = df[df["day"].isin(spike_days)].copy()
    if len(spike_df) and "url" in spike_df.columns:
        top_spike = (
            spike_df.sort_values("total_engagement", ascending=False)
            .head(25)[["day", "platform", "total_engagement", "url", "full_text"]]
            .assign(preview=lambda x: x["full_text"].str.slice(0, 120))
            .drop(columns=["full_text"])
        )
        top_spike.to_csv(OUT_DIR / "spike_day_top_items.csv", index=False)
        print(" Saved spike_day_top_items.csv")

    fig2, ax = plt.subplots(figsize=(9, 5))
    t2 = theme_summary.dropna(subset=["mean_total_engagement"]).sort_values("mean_total_engagement")
    ax.barh(t2["theme"], t2["mean_total_engagement"], color=sns.color_palette("deep", n_colors=len(t2)))
    ax.set_title("Average total_engagement when row matches theme (keyword tag)")
    ax.set_xlabel("Mean total_engagement")
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "engagement_by_narrative_theme.png", bbox_inches="tight")
    plt.close()
    print(" Saved engagement_by_narrative_theme.png")

    print("\n" + "=" * 60)
    print("SPIKE + NARRATIVE SUMMARY (for playbook writeup)")
    print("=" * 60)
    print(f"Spike days ({len(spike_days)}): {sorted(str(d) for d in spike_days)[:12]}{' ...' if len(spike_days) > 12 else ''}")
    print("\nTheme row counts (multi-label):")
    for _, r in theme_summary.sort_values("row_count", ascending=False).iterrows():
        print(f"  {r['theme']}: {int(r['row_count'])} rows, mean engagement {r['mean_total_engagement']:.1f}")

    ti = int(df["trust_integrity"].sum())
    tot = len(df)
    print(
        f"\nTrust/integrity-tagged share: {ti}/{tot} ({100*ti/tot:.1f}%) — "
        "use as one layer of the playbook, not the whole story."
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
