import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", palette="muted", rc={
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.4,
})

BLUE = "#378ADD"
CORAL = "#D85A30"


def plot_win_probability(home_team: str, away_team: str, prob: float) -> plt.Figure:
    """Horizontal probability bar showing split between home and away."""
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.barh(0, prob, color=BLUE, height=0.5, label=home_team)
    ax.barh(0, 1 - prob, left=prob, color=CORAL, height=0.5, label=away_team)
    ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.text(prob / 2, 0, f"{prob:.0%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(prob + (1 - prob) / 2, 0, f"{1 - prob:.0%}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4),
              ncol=2, frameon=False, fontsize=10)
    ax.set_title(f"{home_team}  vs  {away_team}", fontsize=12, pad=8)
    fig.tight_layout()
    return fig


def plot_shap_bar(shap_df: pd.DataFrame, n: int = 12) -> plt.Figure:
    """Horizontal bar chart of top SHAP values, coloured by direction."""
    top = shap_df.head(n).copy()
    top["color"] = top["shap_value"].apply(lambda v: BLUE if v > 0 else CORAL)
    fig, ax = plt.subplots(figsize=(6, n * 0.4 + 0.6))
    ax.barh(
        top["feature"][::-1],
        top["shap_value"][::-1],
        color=top["color"][::-1],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (positive = favours home team)", fontsize=10)
    ax.set_title("Feature impact on prediction", fontsize=12)
    fig.tight_layout()
    return fig


def plot_calibration(y_true, y_prob, n_bins: int = 8) -> plt.Figure:
    """Reliability diagram — are predicted 60% games actually 60% wins?"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=BLUE, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_h2h(df_feat: pd.DataFrame, team_a: str, team_b: str) -> plt.Figure:
    """Bar chart of playoff H2H win rate by season."""
    mask = (
        (df_feat["HOME_TEAM"].str.contains(team_a, case=False, na=False) &
         df_feat["AWAY_TEAM"].str.contains(team_b, case=False, na=False)) |
        (df_feat["HOME_TEAM"].str.contains(team_b, case=False, na=False) &
         df_feat["AWAY_TEAM"].str.contains(team_a, case=False, na=False))
    )
    h2h = df_feat[mask].copy()

    if h2h.empty:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No playoff H2H history found",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        return fig

    h2h["A_WIN"] = h2h.apply(
        lambda r: (
            int(r["HOME_WIN"] == 1)
            if team_a.lower() in r["HOME_TEAM"].lower()
            else int(r["HOME_WIN"] == 0)
        ),
        axis=1,
    )
    by_year = h2h.groupby("SEASON_YEAR")["A_WIN"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(by_year["SEASON_YEAR"].astype(str), by_year["A_WIN"],
           color=BLUE, alpha=0.8)
    ax.axhline(0.5, color=CORAL, linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{team_a} win rate")
    ax.set_xlabel("Season")
    ax.set_title(f"Playoff H2H: {team_a} vs {team_b}", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_radar(
    home_stats: dict, away_stats: dict,
    home_name: str, away_name: str,
) -> plt.Figure:
    """Polar radar chart comparing six team dimensions."""
    cats = ["ORTG", "DRTG\n(inv)", "TS%", "TOV%\n(inv)", "OREB%", "SRS"]
    h_vals = [
        home_stats["ORTG"],
        200 - home_stats["DRTG"],    # invert so higher = better
        home_stats["TS"] * 100,
        100 - home_stats["TOV"],     # invert so higher = better
        home_stats["OREB"],
        50 + home_stats["SRS"] * 5,
    ]
    a_vals = [
        away_stats["ORTG"],
        200 - away_stats["DRTG"],
        away_stats["TS"] * 100,
        100 - away_stats["TOV"],
        away_stats["OREB"],
        50 + away_stats["SRS"] * 5,
    ]
    all_vals = h_vals + a_vals
    mn, mx = min(all_vals) - 1, max(all_vals) + 1
    h_norm = [(v - mn) / (mx - mn) for v in h_vals]
    a_norm = [(v - mn) / (mx - mn) for v in a_vals]

    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    h_norm += h_norm[:1]
    a_norm += a_norm[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, h_norm, "o-", linewidth=1.5, color=BLUE, label=home_name)
    ax.fill(angles, h_norm, alpha=0.15, color=BLUE)
    ax.plot(angles, a_norm, "o-", linewidth=1.5, color=CORAL, label=away_name)
    ax.fill(angles, a_norm, alpha=0.15, color=CORAL)
    ax.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    ax.set_title("Team comparison", fontsize=12, pad=18)
    fig.tight_layout()
    return fig