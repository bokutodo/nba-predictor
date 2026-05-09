import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.predict import predict_matchup, explain_prediction, get_team_stats
from src.viz import plot_h2h, plot_radar

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Playoffs Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* App title */
.app-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    line-height: 1;
    margin: 0 0 0.15rem;
    color: #f0f0f0;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 2rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #2a2a2a;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #666;
    padding: 0.6rem 1.4rem;
    border-bottom: 2px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #ff6b35 !important;
    border-bottom: 2px solid #ff6b35 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #f0f0f0 !important;
    border-radius: 6px;
}

/* Predict button */
.stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: #ff6b35 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.metric-card {
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
}
.metric-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f0f0;
    line-height: 1;
}
.metric-value.accent { color: #ff6b35; }
.metric-sub {
    font-size: 0.75rem;
    color: #555;
    margin-top: 3px;
}

/* Prob bar container */
.prob-wrap {
    margin: 1.5rem 0 0.5rem;
}
.prob-teams {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
}
.prob-team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.prob-pct {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #888;
}
.prob-bar-outer {
    height: 10px;
    background: #1e1e1e;
    border-radius: 5px;
    overflow: hidden;
    display: flex;
}
.prob-bar-home {
    background: #ff6b35;
    height: 100%;
    transition: width 0.6s ease;
}
.prob-bar-away {
    background: #3a8eff;
    height: 100%;
}

/* Section headers */
.section-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 6px;
    margin: 2rem 0 1rem;
}

/* Stat comparison table */
.stat-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.stat-table th {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
    padding: 6px 10px;
    text-align: center;
    border-bottom: 1px solid #1e1e1e;
}
.stat-table td {
    padding: 8px 10px;
    text-align: center;
    border-bottom: 1px solid #161616;
    color: #ccc;
    font-variant-numeric: tabular-nums;
}
.stat-table td.stat-name {
    text-align: left;
    color: #777;
    font-size: 0.8rem;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.stat-table td.winner {
    color: #f0f0f0;
    font-weight: 500;
}
.stat-table td.loser { color: #444; }

/* Glossary cards */
.gloss-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 1rem;
}
.gloss-card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.gloss-term {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #ff6b35;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.gloss-abbr {
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.gloss-desc {
    font-size: 0.83rem;
    color: #888;
    line-height: 1.55;
}
.gloss-example {
    font-size: 0.75rem;
    color: #555;
    margin-top: 5px;
    font-style: italic;
}

/* Winner banner */
.winner-banner {
    background: linear-gradient(135deg, #1a1108 0%, #1a1108 100%);
    border: 1px solid #3d2800;
    border-left: 3px solid #ff6b35;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.winner-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aa6600;
}
.winner-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ff6b35;
    line-height: 1;
}

/* Info box */
.info-box {
    background: #0d0d0d;
    border: 1px dashed #222;
    border-radius: 10px;
    padding: 2.5rem;
    text-align: center;
    color: #444;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_parquet("data/processed/features.parquet")

df_feat = load_features()
all_teams = sorted(df_feat["HOME_TEAM"].unique().tolist())
seasons   = sorted(df_feat["SEASON_YEAR"].unique().tolist(), reverse=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 1rem;'>
        <div style='font-family: Barlow Condensed, sans-serif; font-size: 1.5rem;
                    font-weight: 700; color: #f0f0f0; letter-spacing: -0.3px;'>
            🏀 Playoffs
        </div>
        <div style='font-size: 0.7rem; color: #555; letter-spacing: 0.1em;
                    text-transform: uppercase; margin-top: 2px;'>
            Win Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    home_team   = st.selectbox("Home team",   all_teams, index=0)
    away_team   = st.selectbox("Away team",   all_teams, index=min(1, len(all_teams)-1))
    season_year = st.selectbox("Season",      seasons,   index=0)

    st.markdown("---")
    st.markdown("""
    <div style='font-family: Barlow Condensed, sans-serif; font-size: 0.7rem;
                letter-spacing: 0.1em; text-transform: uppercase; color: #555;
                margin-bottom: 8px;'>
        Simulate injuries
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"{home_team.split()[-1]} adjustments"):
        home_ortg_adj = st.slider("Offensive rating shift", -15.0, 15.0, 0.0, 0.5,
                                   help="Lower if a key scorer is out")
        home_drtg_adj = st.slider("Defensive rating shift", -15.0, 15.0, 0.0, 0.5,
                                   help="Raise if a key defender is out")

    with st.expander(f"{away_team.split()[-1]} adjustments"):
        away_ortg_adj = st.slider("Offensive rating shift", -15.0, 15.0, 0.0, 0.5)
        away_drtg_adj = st.slider("Defensive rating shift", -15.0, 15.0, 0.0, 0.5)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    run = st.button("Run Prediction")

    st.markdown("""
    <div style='margin-top: 2rem; font-size: 0.7rem; color: #333; line-height: 1.6;'>
        Model: XGBoost + LightGBM ensemble<br>
        Data: Basketball-Reference 2010–2024<br>
        Features: 19 team-level statistics
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-title'>NBA Playoffs Predictor</div>
<div class='app-subtitle'>Machine learning · 2010–2024 historical data</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_gloss = st.tabs(["📊  Prediction", "📖  Stat Glossary"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    if not run:
        st.markdown("""
        <div class='info-box'>
            Select two teams in the sidebar and click <strong>Run Prediction</strong>
        </div>
        """, unsafe_allow_html=True)

    else:
        if home_team == away_team:
            st.error("Please select two different teams.")
            st.stop()

        home_stats = get_team_stats(home_team, season_year, df_feat)
        away_stats = get_team_stats(away_team, season_year, df_feat)

        if home_stats is None or away_stats is None:
            st.error("Could not find stats for one or both teams in that season. Try a different season.")
            st.stop()

        home_stats["ORTG"] += home_ortg_adj
        home_stats["DRTG"] += home_drtg_adj
        away_stats["ORTG"] += away_ortg_adj
        away_stats["DRTG"] += away_drtg_adj

        prob, _   = predict_matchup(home_stats, away_stats)
        shap_df   = explain_prediction(home_stats, away_stats)

        winner    = home_team if prob >= 0.5 else away_team
        conf      = max(prob, 1 - prob)
        conf_label = "High" if conf > 0.65 else "Moderate" if conf > 0.55 else "Toss-up"
        home_pct  = f"{prob:.0%}"
        away_pct  = f"{1-prob:.0%}"
        home_bar  = f"{prob*100:.1f}%"
        away_bar  = f"{(1-prob)*100:.1f}%"

        # Winner banner
        st.markdown(f"""
        <div class='winner-banner'>
            <div>
                <div class='winner-label'>Predicted winner</div>
                <div class='winner-name'>{winner}</div>
            </div>
            <div style='margin-left:auto; text-align:right;'>
                <div class='winner-label'>Confidence</div>
                <div style='font-family: Barlow Condensed, sans-serif; font-size: 1.4rem;
                            font-weight: 700; color: #ff6b35;'>{conf:.0%} &nbsp;
                    <span style='font-size:0.9rem; color:#666;'>{conf_label}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown(f"""
        <div class='prob-wrap'>
            <div class='prob-teams'>
                <div>
                    <span class='prob-team-name' style='color:#ff6b35;'>
                        {home_team.split()[-1]}
                    </span>
                    <span style='font-size:0.72rem; color:#555; margin-left:6px;'>HOME</span>
                </div>
                <div>
                    <span style='font-size:0.72rem; color:#555; margin-right:6px;'>AWAY</span>
                    <span class='prob-team-name' style='color:#3a8eff;'>
                        {away_team.split()[-1]}
                    </span>
                </div>
            </div>
            <div class='prob-bar-outer'>
                <div class='prob-bar-home' style='width:{home_bar};'></div>
                <div class='prob-bar-away' style='width:{away_bar};'></div>
            </div>
            <div class='prob-teams' style='margin-top:5px;'>
                <span style='font-family: Barlow Condensed, sans-serif; font-size:1.2rem;
                             font-weight:700; color:#ff6b35;'>{home_pct}</span>
                <span style='font-family: Barlow Condensed, sans-serif; font-size:1.2rem;
                             font-weight:700; color:#3a8eff;'>{away_pct}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metrics ──
        net_home = home_stats["ORTG"] - home_stats["DRTG"]
        net_away = away_stats["ORTG"] - away_stats["DRTG"]
        srs_diff = home_stats["SRS"] - away_stats["SRS"]

        st.markdown("<div class='section-head'>Key indicators</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-label'>Net Rating edge</div>
                <div class='metric-value {"accent" if net_home > net_away else ""}'>
                    {net_home - net_away:+.1f}
                </div>
                <div class='metric-sub'>
                    {home_team.split()[-1]} {net_home:+.1f} &nbsp;·&nbsp;
                    {away_team.split()[-1]} {net_away:+.1f}
                </div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>True Shooting edge</div>
                <div class='metric-value {"accent" if home_stats["TS"] > away_stats["TS"] else ""}'>
                    {(home_stats["TS"] - away_stats["TS"])*100:+.1f}%
                </div>
                <div class='metric-sub'>
                    {home_team.split()[-1]} {home_stats["TS"]*100:.1f}% &nbsp;·&nbsp;
                    {away_team.split()[-1]} {away_stats["TS"]*100:.1f}%
                </div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>SRS edge</div>
                <div class='metric-value {"accent" if srs_diff > 0 else ""}'>
                    {srs_diff:+.2f}
                </div>
                <div class='metric-sub'>
                    {home_team.split()[-1]} {home_stats["SRS"]:+.2f} &nbsp;·&nbsp;
                    {away_team.split()[-1]} {away_stats["SRS"]:+.2f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Head-to-head stat table ──
        st.markdown("<div class='section-head'>Head-to-head stats</div>", unsafe_allow_html=True)

        def fmt(val, pct=False, invert=False):
            """Format a stat value, bold the better side."""
            if pd.isna(val):
                return "—"
            if pct:
                return f"{val*100:.1f}%" if val < 2 else f"{val:.1f}%"
            return f"{val:.1f}"

        stats_rows = [
            ("Offensive Rating",  "ORTG", home_stats["ORTG"],  away_stats["ORTG"],  False, False),
            ("Defensive Rating",  "DRTG", home_stats["DRTG"],  away_stats["DRTG"],  False, True),
            ("Net Rating",        "NET",  net_home,            net_away,            False, False),
            ("True Shooting %",   "TS%",  home_stats["TS"]*100,away_stats["TS"]*100,False, False),
            ("Turnover Rate",     "TOV%", home_stats["TOV"],   away_stats["TOV"],   False, True),
            ("Off. Reb %",        "ORB%", home_stats["OREB"],  away_stats["OREB"],  False, False),
            ("SRS",               "SRS",  home_stats["SRS"],   away_stats["SRS"],   False, False),
        ]

        home_name_short = home_team.split()[-1]
        away_name_short = away_team.split()[-1]

        rows_html = ""
        for label, abbr, hval, aval, is_pct, lower_is_better in stats_rows:
            h_wins = (hval < aval) if lower_is_better else (hval > aval)
            h_class = "winner" if h_wins else "loser"
            a_class = "winner" if not h_wins else "loser"
            h_str = f"{hval:.1f}" if not is_pct else f"{hval:.1f}%"
            a_str = f"{aval:.1f}" if not is_pct else f"{aval:.1f}%"
            rows_html += f"""
            <tr>
                <td class='stat-name'>{label}</td>
                <td class='{h_class}'>{h_str}</td>
                <td class='{a_class}'>{a_str}</td>
            </tr>"""

        st.markdown(f"""
        <table class='stat-table'>
            <thead>
                <tr>
                    <th style='text-align:left;'>Stat</th>
                    <th style='color:#ff6b35;'>{home_name_short} (H)</th>
                    <th style='color:#3a8eff;'>{away_name_short} (A)</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── SHAP impact + H2H charts side by side ──
        st.markdown("<div class='section-head'>What's driving this prediction</div>",
                    unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            # Clean SHAP bar chart
            top = shap_df.head(10).copy()
            # Simplify feature names for display
            name_map = {
                "SRS_DIFF":      "Overall strength gap",
                "ORTG_DIFF":     "Offensive vs defensive edge",
                "DRTG_DIFF":     "Defensive vs offensive edge",
                "NET_RTG_DIFF":  "Net rating difference",
                "TS_DIFF":       "Shooting efficiency edge",
                "TOV_DIFF":      "Turnover advantage",
                "OREB_DIFF":     "Offensive rebounding edge",
                "HOME_SRS":      "Home team overall rating",
                "HOME_ORTG":     "Home offensive rating",
                "HOME_DRTG":     "Home defensive rating",
                "HOME_TS":       "Home true shooting %",
                "HOME_TOV":      "Home turnover rate",
                "HOME_OREB":     "Home off. rebound %",
                "AWAY_SRS":      "Away team overall rating",
                "AWAY_ORTG":     "Away offensive rating",
                "AWAY_DRTG":     "Away defensive rating",
                "AWAY_TS":       "Away true shooting %",
                "AWAY_TOV":      "Away turnover rate",
                "AWAY_OREB":     "Away off. rebound %",
            }
            top["label"] = top["feature"].map(name_map).fillna(top["feature"])
            top["color"] = top["shap_value"].apply(
                lambda v: "#ff6b35" if v > 0 else "#3a8eff")

            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            bars = ax.barh(top["label"][::-1], top["shap_value"][::-1],
                           color=top["color"][::-1], height=0.6)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlabel("Impact on home-team win probability",
                          fontsize=8, color="#555")
            ax.tick_params(colors="#666", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#1e1e1e")
            ax.xaxis.label.set_color("#555")
            # Legend
            p1 = mpatches.Patch(color="#ff6b35", label=f"Favours {home_name_short}")
            p2 = mpatches.Patch(color="#3a8eff", label=f"Favours {away_name_short}")
            ax.legend(handles=[p1, p2], fontsize=7, framealpha=0,
                      labelcolor="#888", loc="lower right")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig)
            plt.close(fig)

        with col_r:
            # H2H history
            fig_h2h = plot_h2h(
                df_feat,
                home_team.split()[-1],
                away_team.split()[-1],
            )
            # Restyle to match dark theme
            fig_h2h.patch.set_facecolor("#0d0d0d")
            for ax in fig_h2h.axes:
                ax.set_facecolor("#0d0d0d")
                ax.tick_params(colors="#666", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#1e1e1e")
                ax.xaxis.label.set_color("#555")
                ax.yaxis.label.set_color("#555")
                ax.title.set_color("#888")
                ax.title.set_fontsize(10)
            st.pyplot(fig_h2h)
            plt.close(fig_h2h)

        # ── Raw data expander ──
        with st.expander("Show raw model features"):
            display = shap_df[["feature", "raw_value", "shap_value"]].copy()
            display["feature"] = display["feature"].map(name_map).fillna(display["feature"])
            display.columns = ["Feature", "Value", "Model impact"]
            display["Value"] = display["Value"].round(3)
            display["Model impact"] = display["Model impact"].round(4)
            st.dataframe(display.reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GLOSSARY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gloss:
    st.markdown("""
    <div style='font-family: Barlow Condensed, sans-serif; font-size: 0.72rem;
                letter-spacing: 0.12em; text-transform: uppercase; color: #555;
                margin: 1rem 0 1.2rem;'>
        Every stat this model uses — explained in plain English
    </div>
    """, unsafe_allow_html=True)

    glossary = [
        {
            "term": "True Shooting %",
            "abbr": "TS%",
            "desc": "The most complete shooting efficiency stat. It accounts for 2-pointers, 3-pointers, and free throws all in one number. A team with a high TS% is getting good value from every shot attempt.",
            "example": "League average is typically around 56–57%. Elite offenses hit 59%+.",
            "tv": True,
        },
        {
            "term": "Offensive Rating",
            "abbr": "ORtg",
            "desc": "Points scored per 100 possessions. This removes pace from the equation — a fast team that scores a lot of points might look better than they are, but ORtg corrects for that.",
            "example": "A top-5 offense typically rates above 118. Below 110 is a concern.",
            "tv": True,
        },
        {
            "term": "Defensive Rating",
            "abbr": "DRtg",
            "desc": "Points allowed per 100 possessions. Lower is better. The best defenses in history have posted DRtg below 105. Elite playoff teams usually hold opponents well below league average.",
            "example": "League average is ~115. Elite defenses: 108 or lower.",
            "tv": True,
        },
        {
            "term": "Net Rating",
            "abbr": "NET",
            "desc": "Offensive Rating minus Defensive Rating. The single best predictor of team quality. A positive net rating means a team outscores opponents per 100 possessions when they're on the floor.",
            "example": "+8 or above is a championship-calibre pace. Negative = more losses than wins.",
            "tv": False,
        },
        {
            "term": "Simple Rating System",
            "abbr": "SRS",
            "desc": "A measure of team strength that accounts for margin of victory and strength of schedule. Zero is exactly league average. Positive means better than average, negative means worse.",
            "example": "Championship teams typically post SRS above +6. Lottery teams sit below -3.",
            "tv": False,
        },
        {
            "term": "Turnover Rate",
            "abbr": "TOV%",
            "desc": "Percentage of possessions that end in a turnover. Turnovers kill offenses — they're essentially free possessions handed to the other team. Lower is better for offense.",
            "example": "Good offenses keep this below 13%. Above 16% is a problem.",
            "tv": True,
        },
        {
            "term": "Offensive Rebound %",
            "abbr": "ORB%",
            "desc": "The percentage of available offensive rebounds a team grabs. Strong offensive rebounding gives teams second-chance points and wears down defenses over a long series.",
            "example": "Elite offensive rebounding teams: above 30%. Average is around 23%.",
            "tv": True,
        },
        {
            "term": "Net Rating Difference",
            "abbr": "NET DIFF",
            "desc": "The gap between the home team's net rating and the away team's net rating. This is one of the strongest single predictors of playoff game outcomes in this model.",
            "example": "A gap of +5 or more strongly favours the better-rated team.",
            "tv": False,
        },
        {
            "term": "SRS Difference",
            "abbr": "SRS DIFF",
            "desc": "Home team SRS minus away team SRS. Combines margin of victory and schedule strength into one matchup number. Captures season-long dominance in a single figure.",
            "example": "An SRS gap of 4+ points historically corresponds to ~65% win probability.",
            "tv": False,
        },
        {
            "term": "Offensive vs Defensive Edge",
            "abbr": "ORtg DIFF",
            "desc": "Compares the home team's offensive rating against the away team's defensive rating. A high number here means the home offense should thrive against this particular defense.",
            "example": "Home ORtg 120 vs Away DRtg 108 = a +12 offensive edge.",
            "tv": False,
        },
        {
            "term": "SHAP Value",
            "abbr": "SHAP",
            "desc": "Shapley Additive Explanations — a method from game theory that shows how much each stat pushed the model's prediction toward one team or the other. Positive = helped home team. Negative = helped away team.",
            "example": "If SRS DIFF has a SHAP of +0.15, that stat alone added 15% to the home team's predicted probability.",
            "tv": False,
        },
        {
            "term": "Confidence level",
            "abbr": "CONF",
            "desc": "How certain the model is. High (>65%) means the stats strongly favour one team. Moderate (55–65%) means it's competitive. Toss-up (<55%) means both teams are evenly matched — anything can happen.",
            "example": "Toss-ups are common in the NBA playoffs. Even the 'wrong' team wins often.",
            "tv": False,
        },
    ]

    # Separate TV stats from model-only stats
    tv_stats   = [g for g in glossary if g["tv"]]
    model_stats = [g for g in glossary if not g["tv"]]

    st.markdown("""
    <div class='section-head'>Stats you'll see on TV broadcasts</div>
    """, unsafe_allow_html=True)

    cards_html = "<div class='gloss-grid'>"
    for g in tv_stats:
        cards_html += f"""
        <div class='gloss-card'>
            <div class='gloss-term'>{g["term"]}</div>
            <div class='gloss-abbr'>{g["abbr"]}</div>
            <div class='gloss-desc'>{g["desc"]}</div>
            <div class='gloss-example'>📺 {g["example"]}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-head' style='margin-top:2.5rem;'>
        Advanced & model-specific stats
    </div>
    """, unsafe_allow_html=True)

    cards_html2 = "<div class='gloss-grid'>"
    for g in model_stats:
        cards_html2 += f"""
        <div class='gloss-card'>
            <div class='gloss-term'>{g["term"]}</div>
            <div class='gloss-abbr'>{g["abbr"]}</div>
            <div class='gloss-desc'>{g["desc"]}</div>
            <div class='gloss-example'>💡 {g["example"]}</div>
        </div>"""
    cards_html2 += "</div>"
    st.markdown(cards_html2, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top: 3rem; padding: 1rem 1.2rem;
                background: #0d0d0d; border-radius: 8px;
                border: 1px dashed #1e1e1e;'>
        <div style='font-family: Barlow Condensed, sans-serif; font-size: 0.7rem;
                    letter-spacing: 0.1em; text-transform: uppercase; color: #444;
                    margin-bottom: 6px;'>A note on predictions</div>
        <div style='font-size: 0.82rem; color: #555; line-height: 1.65;'>
            No model can perfectly predict NBA games — that's what makes them worth watching.
            This tool gives you a data-driven starting point based on 14 seasons of playoff history.
            Use the confidence level as a guide: high confidence means the stats are lopsided,
            not that the outcome is guaranteed. Upsets happen. Injuries change everything.
            The model cannot account for momentum, crowd energy, or a player having the
            game of their life.
        </div>
    </div>
    """, unsafe_allow_html=True)