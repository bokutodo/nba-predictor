import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

MODELS = Path("data/models")

FEATURE_COLS = [
    "SRS_DIFF", "ORTG_DIFF", "DRTG_DIFF",
    "NET_RTG_DIFF", "TS_DIFF", "TOV_DIFF", "OREB_DIFF",
    "HOME_SRS", "HOME_ORTG", "HOME_DRTG", "HOME_TS", "HOME_TOV", "HOME_OREB",
    "AWAY_SRS", "AWAY_ORTG", "AWAY_DRTG", "AWAY_TS", "AWAY_TOV", "AWAY_OREB",
]

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODELS / "model.pkl")
        _scaler = joblib.load(MODELS / "scaler.pkl")


def get_team_stats(team_name: str, season_year: int, df_feat: pd.DataFrame) -> dict:
    keyword = team_name.strip().split()[-1]

    mask = (
        df_feat["HOME_TEAM"].str.contains(keyword, case=False, na=False) &
        (df_feat["SEASON_YEAR"] == season_year)
    )
    rows = df_feat[mask]

    if rows.empty:
        mask2 = df_feat["HOME_TEAM"].str.contains(keyword, case=False, na=False)
        rows = df_feat[mask2]

    if rows.empty:
        return None

    last = rows.sort_values("GAME_DATE").iloc[-1]
    return {
        "SRS":  float(last["HOME_SRS"]),
        "ORTG": float(last["HOME_ORTG"]),
        "DRTG": float(last["HOME_DRTG"]),
        "TS":   float(last["HOME_TS"]),
        "TOV":  float(last["HOME_TOV"]),
        "OREB": float(last["HOME_OREB"]),
    }


def build_feature_vector(home_stats: dict, away_stats: dict) -> np.ndarray:
    net_home = home_stats["ORTG"] - home_stats["DRTG"]
    net_away = away_stats["ORTG"] - away_stats["DRTG"]

    return np.array([
        home_stats["SRS"]  - away_stats["SRS"],    # SRS_DIFF
        home_stats["ORTG"] - away_stats["DRTG"],   # ORTG_DIFF
        away_stats["ORTG"] - home_stats["DRTG"],   # DRTG_DIFF
        net_home - net_away,                        # NET_RTG_DIFF
        home_stats["TS"]   - away_stats["TS"],     # TS_DIFF
        away_stats["TOV"]  - home_stats["TOV"],    # TOV_DIFF
        home_stats["OREB"] - away_stats["OREB"],   # OREB_DIFF
        home_stats["SRS"],                          # HOME_SRS
        home_stats["ORTG"],                         # HOME_ORTG
        home_stats["DRTG"],                         # HOME_DRTG
        home_stats["TS"],                           # HOME_TS
        home_stats["TOV"],                          # HOME_TOV
        home_stats["OREB"],                         # HOME_OREB
        away_stats["SRS"],                          # AWAY_SRS
        away_stats["ORTG"],                         # AWAY_ORTG
        away_stats["DRTG"],                         # AWAY_DRTG
        away_stats["TS"],                           # AWAY_TS
        away_stats["TOV"],                          # AWAY_TOV
        away_stats["OREB"],                         # AWAY_OREB
    ], dtype=float)


def predict_matchup(home_stats: dict, away_stats: dict):
    _load()
    raw_vec = build_feature_vector(home_stats, away_stats)
    scaled = _scaler.transform(raw_vec.reshape(1, -1))
    prob = float(_model.predict_proba(scaled)[0, 1])
    return prob, raw_vec


def explain_prediction(home_stats: dict, away_stats: dict) -> pd.DataFrame:
    _load()
    raw_vec = build_feature_vector(home_stats, away_stats)
    scaled = _scaler.transform(raw_vec.reshape(1, -1))

    xgb_model = _model.named_estimators_["xgb"]

    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(scaled)
    if isinstance(sv, list):
        sv = sv[1]

    return pd.DataFrame({
        "feature":    FEATURE_COLS,
        "shap_value": sv[0],
        "raw_value":  raw_vec,
    }).sort_values("shap_value", key=abs, ascending=False)