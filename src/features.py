import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

RAW = Path("data/raw")
PROC = Path("data/processed")
MODELS = Path("data/models")
PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "SRS_DIFF", "ORTG_DIFF", "DRTG_DIFF",
    "NET_RTG_DIFF", "TS_DIFF", "TOV_DIFF", "OREB_DIFF",
    "HOME_SRS", "HOME_ORTG", "HOME_DRTG", "HOME_TS", "HOME_TOV", "HOME_OREB",
    "AWAY_SRS", "AWAY_ORTG", "AWAY_DRTG", "AWAY_TS", "AWAY_TOV", "AWAY_OREB",
]


def load_raw():
    games = pd.read_parquet(RAW / "playoff_games.parquet")
    bref = pd.read_parquet(RAW / "bref_advanced.parquet")
    return games, bref


def build_team_lookup(bref: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and clean the columns we need from the Basketball-Reference
    advanced stats table. Handles the mangled multi-level column headers
    that come out of pd.read_html on bref pages.
    """
    bref = bref.copy()
    bref.columns = [str(c).strip() for c in bref.columns]

    # Find each target column by searching for a keyword in the column name
    # Column names look like: 'Unnamed: 1_level_0_Team', 'Unnamed: 10_level_0_ORtg'
    def find_col(keywords):
        for col in bref.columns:
            for kw in keywords:
                if col.endswith(kw) or col == kw:
                    return col
        return None

    team_col  = find_col(["Team", "_Team"])
    srs_col   = find_col(["SRS", "_SRS"])
    ortg_col  = find_col(["ORtg", "_ORtg"])
    drtg_col  = find_col(["DRtg", "_DRtg"])
    pace_col  = find_col(["Pace", "_Pace"])
    ts_col    = find_col(["TS%", "_TS%"])
    tov_col   = find_col(["Offense Four Factors_TOV%"])
    oreb_col  = find_col(["Offense Four Factors_ORB%"])

    print(f"  Column mapping:")
    print(f"    Team  -> {team_col}")
    print(f"    SRS   -> {srs_col}")
    print(f"    ORtg  -> {ortg_col}")
    print(f"    DRtg  -> {drtg_col}")
    print(f"    TS%   -> {ts_col}")
    print(f"    TOV%  -> {tov_col}")
    print(f"    ORB%  -> {oreb_col}")

    if team_col is None:
        raise ValueError(f"Could not find Team column. Columns are:\n{bref.columns.tolist()}")

    rename = {team_col: "TEAM_NAME"}
    for src, dst in [
        (srs_col,  "SRS"),
        (ortg_col, "ORTG"),
        (drtg_col, "DRTG"),
        (pace_col, "PACE"),
        (ts_col,   "TS_PCT"),
        (tov_col,  "TOV_PCT"),
        (oreb_col, "OREB_PCT"),
    ]:
        if src is not None:
            rename[src] = dst

    bref = bref.rename(columns=rename)

    needed = ["TEAM_NAME", "SEASON_YEAR"]
    for col in ["SRS", "ORTG", "DRTG", "PACE", "TS_PCT", "TOV_PCT", "OREB_PCT"]:
        if col in bref.columns:
            needed.append(col)

    bref = bref[needed].copy()
    numeric = [c for c in needed if c not in ("TEAM_NAME", "SEASON_YEAR")]
    bref[numeric] = bref[numeric].apply(pd.to_numeric, errors="coerce")

    bref = bref[bref["TEAM_NAME"].notna()].copy()
    bref = bref[~bref["TEAM_NAME"].isin(["League Average", "Team", ""])].copy()
    bref = bref[~bref["TEAM_NAME"].str.match(r"^\d+$", na=False)].copy()

    return bref.reset_index(drop=True)


def clean_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the playoff games table scraped from Basketball-Reference.
    Expected columns: Date, Visitor/Neutral, PTS (visitor),
                      Home/Neutral, PTS (home).
    """
    games = games.copy()
    games.columns = [str(c).strip() for c in games.columns]

    # Rename columns to standard names regardless of bref column order
    col_aliases = {
        "Date": "DATE",
        "Visitor/Neutral": "AWAY_TEAM",
        "Home/Neutral": "HOME_TEAM",
    }
    for old, new in col_aliases.items():
        if old in games.columns:
            games.rename(columns={old: new}, inplace=True)

    # PTS columns appear twice with same name — rename by position
    pts_cols = [c for c in games.columns if "PTS" in str(c).upper() or
                games.columns.tolist().count(c) > 1]

    # Safer: find numeric columns that are likely scores
    games_renamed = games.copy()
    cols = games_renamed.columns.tolist()

    # bref layout: Date | Visitor | PTS | Home | PTS | ...
    if len(cols) >= 5:
        games_renamed.columns = [
            cols[i] if cols[i] not in ("", "Unnamed: 0")
            else f"COL_{i}" for i in range(len(cols))
        ]
        # Try to identify score columns by position
        try:
            games_renamed["AWAY_PTS"] = pd.to_numeric(
                games_renamed.iloc[:, 2], errors="coerce")
            games_renamed["HOME_PTS"] = pd.to_numeric(
                games_renamed.iloc[:, 4], errors="coerce")
        except Exception:
            pass

    # Drop header rows that bref repeats mid-table
    if "DATE" in games_renamed.columns:
        games_renamed = games_renamed[
            games_renamed["DATE"].notna() &
            (games_renamed["DATE"] != "Date")
        ].copy()

    games_renamed["DATE"] = pd.to_datetime(
        games_renamed.get("DATE", pd.Series()), errors="coerce")

    return games_renamed


def match_team_name(name: str, lookup: pd.DataFrame, year: int) -> dict:
    """
    Find a team's advanced stats by fuzzy-matching the last word of
    the team name (e.g. 'Lakers' from 'Los Angeles Lakers').
    """
    empty = {k: np.nan for k in
             ["SRS", "ORTG", "DRTG", "TS_PCT", "TOV_PCT", "OREB_PCT"]}
    if not isinstance(name, str) or pd.isna(name):
        return empty

    keyword = name.strip().split()[-1]
    subset = lookup[lookup["SEASON_YEAR"] == year]
    if subset.empty:
        subset = lookup

    match = subset[
        subset["TEAM_NAME"].str.contains(keyword, case=False, na=False)
    ]
    if match.empty:
        return empty

    row = match.iloc[-1]
    return {k: row.get(k, np.nan) for k in
            ["SRS", "ORTG", "DRTG", "TS_PCT", "TOV_PCT", "OREB_PCT"]}


def build_feature_rows(games: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Build one feature row per playoff game."""
    rows = []
    for _, g in games.iterrows():
        year = int(g.get("SEASON_YEAR", 0))
        home = str(g.get("HOME_TEAM", ""))
        away = str(g.get("AWAY_TEAM", ""))
        home_pts = pd.to_numeric(g.get("HOME_PTS", np.nan), errors="coerce")
        away_pts = pd.to_numeric(g.get("AWAY_PTS", np.nan), errors="coerce")

        if not home or not away or home == away:
            continue
        if pd.isna(home_pts) or pd.isna(away_pts):
            continue

        ha = match_team_name(home, lookup, year)
        aa = match_team_name(away, lookup, year)

        rows.append({
            "GAME_DATE": g.get("DATE", pd.NaT),
            "SEASON_YEAR": year,
            "HOME_TEAM": home,
            "AWAY_TEAM": away,
            "HOME_WIN": int(home_pts > away_pts),
            "HOME_PTS": home_pts,
            "AWAY_PTS": away_pts,
            "HOME_SRS": ha["SRS"],
            "HOME_ORTG": ha["ORTG"],
            "HOME_DRTG": ha["DRTG"],
            "HOME_TS": ha["TS_PCT"],
            "HOME_TOV": ha["TOV_PCT"],
            "HOME_OREB": ha["OREB_PCT"],
            "AWAY_SRS": aa["SRS"],
            "AWAY_ORTG": aa["ORTG"],
            "AWAY_DRTG": aa["DRTG"],
            "AWAY_TS": aa["TS_PCT"],
            "AWAY_TOV": aa["TOV_PCT"],
            "AWAY_OREB": aa["OREB_PCT"],
            "SRS_DIFF": ha["SRS"] - aa["SRS"],
            "ORTG_DIFF": ha["ORTG"] - aa["DRTG"],
            "DRTG_DIFF": aa["ORTG"] - ha["DRTG"],
        })

    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NET_RTG_HOME"] = df["HOME_ORTG"] - df["HOME_DRTG"]
    df["NET_RTG_AWAY"] = df["AWAY_ORTG"] - df["AWAY_DRTG"]
    df["NET_RTG_DIFF"] = df["NET_RTG_HOME"] - df["NET_RTG_AWAY"]
    df["TS_DIFF"] = df["HOME_TS"] - df["AWAY_TS"]
    df["TOV_DIFF"] = df["AWAY_TOV"] - df["HOME_TOV"]
    df["OREB_DIFF"] = df["HOME_OREB"] - df["AWAY_OREB"]
    df.sort_values("GAME_DATE", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scale_and_save(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X.fillna(X.median(), inplace=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)

    meta_cols = ["GAME_DATE", "SEASON_YEAR", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN"]
    df_out = pd.concat([
        df[meta_cols].reset_index(drop=True),
        X_scaled,
    ], axis=1)

    df_out.to_parquet(PROC / "features.parquet", index=False)
    joblib.dump(scaler, MODELS / "scaler.pkl")
    print(f"Feature matrix saved: {df_out.shape[0]} rows, {df_out.shape[1]} columns")
    print("Scaler saved to data/models/scaler.pkl")
    return df_out


if __name__ == "__main__":
    print("Loading raw data...")
    games, bref = load_raw()

    print("Building team lookup from Basketball-Reference advanced stats...")
    lookup = build_team_lookup(bref)
    print(f"  Lookup table: {lookup.shape[0]} team-season rows")
    print(f"  Columns: {lookup.columns.tolist()}")

    print("Cleaning game results...")
    games_clean = clean_games(games)
    print(f"  Games table: {games_clean.shape[0]} rows")

    print("Building feature rows...")
    df_feat = build_feature_rows(games_clean, lookup)
    print(f"  Feature rows: {len(df_feat)}")

    print("Engineering derived features...")
    df_feat = engineer_features(df_feat)

    print("Scaling and saving...")
    df_final = scale_and_save(df_feat)

    print("\nSample:")
    print(df_final[["HOME_TEAM", "AWAY_TEAM", "SEASON_YEAR", "HOME_WIN"]].head(10))