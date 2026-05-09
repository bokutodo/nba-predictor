import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def extract_table(html: str, table_id: str) -> pd.DataFrame:
    """
    Pull a specific table by ID from raw HTML.
    Basketball-Reference sometimes hides tables inside HTML comments
    so we strip comment markers before parsing.
    """
    # Remove HTML comment wrappers that bref uses to hide some tables
    html = html.replace("<!--", "").replace("-->", "")

    try:
        tables = pd.read_html(StringIO(html), attrs={"id": table_id})
        if tables:
            return tables[0]
    except Exception as e:
        print(f"    extract_table({table_id}) failed: {e}")
    return pd.DataFrame()


def scrape_advanced_stats(year: int) -> pd.DataFrame:
    """
    Scrape team advanced stats from the main season page.
    Target table IDs in order of preference.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
    print(f"  Advanced stats {year}...", end=" ")
    html = get_html(url)

    for table_id in ["advanced-team", "team-ratings", "misc_stats"]:
        df = extract_table(html, table_id)
        if df.empty:
            continue

        # Flatten multi-level column headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                f"{a}_{b}".strip("_") if b and b != a else a
                for a, b in df.columns
            ]

        df.columns = [str(c).strip() for c in df.columns]

        # Drop repeated header rows bref inserts mid-table
        first_col = df.columns[0]
        df = df[df[first_col] != first_col].copy()
        df = df[df[first_col].notna()].copy()
        df = df[~df[first_col].isin(["Team", "Rk", ""])].copy()

        df["SEASON_YEAR"] = year
        print(f"table='{table_id}' shape={df.shape}")
        return df

    print("no table found.")
    return pd.DataFrame()


def scrape_playoff_games(year: int) -> pd.DataFrame:
    """
    Scrape individual playoff game results from the playoff schedule page.
    Returns columns: SEASON_YEAR, DATE, AWAY_TEAM, AWAY_PTS, HOME_TEAM, HOME_PTS
    """
    url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_games.html"
    print(f"  Playoff games {year}...", end=" ")
    html = get_html(url)
    html = html.replace("<!--", "").replace("-->", "")

    try:
        all_tables = pd.read_html(StringIO(html))
    except Exception as e:
        print(f"no tables found: {e}")
        return pd.DataFrame()

    frames = []
    for df in all_tables:
        df.columns = [str(c).strip() for c in df.columns]

        # Identify the right table by checking for expected columns
        has_visitor = any("visitor" in c.lower() or "neutral" in c.lower()
                          for c in df.columns)
        has_home = any("home" in c.lower() for c in df.columns)

        if not (has_visitor and has_home):
            continue

        # Rename columns to standard names
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if "visitor" in cl or ("neutral" in cl and "visitor" in cl):
                col_map[c] = "AWAY_TEAM"
            elif "home" in cl and "team" not in cl and "pts" not in cl:
                col_map[c] = "HOME_TEAM"
            elif "date" in cl:
                col_map[c] = "DATE"

        df = df.rename(columns=col_map)

        # PTS columns: bref uses same header twice, grab by position
        cols = df.columns.tolist()
        pts_positions = [i for i, c in enumerate(cols)
                         if "pts" in str(c).lower() or c == "PTS"]

        if len(pts_positions) >= 2:
            df = df.copy()
            df.rename(columns={cols[pts_positions[0]]: "AWAY_PTS",
                                cols[pts_positions[1]]: "HOME_PTS"}, inplace=True)
        elif "AWAY_TEAM" in df.columns and "HOME_TEAM" in df.columns:
            # Fallback: find numeric columns by position
            numeric_cols = [c for c in df.columns
                            if df[c].apply(pd.to_numeric, errors="coerce").notna().sum() > 5]
            if len(numeric_cols) >= 2:
                df["AWAY_PTS"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
                df["HOME_PTS"] = pd.to_numeric(df[numeric_cols[1]], errors="coerce")

        # Keep only rows where both teams and scores are present
        needed = ["DATE", "AWAY_TEAM", "AWAY_PTS", "HOME_TEAM", "HOME_PTS"]
        if not all(c in df.columns for c in needed):
            continue

        df = df[needed].copy()
        df["AWAY_PTS"] = pd.to_numeric(df["AWAY_PTS"], errors="coerce")
        df["HOME_PTS"] = pd.to_numeric(df["HOME_PTS"], errors="coerce")
        df = df[df["AWAY_PTS"].notna() & df["HOME_PTS"].notna()].copy()
        df = df[df["DATE"] != "Date"].copy()  # drop repeated header rows
        df["SEASON_YEAR"] = year
        frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        print(f"shape={out.shape}")
        return out

    print("no valid game tables found.")
    return pd.DataFrame()


def fetch_all(start: int = 2010, end: int = 2024):
    all_ratings = []
    all_games = []

    for year in range(start, end + 1):
        print(f"\n── {year} ──")

        try:
            ratings = scrape_advanced_stats(year)
            if not ratings.empty:
                all_ratings.append(ratings)
        except Exception as e:
            print(f"  Ratings error: {e}")

        try:
            games = scrape_playoff_games(year)
            if not games.empty:
                all_games.append(games)
        except Exception as e:
            print(f"  Games error: {e}")

        time.sleep(4)

    if all_ratings:
        df = pd.concat(all_ratings, ignore_index=True)
        df.to_parquet(RAW / "bref_advanced.parquet", index=False)
        print(f"\n✓ bref_advanced.parquet  {df.shape}")
    else:
        print("\n✗ No ratings data — check scraper output above.")

    if all_games:
        df = pd.concat(all_games, ignore_index=True)
        df.to_parquet(RAW / "playoff_games.parquet", index=False)
        print(f"✓ playoff_games.parquet  {df.shape}")
    else:
        print("✗ No games data — check scraper output above.")


if __name__ == "__main__":
    print("Scraping Basketball-Reference (2010–2024)...")
    print("~3 minutes total due to rate-limit sleeps.\n")
    fetch_all(start=2010, end=2024)
    print("\nDone.")