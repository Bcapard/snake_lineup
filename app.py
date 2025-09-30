import json, base64, io
import os                         # NEW
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update, ctx
import pandas as pd
import numpy as np


# =========================================
# CONFIG & CONSTANTS
# =========================================
BASE = Path(__file__).resolve().parent
# CHANGED: Use DATA_DIR env var (Render) or fallback to ./data locally
DATA = Path(os.getenv("DATA_DIR", BASE / "data"))   # CHANGED
DATA.mkdir(parents=True, exist_ok=True)

PERSIST_PLAYERS = DATA / "players_db.json"
PERSIST_WEIGHTS = DATA / "weights_db.json"
EXPORT_LINEUPS  = DATA / "lineups_export.csv"
EXPORT_SEEDING  = DATA / "seeding_export.csv"
EXPORT_WIDE     = DATA / "lineups_wide_export.csv"

PLAYER_COLUMNS = [
    "player_id", "name", "jersey",
    "Scoring", "Defense", "BallHandling", "Height", "Hustle"
]
SKILL_COLS = ["Scoring", "Defense", "BallHandling", "Height", "Hustle"]

WEIGHTS_COLUMNS = ["Metric", "Weight"]
WEIGHTS_ALLOWED_METRICS = SKILL_COLS
WEIGHTS_TARGET_SUM = 100

NUM_PERIODS_DEFAULT = 8
PLAYERS_ON_COURT_DEFAULT = 5
MAX_ATTENDING = 12

# =========================================
# HELPERS: Shared
# =========================================
def _normalize_headers(cols):
    norm_map = {
        "player_id":"player_id", "id":"player_id", "pid":"player_id",
        "name":"name", "player_name":"name", "full_name":"name",
        "jersey":"jersey", "jersey_number":"jersey", "number":"jersey",
        "scoring":"Scoring", "defense":"Defense", "ballhandling":"BallHandling",
        "height":"Height", "hustle":"Hustle",
        "metric": "Metric", "weight": "Weight"
    }
    out = []
    for c in cols:
        key = str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        mapped = None
        for k, v in norm_map.items():
            if key == k.replace("_",""):
                mapped = v
                break
        out.append(mapped if mapped else str(c))
    return out

def _upload_to_df(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(decoded))
    elif filename.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(decoded))
    else:
        raise ValueError("Unsupported file format. Please upload CSV or XLSX.")

# =========================================
# PLAYERS
# =========================================
def players_load():
    if PERSIST_PLAYERS.exists():
        with open(PERSIST_PLAYERS, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None

def players_save(df: pd.DataFrame):
    records = df[PLAYER_COLUMNS].replace({np.nan: None}).to_dict(orient="records")
    PERSIST_PLAYERS.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSIST_PLAYERS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def players_parse_uploaded(contents, filename):
    df = _upload_to_df(contents, filename)
    df.columns = _normalize_headers(df.columns)
    for col in PLAYER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[[*PLAYER_COLUMNS]]
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()
    for c in SKILL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def players_validate(df: pd.DataFrame):
    issues = []
    missing = [c for c in PLAYER_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    if df["player_id"].isna().any():
        issues.append("Some player_id values are missing or invalid.")
    if df["name"].str.strip().eq("").any():
        issues.append("Some names are empty.")
    if df["jersey"].isna().any():
        issues.append("Some jersey numbers are missing or invalid.")
    dup_pid = df["player_id"][df["player_id"].duplicated(keep=False)]
    if dup_pid.notna().any():
        issues.append("Duplicate player_id detected.")
    dup_j = df["jersey"][df["jersey"].duplicated(keep=False)]
    if dup_j.notna().any():
        issues.append("Duplicate jersey numbers detected.")
    for c in SKILL_COLS:
        bad = df[c].isna() | ~df[c].between(1, 5)
        if bad.any():
            issues.append(f"{c}: {int(bad.sum())} value(s) missing or out of 1–5 range.")
    return issues

def players_empty_table():
    row = {c: None for c in PLAYER_COLUMNS}
    row["name"] = ""
    return pd.DataFrame([row])

# =========================================
# WEIGHTS
# =========================================
def weights_load():
    if PERSIST_WEIGHTS.exists():
        with open(PERSIST_WEIGHTS, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None

def weights_save(df: pd.DataFrame):
    records = df.replace({np.nan: None}).to_dict(orient="records")
    PERSIST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PERSIST_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def weights_default():
    per = WEIGHTS_TARGET_SUM / len(WEIGHTS_ALLOWED_METRICS)
    return pd.DataFrame({"Metric": WEIGHTS_ALLOWED_METRICS, "Weight": [per]*len(WEIGHTS_ALLOWED_METRICS)})

def weights_empty_table():
    return weights_default().copy()

def weights_parse_uploaded(contents, filename):
    df = _upload_to_df(contents, filename)
    df.columns = _normalize_headers(df.columns)
    cols_lower = [c.lower() for c in df.columns]
    tidy = ("metric" in cols_lower) and ("weight" in cols_lower)
    if tidy:
        df = df[[col for col in df.columns if col in ["Metric", "Weight"]]]
        df["Metric"] = df["Metric"].astype(str).str.strip()
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    else:
        keep = [c for c in df.columns if c in WEIGHTS_ALLOWED_METRICS]
        if not keep:
            df = weights_default()
        else:
            melted = df[keep].iloc[:1].melt(var_name="Metric", value_name="Weight")
            melted["Weight"] = pd.to_numeric(melted["Weight"], errors="coerce")
            df = melted
    return df

def weights_validate(df: pd.DataFrame):
    issues = []
    missing = [c for c in WEIGHTS_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    df["Metric"] = df["Metric"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    bad_metric = ~df["Metric"].isin(WEIGHTS_ALLOWED_METRICS)
    if bad_metric.any():
        bad_list = df.loc[bad_metric, "Metric"].unique().tolist()
        issues.append(f"Unknown metrics: {bad_list}. Allowed: {WEIGHTS_ALLOWED_METRICS}")
    dup_m = df["Metric"][df["Metric"].duplicated(keep=False)]
    if dup_m.any():
        issues.append("Duplicate Metric rows detected.")
    if df["Weight"].isna().any():
        issues.append("Some Weight values are missing or invalid.")
    if (df["Weight"] < 0).any():
        issues.append("Negative Weight values are not allowed.")
    wsum = float(df["Weight"].sum()) if df["Weight"].notna().all() else np.nan
    if not np.isnan(wsum) and abs(wsum - WEIGHTS_TARGET_SUM) > 1e-6:
        issues.append(f"Weights must sum to {WEIGHTS_TARGET_SUM}. Current sum = {wsum:g}.")
    return issues

# =========================================
# SNAKE GENERATOR
# =========================================
def compute_composites(players_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    wmap = {r["Metric"]: float(r["Weight"]) for _, r in weights_df.iterrows()}
    wsum = sum(wmap.get(k, 0.0) for k in SKILL_COLS)
    if wsum <= 0:
        wmap = {k: 1.0 for k in SKILL_COLS}
        wsum = float(len(SKILL_COLS))
    norm_w = {k: wmap.get(k, 0.0) / wsum for k in SKILL_COLS}
    comp = np.zeros(len(players_df), dtype=float)
    for k in SKILL_COLS:
        comp += players_df[k].fillna(0).astype(float).values * norm_w[k]
    out = players_df.copy()
    out["composite"] = np.round(comp, 4)
    return out

def build_u10_snake_seeding(comp_df: pd.DataFrame, k_on: int):
    df = comp_df.sort_values(["composite", "player_id"], ascending=[False, True]).reset_index(drop=True)
    df["initial_rank"] = np.arange(1, len(df)+1, dtype=int)
    df["chunk"] = ((df["initial_rank"] - 1) // k_on) + 1
    df["position_in_chunk"] = ((df["initial_rank"] - 1) % k_on) + 1
    chunk_sizes = df.groupby("chunk")["player_id"].size().to_dict()
    df["chunk_size"] = df["chunk"].map(chunk_sizes)
    df["position_for_sort"] = np.where(
        (df["chunk"] % 2) == 1,
        df["position_in_chunk"],
        df["chunk_size"] - df["position_in_chunk"] + 1
    )
    df = df.sort_values(["chunk", "position_for_sort", "player_id"], ascending=[True, True, True]).reset_index(drop=True)
    df["seed_order"] = np.arange(1, len(df)+1, dtype=int)
    return df

def _row(rec, period, pos):
    return {
        "period": period,
        "pos": pos,
        "player_id": int(rec["player_id"]),
        "name":  str(rec["name"]),
        "jersey": int(rec["jersey"]) if not pd.isna(rec["jersey"]) else None,
        "seed_order": int(rec["seed_order"]),
        "chunk": int(rec["chunk"]),
        "position_in_chunk": int(rec["position_in_chunk"]),
        "composite": float(rec["composite"]),
    }

def build_full_schedule(comp_df: pd.DataFrame, periods: int, k_on: int):
    seeded = build_u10_snake_seeding(comp_df, k_on)
    order = seeded.sort_values("seed_order").reset_index(drop=True)
    N = len(order)

    rows = []
    for p in range(1, periods + 1):
        start = ((p - 1) * k_on) % N
        indices = [(start + i) % N for i in range(k_on)]
        picks = order.iloc[indices].to_dict("records")
        for i, rec in enumerate(picks, start=1):
            rows.append(_row(rec, p, i))
    schedule_df = pd.DataFrame(rows)
    return seeded, schedule_df

def schedule_to_wide(schedule_df: pd.DataFrame, seeded: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()
    ordered_names = seeded.sort_values("seed_order")["name"].tolist()
    periods = sorted(schedule_df["period"].unique())
    wide = pd.DataFrame({"period": periods})
    for name in ordered_names:
        wide[name] = ""
    for p in periods:
        subset = schedule_df[schedule_df["period"] == p]
        for _, r in subset.iterrows():
            wide.loc[wide["period"] == p, r["name"]] = str(int(r["pos"]))
    return wide

def schedule_to_names(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["period", "players"])
    rows = []
    for p in sorted(schedule_df["period"].unique()):
        names = (schedule_df[schedule_df["period"] == p]
                 .sort_values("pos")["name"].tolist())
        rows.append({"period": p, "players": ", ".join(names)})
    return pd.DataFrame(rows)

# =========================================
# APP (Tabs 1–3)
# =========================================
app = Dash(__name__, title="U10 Lineup", suppress_callback_exceptions=True)
server = app.server   # NEW: explicit Flask server handle for gunicorn

# PWA index
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <meta name="apple-mobile-web-app-capable" content="yes"/>
    <meta name="apple-mobile-web-app-status-bar-style" content="default"/>
    <meta name="theme-color" content="#0b5ed7"/>
    <link rel="manifest" href="/assets/manifest.json"/>
    <link rel="apple-touch-icon" href="/assets/icon-192.png"/>
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      <script>
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.register('/assets/sw.js');
        }
      </script>
      {%renderer%}
    </footer>
  </body>
</html>
"""

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
    children=[
        html.H2("Team Lineup — Data Manager"),
        dcc.Tabs(
            id="tabs",
            value="tab-players",
            children=[
                # [tabs unchanged — omitted here for brevity in this snippet]
            ]
        )
    ]
)

# [ALL your callbacks remain exactly as you pasted them — no changes needed]


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    PERSIST_PLAYERS.parent.mkdir(parents=True, exist_ok=True)
    PERSIST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print("👉 Open http://127.0.0.1:8052")
    import threading, webbrowser, time
    url = "http://127.0.0.1:8052"
    def open_browser():
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=8052)
