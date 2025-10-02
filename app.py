import os
import json, base64, io
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, no_update, ctx
import pandas as pd
import numpy as np

# =========================================
# CONFIG & CONSTANTS
# =========================================
# Allow overriding the data dir in hosted environments; default to ./data locally
BASE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_DIR", BASE / "data"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)

PERSIST_PLAYERS = DATA_ROOT / "players_db.json"
PERSIST_WEIGHTS = DATA_ROOT / "weights_db.json"
EXPORT_LINEUPS  = DATA_ROOT / "lineups_export.csv"
EXPORT_SEEDING  = DATA_ROOT / "seeding_export.csv"
EXPORT_WIDE     = DATA_ROOT / "lineups_wide_export.csv"

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

# If you ever want to also persist to server files, set env USE_SERVER_PERSIST=1
USE_SERVER_PERSIST = os.getenv("USE_SERVER_PERSIST", "0") == "1"

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
    """Server-side save (optional). Disabled unless USE_SERVER_PERSIST=1."""
    if not USE_SERVER_PERSIST:
        return
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
    """Server-side save (optional). Disabled unless USE_SERVER_PERSIST=1."""
    if not USE_SERVER_PERSIST:
        return
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
        "pos": pos,  # 1..k_on
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
server = app.server  # <-- WSGI entrypoint for gunicorn

# PWA hooks (make sure /assets/manifest.json and /assets/sw.js exist)
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
        # Header row with icon + title
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "marginBottom": "12px",
            },
            children=[
                html.Img(
                    src="/assets/app-icon.png?v=1",
                    alt="App icon",
                    draggable="false",
                    style={"width": "36px", "height": "36px", "objectFit": "contain"},
                ),
                html.H2("Snake Lineup Generator", style={"margin": 0}),
            ],
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-players",
            className="u10-tabs",
            children=[
                # ---------- TAB 1 ----------
                dcc.Tab(label="1) Players — Upload / Edit / Save", value="tab-players", className="u10-tabs", children=[
                    dcc.Store(id="players-store", storage_type="local"),
                    dcc.Store(id="players-pending-upload"),
                    html.Div(style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Step 1 — Upload Players (CSV/XLSX)"),
                        html.P("If already saved, you can skip this step and click 'Load Saved'."),
                        dcc.Upload(
                            id="players-uploader",
                            children=html.Div(["Drag & Drop or ", html.U("Select a file")]),
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "8px", "textAlign": "center"
                            },
                            multiple=False,
                        ),
                        html.Div(id="players-upload-filename", style={"marginTop": "6px", "fontStyle": "italic"}),
                    ]),
                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Step 2 — Edit Players"),
                        dash_table.DataTable(
                            id="players-table",
                            data=[],
                            columns=[
                                {"name": "player_id", "id": "player_id", "type": "numeric"},
                                {"name": "name", "id": "name", "type": "text"},
                                {"name": "jersey", "id": "jersey", "type": "numeric"},
                                {"name": "Scoring", "id": "Scoring", "type": "numeric"},
                                {"name": "Defense", "id": "Defense", "type": "numeric"},
                                {"name": "BallHandling", "id": "BallHandling", "type": "numeric"},
                                {"name": "Height", "id": "Height", "type": "numeric"},
                                {"name": "Hustle", "id": "Hustle", "type": "numeric"},
                            ],
                            editable=True,
                            row_deletable=True,
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 110, "maxWidth": 180, "whiteSpace": "normal"},
                        ),
                        html.Div(style={"marginTop": "10px", "display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                            html.Button("Add Row", id="players-add-row", n_clicks=0),
                            html.Button("Save (Persist)", id="players-save", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                            html.Button("Load Saved", id="players-load-saved", n_clicks=0),
                        ]),
                        html.Div(id="players-validation", style={"marginTop": "12px", "color": "#b00020"}),
                        html.Div(id="players-save-status", style={"marginTop": "6px", "color": "#088a2a"}),
                    ]),
                ]),
                # ---------- TAB 2 ----------
                dcc.Tab(label="2) Weights — Upload / Edit / Save", value="tab-weights", className="u10-tabs", children=[
                    dcc.Store(id="weights-store", storage_type="local"),
                    html.Div(style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Weights — Upload (CSV/XLSX)"),
                        html.P(f"Expected tidy format: columns {WEIGHTS_COLUMNS} — Metric in {WEIGHTS_ALLOWED_METRICS}, weights sum to {WEIGHTS_TARGET_SUM}."),
                        dcc.Upload(
                            id="weights-uploader",
                            children=html.Div(["Drag & Drop or ", html.U("Select a file")]),
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "8px", "textAlign": "center"
                            },
                            multiple=False,
                        ),
                        html.Div(id="weights-upload-filename", style={"marginTop": "6px", "fontStyle": "italic"}),
                    ]),
                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Edit Weights"),
                        dash_table.DataTable(
                            id="weights-table",
                            data=[],
                            columns=[
                                {"name": "Metric", "id": "Metric", "type": "text", "presentation": "dropdown"},
                                {"name": f"Weight (sum to {WEIGHTS_TARGET_SUM})", "id": "Weight", "type": "numeric"},
                            ],
                            editable=True,
                            row_deletable=False,
                            dropdown={"Metric": {"options": [{"label": m, "value": m} for m in WEIGHTS_ALLOWED_METRICS]}},
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 120, "maxWidth": 220, "whiteSpace": "normal"},
                        ),
                        html.Div(style={"marginTop": "10px", "display": "flex", "gap": "8px", "flexWrap": "wrap"}, children=[
                            html.Button("Reset to Even", id="weights-reset", n_clicks=0),
                            html.Button("Save (Persist)", id="weights-save", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                            html.Button("Load Saved", id="weights-load-saved", n_clicks=0),
                        ]),
                        html.Div(id="weights-msg", style={"marginTop": "10px", "color": "#088a2a"}),
                        html.Div(id="weights-err", style={"marginTop": "6px", "color": "#b00020"}),
                        html.Div(id="weights-sum", style={"marginTop": "6px", "fontStyle": "italic"}),
                    ]),
                ]),
                # ---------- TAB 3 ----------
                dcc.Tab(label="3) Lineup — Snake Generator", value="tab-snake", className="u10-tabs", children=[
                    # stores for the mobile picker
                    dcc.Store(id="snake-attending-prev", data=[]),
                    dcc.Store(id="snake-attending-all", data=[]),
                    dcc.Store(id="snake-seed-store"),
                    dcc.Store(id="snake-lineups-store"),
                    dcc.Store(id="snake-lineups-wide-store"),

                    html.Div(style={"marginTop": "12px", "padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Inputs"),
                        html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
                            html.Div(children=[html.Label("Players on court (per period)"), dcc.Input(id="snake-k", type="number", min=1, max=10, step=1, value=PLAYERS_ON_COURT_DEFAULT)]),
                            html.Div(children=[html.Label("Number of periods"), dcc.Input(id="snake-periods", type="number", min=1, max=16, step=1, value=NUM_PERIODS_DEFAULT)]),
                        ]),
                        html.Br(),

                        # --- New Mobile-Friendly Attending Picker ---
                        html.Label(f"Select attending players (max {MAX_ATTENDING})", style={"marginBottom": "10px"}),
                        html.Div(style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "alignItems": "center"}, children=[
                            dcc.Input(
                                id="snake-search",
                                type="text",
                                placeholder="Search name or #...",
                                debounce=True,
                                style={"flex": "1 1 220px", "height": "40px", "padding": "0 10px", "borderRadius": "8px", "border": "1px solid #999", "display": "none"}
                            ),
                            html.Button("Select All", id="snake-select-all", n_clicks=0),
                            html.Button("Clear", id="snake-clear", n_clicks=0),
                            html.Button("Invert", id="snake-invert", n_clicks=0, style={"display": "none"}),
                            html.Div(id="snake-count", style={"marginLeft": "auto", "fontWeight": 600}),
                        ]),

                        # Checklist with big tap targets
                        dcc.Checklist(
                            id="snake-attending-list",
                            options=[],
                            value=[],
                            labelStyle={
                                "display": "inline-flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "padding": "10px 12px",
                                "margin": "6px",
                                "border": "1px solid #999",
                                "borderRadius": "10px",
                                "minWidth": "140px",
                                "userSelect": "none",
                                "fontSize": "16px"
                            },
                            inputStyle={"width": "20px", "height": "20px"},
                            style={"marginTop": "10px", "display": "flex", "flexWrap": "wrap"}
                        ),

                        html.Div(id="snake-picker-err", style={"marginTop": "8px", "color": "#b00020"}),

                        html.Br(),
                        html.Button("Generate Lineups", id="snake-generate", n_clicks=0, style={"background": "#0E2B5C", "color": "white"}),
                        html.Button("Export CSV", id="snake-export", n_clicks=0, style={"marginLeft": "8px", "display": "none"}),
                        html.Div(id="snake-err", style={"marginTop": "10px", "color": "#b00020"}),
                        html.Div(id="snake-msg", style={"marginTop": "6px", "color": "#088a2a"}),
                    ]),

                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Lineups by Period (wide)"),
                        dash_table.DataTable(
                            id="snake-lineups-wide",
                            data=[],
                            columns=[{"name": "period", "id": "period"}],
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 80, "whiteSpace": "normal"},
                        ),
                    ]),
                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px"}, children=[
                        html.H4("Lineups by Period (names)"),
                        dash_table.DataTable(
                            id="snake-lineups-names",
                            data=[],
                            columns=[{"name": "period", "id": "period"}, {"name": "players", "id": "players"}],
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 120, "whiteSpace": "normal"},
                        ),
                    ]),
                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px", "display": "none"}, children=[
                        html.H4("Players — composites & seeding"),
                        dash_table.DataTable(
                            id="snake-seeding-table",
                            data=[],
                            columns=[
                                {"name": "seed_order", "id": "seed_order", "type": "numeric"},
                                {"name": "player_id", "id": "player_id", "type": "numeric"},
                                {"name": "name", "id": "name", "type": "text"},
                                {"name": "jersey", "id": "jersey", "type": "numeric"},
                                {"name": "composite", "id": "composite", "type": "numeric"},
                                {"name": "chunk", "id": "chunk", "type": "numeric"},
                                {"name": "position_in_chunk", "id": "position_in_chunk", "type": "numeric"},
                                {"name": "chunk_size", "id": "chunk_size", "type": "numeric"},
                                {"name": "position_for_sort", "id": "position_for_sort", "type": "numeric"},
                                {"name": "initial_rank", "id": "initial_rank", "type": "numeric"},
                            ],
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 90, "maxWidth": 150, "whiteSpace": "normal"},
                        ),
                    ]),
                    html.Div(style={"height": "14px"}),
                    html.Div(style={"padding": "16px", "border": "1px solid #333", "borderRadius": "12px", "display": "none"}, children=[
                        html.H4("Lineups by Period (long)"),
                        dash_table.DataTable(
                            id="snake-lineups-table",
                            data=[],
                            columns=[
                                {"name": "period", "id": "period", "type": "numeric"},
                                {"name": "pos", "id": "pos", "type": "numeric"},
                                {"name": "player_id", "id": "player_id", "type": "numeric"},
                                {"name": "name", "id": "name", "type": "text"},
                                {"name": "jersey", "id": "jersey", "type": "numeric"},
                                {"name": "seed_order", "id": "seed_order", "type": "numeric"},
                                {"name": "chunk", "id": "chunk", "type": "numeric"},
                                {"name": "position_in_chunk", "id": "position_in_chunk", "type": "numeric"},
                                {"name": "composite", "id": "composite", "type": "numeric"},
                            ],
                            page_size=40,
                            style_table={"overflowX": "auto"},
                            style_cell={"minWidth": 90, "maxWidth": 140, "whiteSpace": "normal"},
                        ),
                    ]),
                ]),
            ]
        )
    ]
)

# ---------- TAB 1 CALLBACKS ----------
@app.callback(
    Output("players-upload-filename", "children"),
    Output("players-pending-upload", "data"),
    Input("players-uploader", "contents"),
    State("players-uploader", "filename"),
    prevent_initial_call=True
)
def players_handle_upload(contents, filename):
    if contents is None:
        return no_update, no_update
    try:
        df = players_parse_uploaded(contents, filename)
        return f"Uploaded: {filename}", df.replace({np.nan: None}).to_dict(orient="records")
    except Exception as e:
        return f"Upload error: {e}", None

@app.callback(
    Output("players-table", "data"),
    Output("players-store", "data", allow_duplicate=True),
    Input("players-store", "data"),
    Input("players-pending-upload", "data"),
    Input("players-load-saved", "n_clicks"),
    prevent_initial_call="initial_duplicate",
)
def players_seed_table(store_data, pending, n_load):
    trig = ctx.triggered_id
    if trig == "players-pending-upload" and pending:
        return pending, pending
    if trig == "players-load-saved" and n_load:
        if store_data:
            return store_data, store_data
        df = players_load()
        if df is not None:
            data = df.replace({np.nan: None}).to_dict(orient="records")
            return data, data
        data = players_empty_table().replace({np.nan: None}).to_dict(orient="records")
        return data, data
    if store_data:
        return store_data, store_data
    df0 = players_load()
    if df0 is not None:
        data = df0.replace({np.nan: None}).to_dict(orient="records")
        return data, data
    data = players_empty_table().replace({np.nan: None}).to_dict(orient="records")
    return data, data

@app.callback(
    Output("players-table", "data", allow_duplicate=True),
    Input("players-add-row", "n_clicks"),
    State("players-table", "data"),
    prevent_initial_call=True
)
def players_add_row(n, rows):
    if not n: return no_update
    new = {c: None for c in PLAYER_COLUMNS}
    new["name"] = ""
    return rows + [new]

@app.callback(
    Output("players-validation", "children"),
    Output("players-save-status", "children"),
    Output("players-store", "data", allow_duplicate=True),
    Input("players-save", "n_clicks"),
    State("players-table", "data"),
    prevent_initial_call=True
)
def players_save_cb(n, rows):
    if not n:
        return no_update, no_update, no_update
    df = pd.DataFrame(rows)
    for col in PLAYER_COLUMNS:
        if col not in df.columns: df[col] = np.nan
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").astype("Int64")
    df["name"] = df["name"].astype(str).str.strip()
    for c in SKILL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    issues = players_validate(df)
    if issues:
        return [html.Ul([html.Li(i) for i in issues])], "", no_update
    players_save(df)
    clean_records = df.replace({np.nan: None}).to_dict(orient="records")
    return "", "Saved! (This browser)", clean_records

# ---------- TAB 2 CALLBACKS ----------
@app.callback(
    Output("weights-store", "data"),
    Output("weights-upload-filename", "children"),
    Output("weights-msg", "children"),
    Output("weights-err", "children"),
    Output("weights-sum", "children"),
    Output("weights-table", "data"),
    Input("weights-store", "data"),
    Input("weights-uploader", "contents"),
    Input("weights-save", "n_clicks"),
    Input("weights-load-saved", "n_clicks"),
    Input("weights-reset", "n_clicks"),
    State("weights-uploader", "filename"),
    State("weights-table", "data"),
    prevent_initial_call=False,
)
def weights_master_cb(store_data, contents, n_save, n_load, n_reset, filename, table_rows):
    trig = ctx.triggered_id
    cur_df = (pd.DataFrame(store_data) if store_data is not None and store_data != []
              else (weights_load() if weights_load() is not None else weights_empty_table()))
    out_file = no_update
    out_msg = ""
    out_err = ""
    out_sum = ""
    out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")

    try:
        if trig == "weights-uploader" and contents:
            df_up = weights_parse_uploaded(contents, filename)
            issues = weights_validate(df_up)
            if issues:
                out_err = html.Ul([html.Li(i) for i in issues]); out_msg = ""
            else:
                cur_df = df_up
                out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
                out_msg = "Weights uploaded."; out_err = ""
            out_file = f"Uploaded: {filename}"
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-save" and n_save:
            df_cur = pd.DataFrame(table_rows) if table_rows else cur_df
            issues = weights_validate(df_cur)
            if issues:
                out_err = html.Ul([html.Li(i) for i in issues]); out_msg = ""
            else:
                cur_df = df_cur
                out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
                weights_save(cur_df)
                out_msg = "Saved! (This browser)"; out_err = ""
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-load-saved" and n_load:
            if store_data:
                cur_df = pd.DataFrame(store_data)
                out_msg = "Loaded weights from this browser."
            else:
                df_s = weights_load()
                if df_s is None:
                    cur_df = weights_empty_table(); out_msg = "No server copy. Loaded defaults."
                else:
                    cur_df = df_s; out_msg = "Loaded server copy."
            out_err = ""
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        elif trig == "weights-reset" and n_reset:
            cur_df = weights_empty_table()
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_msg = "Reset to even weights."; out_err = ""
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

        else:
            if store_data:
                cur_df = pd.DataFrame(store_data); out_msg = "Loaded weights from this browser."
            else:
                df0 = weights_load()
                cur_df = df0 if df0 is not None else weights_empty_table()
                out_msg = "Loaded server copy." if df0 is not None else "Initialized with default weights."
            out_err = ""
            out_table = cur_df.replace({np.nan: None}).to_dict(orient="records")
            out_sum = f"Current sum: {float(cur_df['Weight'].sum()):g} / {WEIGHTS_TARGET_SUM}"

    except Exception as e:
        out_err = f"Error: {e}"

    out_store = cur_df.replace({np.nan: None}).to_dict(orient="records")
    return out_store, out_file, out_msg, out_err, out_sum, out_table

# ---------- TAB 3: NEW MOBILE PICKER CALLBACKS ----------
@app.callback(
    Output("snake-attending-list", "options"),
    Output("snake-attending-list", "value"),
    Output("snake-attending-prev", "data"),
    Output("snake-attending-all", "data"),
    Output("snake-count", "children"),
    Input("tabs", "value"),
    Input("players-store", "data"),
    State("snake-attending-list", "value"),
    prevent_initial_call=False
)
def snake_seed_attending(tab, store_players, current_val):
    if tab != "tab-snake":
        return no_update, no_update, no_update, no_update, no_update

    # Prefer local browser copy; fallback to server file
    if store_players:
        players_df = pd.DataFrame(store_players)
    else:
        players_df = players_load()
        if players_df is None or players_df.empty:
            return [], [], [], [], "0 selected"

    if players_df is None or players_df.empty:
        return [], [], [], [], "0 selected"

    # Build options as "checkbox chips"
    def label_for(r):
        j = f"#{int(r['jersey'])}" if not pd.isna(r["jersey"]) else ""
        return f"{r['name']} {j}".strip()

    options = [{"label": label_for(r), "value": int(r["player_id"])} for _, r in players_df.iterrows()]
    all_ids = [int(r["player_id"]) for _, r in players_df.iterrows()]

    # Default selection = all up to cap
    default_vals = all_ids[:MAX_ATTENDING]
    val = current_val if current_val else default_vals
    val = val[:MAX_ATTENDING]

    count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
    return options, val, val, all_ids, count_text

@app.callback(
    Output("snake-attending-list", "options", allow_duplicate=True),
    Output("snake-attending-list", "value", allow_duplicate=True),
    Output("snake-attending-prev", "data", allow_duplicate=True),
    Output("snake-picker-err", "children"),
    Output("snake-count", "children", allow_duplicate=True),
    Input("snake-search", "value"),
    Input("snake-select-all", "n_clicks"),
    Input("snake-clear", "n_clicks"),
    Input("snake-invert", "n_clicks"),
    Input("snake-attending-list", "value"),
    State("snake-attending-list", "options"),
    State("snake-attending-prev", "data"),
    State("snake-attending-all", "data"),
    prevent_initial_call=True
)
def snake_picker(search, n_all, n_clear, n_invert, new_val, options, prev_val, all_ids):
    trig = ctx.triggered_id
    options = options or []
    prev_val = prev_val or []
    all_ids = all_ids or []

    # Helper to rebuild options after search
    def filter_options(q):
        if not q:
            return options
        q = str(q).strip().lower()
        def match(lbl):
            return (q in str(lbl).lower()) or q.lstrip("#") in str(lbl).lower()
        return [{"label": o["label"], "value": o["value"]} for o in options if match(o["label"])]

    err = ""
    count_text = f"{len(new_val or [])} selected (max {MAX_ATTENDING})"

    if trig == "snake-select-all" and n_all:
        filtered_opts = filter_options(search)
        filtered_ids = [o["value"] for o in filtered_opts]
        val = filtered_ids[:MAX_ATTENDING]
        count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
        return filtered_opts, val, val, err, count_text

    if trig == "snake-clear" and n_clear:
        filtered_opts = filter_options(search)
        return filtered_opts, [], [], err, "0 selected (max %d)" % MAX_ATTENDING

    if trig == "snake-invert" and n_invert:
        filtered_opts = filter_options(search)
        filtered_ids = [o["value"] for o in filtered_opts]
        current = set(new_val or [])
        inverted = [v for v in filtered_ids if v not in current]
        # enforce cap
        val = inverted[:MAX_ATTENDING]
        if len(inverted) > MAX_ATTENDING:
            err = f"Limited to {MAX_ATTENDING} after invert."
        count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
        return filtered_opts, val, val, err, count_text

    # Checklist changed or search typed
    filtered_opts = filter_options(search)

    if trig == "snake-search":
        # keep current selection but hide non-matching; Dash keeps value even if not visible
        # We leave value as-is; user can still generate.
        return filtered_opts, new_val or [], new_val or [], err, count_text

    if trig == "snake-attending-list":
        # enforce cap; if exceeded, revert to previous
        val = new_val or []
        if len(val) > MAX_ATTENDING:
            err = f"Max {MAX_ATTENDING} players. Extra selections ignored."
            count_text = f"{len(prev_val)} selected (max {MAX_ATTENDING})"
            return filtered_opts, prev_val, prev_val, err, count_text
        else:
            count_text = f"{len(val)} selected (max {MAX_ATTENDING})"
            return filtered_opts, val, val, "", count_text

    # Default pass-through
    return filtered_opts, new_val or [], new_val or [], err, count_text

# ---------- TAB 3 GENERATE CALLBACK ----------
@app.callback(
    Output("snake-seeding-table", "data"),
    Output("snake-lineups-table", "data"),
    Output("snake-lineups-wide", "data"),
    Output("snake-lineups-wide", "columns"),
    Output("snake-lineups-names", "data"),
    Output("snake-lineups-names", "columns"),
    Output("snake-seed-store", "data"),
    Output("snake-lineups-store", "data"),
    Output("snake-lineups-wide-store", "data"),
    Output("snake-msg", "children"),
    Output("snake-err", "children"),
    Input("snake-generate", "n_clicks"),
    State("snake-attending-list", "value"),
    State("snake-k", "value"),
    State("snake-periods", "value"),
    State("players-store", "data"),
    State("weights-store", "data"),
    prevent_initial_call=True
)
def snake_generate(n, attending_ids, k_on, periods, store_players, store_weights):
    if not n:
        return (no_update, no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update, no_update)

    players = pd.DataFrame(store_players) if store_players else players_load()
    weights = pd.DataFrame(store_weights) if store_weights else weights_load()

    empty_cols = [{"name":"period","id":"period"}]
    names_cols = [{"name":"period","id":"period"},{"name":"players","id":"players"}]

    if players is None or players.empty:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", "No saved players found (Tab 1)."

    if weights is None or weights.empty:
        weights = weights_empty_table()

    if not attending_ids:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", "No attending players selected."
    if len(attending_ids) > MAX_ATTENDING:
        attending_ids = attending_ids[:MAX_ATTENDING]
    if k_on is None or k_on < 1:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", "Players on court must be at least 1."
    if periods is None or periods < 1:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", "Number of periods must be at least 1."

    players_att = players[players["player_id"].isin(attending_ids)].copy()
    if len(players_att) < k_on:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", f"Need at least {k_on} attending players (selected: {len(players_att)})."

    wissues = weights_validate(weights.copy())
    if wissues:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", f"Weights invalid: {'; '.join(wissues)}"

    comp_df = compute_composites(players_att, weights)

    try:
        seeded_view, schedule_df = build_full_schedule(comp_df, periods=periods, k_on=k_on)
    except Exception as e:
        return [], [], [], empty_cols, [], names_cols, None, None, None, "", f"Schedule build error: {e}"

    wide_df = schedule_to_wide(schedule_df, seeded_view)
    names_df = schedule_to_names(schedule_df)

    try:
        schedule_df.to_csv(EXPORT_LINEUPS, index=False)
        seeded_view.to_csv(EXPORT_SEEDING, index=False)
        if not wide_df.empty:
            wide_df.to_csv(EXPORT_WIDE, index=False)
        msg = f"Generated {periods} periods."
    except Exception as e:
        msg = f"Generated {periods} periods. Export failed: {e}"

    if wide_df.empty:
        wide_cols = [{"name": "period", "id": "period"}]
        wide_data = []
    else:
        wide_cols = [{"name": c, "id": c} for c in wide_df.columns]
        wide_data = wide_df.to_dict(orient="records")

    if names_df.empty:
        n_cols = [{"name":"period","id":"period"},{"name":"players","id":"players"}]
        names_data = []
    else:
        n_cols = [{"name": c, "id": c} for c in names_df.columns]
        names_data = names_df.to_dict(orient="records")

    return (
        seeded_view.replace({np.nan: None}).to_dict(orient="records"),
        schedule_df.replace({np.nan: None}).to_dict(orient="records"),
        wide_data,
        wide_cols,
        names_data,
        n_cols,
        seeded_view.replace({np.nan: None}).to_dict(orient="records"),
        schedule_df.replace({np.nan: None}).to_dict(orient="records"),
        wide_data,
        msg,
        ""
    )

# ---------- HEALTH ENDPOINT (for hosts) ----------
@server.get("/health")
def health():
    return "ok", 200

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    PERSIST_PLAYERS.parent.mkdir(parents=True, exist_ok=True)
    PERSIST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8052"))
    app.run(debug=False, host=host, port=port)
