# -*- coding: utf-8 -*-
"""
Egypt maize | APSIM NextGen + ML downscaling (0.1°) + Climate (MET) + Soil (ISRIC SoilGrids) + CMIP6 (fallback) + EDA
Python/Colab version (mirrors the structure of the provided R master script).

USAGE (Colab):
  1) Upload your project folder (or mount Google Drive)
  2) Set BASE_DIR / WORK_DIR / COARSE_DIR below
  3) (Optional) Install APSIM NG on Linux OR provide Models executable path
  4) Run all cells (or run this as a script).

NOTES:
- APSIM NextGen runs on Linux, but in Google Colab you typically *can't* run docker (no privileged mode).
  Recommended for Colab:
    * Use a native Linux install (.deb) if available to you, OR
    * Run APSIM on your local machine/HPC, then upload the produced .db files here to do the ML + maps steps.
- APSIM output is stored in a SQLite database (*.db). We read the Report table from there.

References (APSIM NG docs):
- Command line + batch files: https://apsimnextgeneration.netlify.app/usage/commandline/batch/
- Linux / Docker guidance: https://apsimnextgeneration.netlify.app/usage/commandline/command-line-linux/
"""

import os
import re
import json
import math
import time
import glob
import shutil
import sqlite3
import zipfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- plotting (matplotlib only to keep Colab light) ---
import matplotlib.pyplot as plt

# --- geospatial ---
import geopandas as gpd
from shapely.geometry import Point
import shapely

# --- ML ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors


# ==============================================================================
# 0) USER SETTINGS (edit these paths)
# ==============================================================================

# If using Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# BASE_DIR = "/content/drive/MyDrive/APSIMICARDATraining/AnotherFullScript"

BASE_DIR = "/content"  # change
WORK_DIR = os.path.join(BASE_DIR, "EG_Maize_CMIP6_FULL")
COARSE_DIR = os.path.join(WORK_DIR, "coarse")
OUT_DIR = os.path.join(WORK_DIR, "outputs_master")
CACHE_DIR = os.path.join(WORK_DIR, "_cache")

APSIM_TEMPLATE = "MaizeFull.apsimx"
DB_NAME_PREF = "MaizeFull.db"

# Fine grid resolution for downscaling
RES_FINE = 0.1

# Fail threshold (t/ha)
YIELD_FAIL_THRESHOLD = 3.0

# Cores
N_CORES = max(1, (os.cpu_count() or 2) - 1)

# Climate feature season months
SEASON_MONTHS = {3, 4, 5, 6, 7, 8}

# CMIP6 fallback deltas (replace later with real deltas if you have them)
CMIP6 = pd.DataFrame({"scenario": ["SSP245", "SSP585"], "dT": [2.1, 3.8]})

# Uncertainty / probability-of-failure bootstrap settings
DO_BOOTSTRAP_UNCERTAINTY = True
N_BOOT = 50  # 30–80 typical

# SoilGrids settings
DO_SOILGRIDS = True
SOIL_DEPTHS_CM = ["0-5", "5-15", "15-30"]  # will compute weighted 0–30cm
SOIL_PROPS = ["sand", "clay", "silt", "soc", "bdod", "phh2o", "cec"]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def msg(*args):
    print(f"[{time.strftime('%H:%M:%S')}] " + (" ".join(str(a) for a in args)))


# ==============================================================================
# 1) HELPERS
# ==============================================================================

def to_float(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def impute_median(arr: pd.Series) -> pd.Series:
    s = pd.to_numeric(arr, errors="coerce")
    med = np.nanmedian(s.to_numpy(dtype=float))
    if not np.isfinite(med):
        return s
    return s.fillna(med)


def xy_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    m = np.column_stack([lon.astype(float), lat.astype(float)])
    return m


def idw_knn(src_xy: np.ndarray, src_val: np.ndarray, trg_xy: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    """
    kNN inverse-distance weighted interpolation (IDW).
    """
    src_val = np.asarray(src_val, dtype=float)
    if np.all(np.isnan(src_val)):
        raise ValueError("All NA values in interpolation source")

    k_eff = min(k, src_xy.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(src_xy)
    d, idx = nn.kneighbors(trg_xy, return_distance=True)

    d = np.where(d == 0, 1e-12, d)
    w = 1.0 / (d ** power)

    v = src_val[idx]
    num = np.nansum(w * v, axis=1)
    den = np.nansum(w, axis=1)
    out = num / den
    return out


# ==============================================================================
# 2) INDEX COARSE SITES
# ==============================================================================

msg("Indexing coarse site folders...")

site_dirs = [p for p in sorted(Path(COARSE_DIR).glob("EG_[0-9][0-9][0-9][0-9]")) if p.is_dir()]
run_index = pd.DataFrame({
    "site_id": [p.name for p in site_dirs],
    "site_dir": [str(p) for p in site_dirs],
})
run_index["apsimx_fp"] = run_index["site_dir"].apply(lambda d: str(Path(d) / APSIM_TEMPLATE))
run_index = run_index[run_index["apsimx_fp"].apply(lambda fp: Path(fp).exists())].reset_index(drop=True)

if len(run_index) == 0:
    raise FileNotFoundError(f"No site folders found with {APSIM_TEMPLATE} under: {COARSE_DIR}")

msg(f"Found {len(run_index)} site(s). Example:", run_index.loc[0, "apsimx_fp"])


# ==============================================================================
# 3) EDIT REPORT NODE IN .apsimx (JSON)
# ==============================================================================

REPORT_VARS = [
    "[Clock].Today as Date",
    "[Maize].Grain.Total.Wt*10 as Yield",   # g/m2 -> kg/ha (x10)
    "[Maize].Leaf.LAI as LAI",
    "[Maize].AboveGround.Wt as Biomass",
]

def _walk_json(node, fn):
    """
    DFS walk through dict/list json structure; apply fn to each dict node.
    """
    if isinstance(node, dict):
        fn(node)
        for k, v in node.items():
            _walk_json(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk_json(v, fn)

def ensure_report(apsimx_fp: str) -> None:
    """
    Set Report.EventNames and Report.VariableNames.
    This mirrors apsimx::edit_apsimx(node="Report", parm=...) in the R script.

    Assumption: APSIM .apsimx is JSON where a node has "Name": "Report"
    and properties keys may vary by APSIM version. We handle common patterns.
    """
    p = Path(apsimx_fp)
    obj = json.loads(p.read_text(encoding="utf-8"))

    found = {"n": 0}

    def edit_if_report(d: dict):
        if d.get("Name", "").lower() == "report":
            found["n"] += 1
            # Common APSIM schema uses "EventNames" and "VariableNames"
            d["EventNames"] = ["EndOfDay"]
            d["VariableNames"] = REPORT_VARS

    _walk_json(obj, edit_if_report)

    if found["n"] == 0:
        raise ValueError(f"Could not find a JSON node with Name=='Report' in {apsimx_fp}. "
                         "Open the .apsimx (it is JSON) and confirm the Report node name.")

    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ==============================================================================
# 4) RUN APSIM (OPTIONAL): call Models executable
# ==============================================================================

def guess_models_exe() -> Optional[str]:
    """
    Try to locate APSIM NextGen Models executable on Linux.
    Common paths after .deb install include /usr/local/bin/Models.
    """
    candidates = [
        "/usr/local/bin/Models",
        "/usr/local/bin/Models.exe",
        "/usr/bin/Models",
        "/usr/bin/Models.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which("Models") or shutil.which("Models.exe")

def run_apsim_safely(apsimx_fp: str, models_exe: Optional[str] = None) -> None:
    """
    Run an APSIM .apsimx file. If models_exe is None, we try to auto-detect.
    """
    models_exe = models_exe or guess_models_exe()
    if not models_exe:
        raise FileNotFoundError(
            "Could not find APSIM Models executable. "
            "Install APSIM NG on Linux (.deb) or set models_exe explicitly."
        )
    cmd = [models_exe, apsimx_fp]
    # APSIM writes output to <apsimx>.db by default
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


# ==============================================================================
# 5) ROBUST DB READ (SQLite)
# ==============================================================================

def read_report_from_db(site_dir: str, db_name: str = DB_NAME_PREF) -> Optional[pd.DataFrame]:
    """
    Read first Report-like table from APSIM SQLite .db.
    """
    site_dir = str(site_dir)
    db_fp = Path(site_dir) / db_name
    if not db_fp.exists():
        dbs = list(Path(site_dir).glob("*.db"))
        if not dbs:
            return None
        db_fp = dbs[0]

    con = sqlite3.connect(str(db_fp))
    try:
        tabs = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        if not tabs:
            return None
        rep_tabs = [t for t in tabs if not t.startswith("_")]
        pick = [t for t in rep_tabs if re.match(r"^Report", t, flags=re.IGNORECASE)]
        if not pick:
            pick = rep_tabs
        if not pick:
            return None
        df = pd.read_sql(f"SELECT * FROM [{pick[0]}]", con)
        if df.empty:
            return None
        if "Date" in df.columns:
            # APSIM often stores datetime strings
            df["Date"] = pd.to_datetime(df["Date"].astype(str).str.replace(r"\s.*", "", regex=True), errors="coerce").dt.date
        return df
    finally:
        con.close()


# ==============================================================================
# 6) COORDS (from .met header) + MET READER
# ==============================================================================

def parse_met_lonlat(site_dir: str) -> Tuple[float, float]:
    mets = list(Path(site_dir).glob("*.met"))
    if not mets:
        return (np.nan, np.nan)
    x = mets[0].read_text(encoding="utf-8", errors="ignore").splitlines()

    lat_line = next((l for l in x if re.search(r"latitude", l, re.IGNORECASE)), None)
    lon_line = next((l for l in x if re.search(r"longitude", l, re.IGNORECASE)), None)

    def grab_num(s: Optional[str]) -> float:
        if not s:
            return np.nan
        m = re.search(r"(-?\d+\.?\d*)", s)
        return to_float(m.group(1)) if m else np.nan

    return (grab_num(lon_line), grab_num(lat_line))


def read_met_data(site_dir: str) -> Optional[pd.DataFrame]:
    mets = list(Path(site_dir).glob("*.met"))
    if not mets:
        return None
    lines = mets[0].read_text(encoding="utf-8", errors="ignore").splitlines()

    # find header line starting with "year day"
    i0 = None
    for i, l in enumerate(lines):
        if re.match(r"^\s*year\s+day\b", l, flags=re.IGNORECASE):
            i0 = i
            break
    if i0 is None:
        return None

    hdr = re.sub(r"\s+", " ", lines[i0].strip().lower())
    cols = hdr.split(" ")

    start_line = i0 + 2  # skip units line
    if start_line >= len(lines):
        return None

    # fixed-width whitespace
    data_txt = "\n".join(lines[start_line:])
    df = pd.read_csv(
        pd.io.common.StringIO(data_txt),
        delim_whitespace=True,
        header=None,
        engine="python",
    )
    if df.shape[1] < len(cols):
        return None
    df = df.iloc[:, :len(cols)]
    df.columns = cols

    for nm in set(["year", "day", "maxt", "mint", "radn", "rain", "rh", "windspeed"]).intersection(df.columns):
        df[nm] = pd.to_numeric(df[nm], errors="coerce")
    return df


def calc_DD35(tmax: np.ndarray) -> float:
    return float(np.nansum(np.maximum(tmax - 35.0, 0.0)))


def calc_HDW(tmax: np.ndarray, tmin: np.ndarray) -> float:
    tmean = (tmax + tmin) / 2.0
    return float(np.nansum((tmax > 35.0) & (tmean > 30.0)))


def compute_climate_features_site(site_id: str, site_dir: str, season_months: Optional[set] = None) -> Optional[pd.DataFrame]:
    met = read_met_data(site_dir)
    if met is None or not {"year", "day"}.issubset(met.columns):
        return None

    # build dates to filter months
    # date = origin + (day-1)
    d = pd.to_datetime(met["year"].astype(int).astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(met["day"] - 1, unit="D")

    if season_months is not None:
        keep = d.dt.month.isin(sorted(list(season_months)))
        met = met.loc[keep].copy()
        d = d.loc[keep]

    if met.empty:
        return None

    ny = met["year"].nunique()
    if not np.isfinite(ny) or ny <= 0:
        ny = 1

    tmax = met.get("maxt", pd.Series(dtype=float)).to_numpy(dtype=float)
    tmin = met.get("mint", pd.Series(dtype=float)).to_numpy(dtype=float)
    radn = met.get("radn", pd.Series(dtype=float)).to_numpy(dtype=float)
    rain = met.get("rain", pd.Series(dtype=float)).to_numpy(dtype=float)
    rh   = met.get("rh",   pd.Series(dtype=float)).to_numpy(dtype=float)
    wind = met.get("windspeed", pd.Series(dtype=float)).to_numpy(dtype=float)

    out = {
        "site_id": site_id,
        "DD35": calc_DD35(tmax) / ny,
        "HDW":  calc_HDW(tmax, tmin) / ny,
        "tmax_mean": float(np.nanmean(tmax)),
        "tmin_mean": float(np.nanmean(tmin)),
        "tmean_mean": float(np.nanmean((tmax + tmin) / 2.0)),
        "tmax_p95": float(np.nanpercentile(tmax, 95)),
        "tmin_p05": float(np.nanpercentile(tmin, 5)),
        "radn_mean": float(np.nanmean(radn)),
        "rain_sum": float(np.nansum(rain) / ny),
        "rain_p95": float(np.nanpercentile(rain, 95)),
        "rh_mean": float(np.nanmean(rh)),
        "wind_mean": float(np.nanmean(wind)),
        "wind_p95": float(np.nanpercentile(wind, 95)),
    }
    return pd.DataFrame([out])


# ==============================================================================
# 7) SOIL (ISRIC SoilGrids v2 API)
# ==============================================================================

import requests

def soilgrids_point(lon: float, lat: float,
                    props: List[str] = SOIL_PROPS,
                    depths: List[str] = SOIL_DEPTHS_CM,
                    timeout_sec: int = 60) -> pd.DataFrame:
    """
    Always returns exactly 1 row (NA if request fails), to keep joins safe.
    Weighted aggregation to 0–30 cm using (0-5,5-15,15-30) weights 5,10,15.
    """
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    out_cols = [f"{p}_0_30" for p in props]

    weights = {"0-5": 5, "5-15": 10, "15-30": 15}
    w = np.array([weights[d] for d in depths], dtype=float)
    w = w / w.sum()

    out_fail = pd.DataFrame([{c: np.nan for c in out_cols}])

    if not (np.isfinite(lon) and np.isfinite(lat)):
        return out_fail

    params = {
        "lon": lon,
        "lat": lat,
        "property": props,
        "depth": depths,
        "value": "mean",
    }

    try:
        r = requests.get(base_url, params=params, timeout=timeout_sec)
        if r.status_code != 200:
            return out_fail
        js = r.json()
        if "properties" not in js:
            return out_fail

        def get_mean(prop: str, depth: str) -> float:
            p = js["properties"].get(prop)
            if not p or "depths" not in p:
                return np.nan
            top, bot = depth.split("-")
            top = int(top); bot = int(bot)
            for dd in p["depths"]:
                rg = dd.get("range", {})
                if rg.get("top_depth") == top and rg.get("bottom_depth") == bot:
                    v = dd.get("values", {}).get("mean")
                    return to_float(v)
            return np.nan

        out = {}
        for prop in props:
            vals = np.array([get_mean(prop, d) for d in depths], dtype=float)
            out[f"{prop}_0_30"] = float(np.nansum(vals * w))

        # ensure full colset
        for c in out_cols:
            out.setdefault(c, np.nan)

        return pd.DataFrame([out])[out_cols]

    except Exception:
        return out_fail


def get_soil_features_for_sites(site_tbl: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    soil_cache_fp = Path(cache_dir) / "soilgrids_coarse_sites.csv"
    out_cols = ["site_id"] + [f"{p}_0_30" for p in SOIL_PROPS]

    if soil_cache_fp.exists():
        sc = pd.read_csv(soil_cache_fp)
        if "site_id" in sc.columns and set(site_tbl["site_id"]).issubset(set(sc["site_id"])):
            for c in out_cols:
                if c not in sc.columns:
                    sc[c] = np.nan
            return sc[out_cols]

    msg("Querying SoilGrids for coarse sites (cached)...")

    rows = []
    for _, r in site_tbl.iterrows():
        sid = r["site_id"]
        lon = to_float(r["lon"])
        lat = to_float(r["lat"])
        fp = Path(cache_dir) / f"soil_{sid}.json"

        if fp.exists():
            try:
                z = json.loads(fp.read_text(encoding="utf-8"))
                row = {"site_id": sid, **z}
                rows.append(row)
                continue
            except Exception:
                pass

        zdf = soilgrids_point(lon, lat)
        z = zdf.iloc[0].to_dict()
        fp.write_text(json.dumps(z), encoding="utf-8")
        rows.append({"site_id": sid, **z})

    sc = pd.DataFrame(rows)
    for c in out_cols:
        if c not in sc.columns:
            sc[c] = np.nan
    sc = sc[out_cols]
    sc.to_csv(soil_cache_fp, index=False)
    return sc


# ==============================================================================
# 8) PARALLEL: EDIT + RUN + CHECK
# ==============================================================================

def process_one_site(site_id: str, site_dir: str, apsimx_fp: str,
                     do_run: bool = True,
                     models_exe: Optional[str] = None) -> Dict:
    """
    Edit report, run APSIM (optional), then verify DB report exists.
    """
    try:
        ensure_report(apsimx_fp)
    except Exception as e:
        return {"site_id": site_id, "site_dir": site_dir, "ok": False, "step": "edit_report", "msg": str(e)}

    if do_run:
        try:
            run_apsim_safely(apsimx_fp, models_exe=models_exe)
        except Exception as e:
            return {"site_id": site_id, "site_dir": site_dir, "ok": False, "step": "run_apsim", "msg": str(e)}

    try:
        df = read_report_from_db(site_dir)
        if df is None or df.empty:
            return {"site_id": site_id, "site_dir": site_dir, "ok": False, "step": "read_db",
                    "msg": "DB exists but no non-empty Report table found."}
    except Exception as e:
        return {"site_id": site_id, "site_dir": site_dir, "ok": False, "step": "read_db", "msg": str(e)}

    return {"site_id": site_id, "site_dir": site_dir, "ok": True, "step": "done", "msg": ""}


def run_all_sites(run_index: pd.DataFrame,
                  do_run: bool = True,
                  models_exe: Optional[str] = None) -> pd.DataFrame:
    """
    Run all sites with a small parallel pool.
    """
    msg(f"Processing sites (do_run={do_run}) using ~{N_CORES} core(s)...")
    try:
        from joblib import Parallel, delayed
        logs = Parallel(n_jobs=N_CORES)(
            delayed(process_one_site)(
                r.site_id, r.site_dir, r.apsimx_fp, do_run=do_run, models_exe=models_exe
            )
            for r in run_index.itertuples(index=False)
        )
    except Exception:
        # fallback serial
        logs = [process_one_site(r.site_id, r.site_dir, r.apsimx_fp, do_run=do_run, models_exe=models_exe)
                for r in run_index.itertuples(index=False)]
    return pd.DataFrame(logs)


# Set DO_RUN_APSIM=False if you already have .db outputs from elsewhere (recommended for Colab)
DO_RUN_APSIM = False
MODELS_EXE = None  # e.g. "/usr/local/bin/Models"

run_log = run_all_sites(run_index, do_run=DO_RUN_APSIM, models_exe=MODELS_EXE)
run_log.to_csv(os.path.join(OUT_DIR, "run_log.csv"), index=False)
msg("Run success:", int(run_log["ok"].sum()), "/", len(run_log))


# ==============================================================================
# 9) COLLECT COARSE APSIM OUTPUTS (max Yield/LAI/Biomass)
# ==============================================================================

msg("Collecting coarse outputs (max Yield/LAI/Biomass)...")

rows = []
for r in run_index.itertuples(index=False):
    ok = bool(run_log.loc[run_log["site_id"] == r.site_id, "ok"].iloc[0])
    if not ok:
        continue
    df = read_report_from_db(r.site_dir)
    if df is None or df.empty:
        continue

    def get_col(name):
        if name not in df.columns:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

    Yield = get_col("Yield")  # kg/ha because g/m2*10
    LAI = get_col("LAI")
    Biomass = get_col("Biomass")

    Yield_t_ha = Yield / 1000.0

    rows.append({
        "site_id": r.site_id,
        "Yield_max_t_ha": float(np.nanmax(Yield_t_ha)),
        "LAI_max": float(np.nanmax(LAI)),
        "Biomass_max": float(np.nanmax(Biomass)),
    })

coarse_out = pd.DataFrame(rows)
coarse_out.to_csv(os.path.join(OUT_DIR, "coarse_apsim_summary.csv"), index=False)
msg("Coarse summary rows:", len(coarse_out))


# ==============================================================================
# 10) COORDS + CLIMATE FEATURES FROM MET
# ==============================================================================

msg("Extracting site lon/lat + climate features from MET...")

coords = []
for r in run_index.itertuples(index=False):
    lon, lat = parse_met_lonlat(r.site_dir)
    coords.append({"site_id": r.site_id, "lon": lon, "lat": lat})
coords = pd.DataFrame(coords)

clim_rows = []
for r in run_index.itertuples(index=False):
    ok = bool(run_log.loc[run_log["site_id"] == r.site_id, "ok"].iloc[0])
    if not ok:
        continue
    cf = compute_climate_features_site(r.site_id, r.site_dir, season_months=SEASON_MONTHS)
    if cf is not None:
        clim_rows.append(cf)
clim_feat = pd.concat(clim_rows, ignore_index=True) if clim_rows else pd.DataFrame({"site_id": []})

coarse_site = (
    coarse_out.merge(coords, on="site_id", how="left")
             .merge(clim_feat, on="site_id", how="left")
)

# ==============================================================================
# 11) SOIL FEATURES (SoilGrids)
# ==============================================================================

if DO_SOILGRIDS:
    msg("Adding soil predictors (SoilGrids)...")
    soil_in = coarse_site[["site_id", "lon", "lat"]].copy()
    soil_in = soil_in[np.isfinite(soil_in["lon"]) & np.isfinite(soil_in["lat"])]
    soil_feat = get_soil_features_for_sites(soil_in, cache_dir=CACHE_DIR)
    coarse_site = coarse_site.merge(soil_feat, on="site_id", how="left")

coarse_site.to_csv(os.path.join(OUT_DIR, "coarse_site_features_all.csv"), index=False)


# ==============================================================================
# 12) BUILD EGYPT 0.1° GRID (centers inside Egypt polygon)
# ==============================================================================

msg("Building Egypt 0.1° grid...")

# Natural Earth countries via geopandas datasets
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
egy = world[world["name"] == "Egypt"].to_crs("EPSG:4326")

minx, miny, maxx, maxy = egy.total_bounds
lon_vals = np.arange(math.floor(minx / RES_FINE) * RES_FINE, maxx + RES_FINE, RES_FINE)
lat_vals = np.arange(math.floor(miny / RES_FINE) * RES_FINE, maxy + RES_FINE, RES_FINE)

pts = []
for i, lon in enumerate(lon_vals):
    for j, lat in enumerate(lat_vals):
        p = Point(float(lon), float(lat))
        if egy.geometry.iloc[0].contains(p):
            pts.append((lon, lat))

fine = pd.DataFrame(pts, columns=["lon", "lat"])
fine["cell_id"] = [f"EGF_{i:06d}" for i in range(1, len(fine) + 1)]
msg("Fine grid points inside Egypt:", len(fine))


# ==============================================================================
# 13) ML TRAINING (coords + climate + soil) + PREDICT to 0.1°
# ==============================================================================

msg("Training ML model for yield downscaling (coords + climate + soil)...")

train_df = coarse_site.copy()
train_df = train_df[np.isfinite(train_df["Yield_max_t_ha"]) & np.isfinite(train_df["lon"]) & np.isfinite(train_df["lat"])]

if len(train_df) < 10:
    msg("WARNING: Too few training points. Will downscale by IDW Yield only.", len(train_df))

cand = [
    "lon", "lat",
    "DD35", "HDW", "tmax_mean", "tmin_mean", "tmean_mean", "tmax_p95", "tmin_p05",
    "radn_mean", "rain_sum", "rain_p95", "rh_mean", "wind_mean", "wind_p95",
] + [f"{p}_0_30" for p in SOIL_PROPS]

cand = [c for c in cand if c in train_df.columns]

def enough_values(col):
    v = pd.to_numeric(train_df[col], errors="coerce")
    return np.isfinite(v).sum() >= max(10, int(0.5 * len(train_df)))

ok_pred = [c for c in cand if enough_values(c)]
ok_pred = list(dict.fromkeys(["lon", "lat"] + [c for c in ok_pred if c not in ("lon", "lat")]))

msg("Predictors used:", len(ok_pred), ",".join(ok_pred))

for v in ok_pred:
    train_df[v] = impute_median(train_df[v])

src_xy = xy_matrix(train_df["lon"].to_numpy(), train_df["lat"].to_numpy())
trg_xy = xy_matrix(fine["lon"].to_numpy(), fine["lat"].to_numpy())

# interpolate all non-coordinate predictors to fine grid
for v in [x for x in ok_pred if x not in ("lon", "lat")]:
    try:
        fine[v] = idw_knn(src_xy, train_df[v].to_numpy(dtype=float), trg_xy)
    except Exception:
        fine[v] = np.nan
    fine[v] = impute_median(fine[v])

if len(train_df) >= 10:
    X = train_df[ok_pred].to_numpy(dtype=float)
    y = train_df["Yield_max_t_ha"].to_numpy(dtype=float)

    rf = RandomForestRegressor(
        n_estimators=900,
        random_state=123,
        n_jobs=min(8, N_CORES),
        bootstrap=True,
    )
    rf.fit(X, y)

    # Save model (simple joblib)
    try:
        import joblib
        joblib.dump(rf, os.path.join(OUT_DIR, "rf_model_yield.joblib"))
    except Exception:
        pass

    fine["Yield_pred_t_ha"] = rf.predict(fine[ok_pred].to_numpy(dtype=float))
else:
    fine["Yield_pred_t_ha"] = idw_knn(src_xy, train_df["Yield_max_t_ha"].to_numpy(dtype=float), trg_xy)

fine["Fail"] = fine["Yield_pred_t_ha"] < YIELD_FAIL_THRESHOLD
fine.to_csv(os.path.join(OUT_DIR, "yield_baseline_0p1deg.csv"), index=False)
msg("Saved baseline 0.1° yield.")


# ==============================================================================
# 14) UNCERTAINTY + FAIL PROBABILITY (bootstrap)
# ==============================================================================

if DO_BOOTSTRAP_UNCERTAINTY and len(train_df) >= 10:
    msg("Bootstrap uncertainty (n_boot=%d) ..." % N_BOOT)
    X = train_df[ok_pred].to_numpy(dtype=float)
    y = train_df["Yield_max_t_ha"].to_numpy(dtype=float)
    Xp = fine[ok_pred].to_numpy(dtype=float)

    preds = np.empty((len(fine), N_BOOT), dtype=float)
    rng = np.random.default_rng(999)

    for b in range(N_BOOT):
        rf_b = RandomForestRegressor(
            n_estimators=450,
            random_state=1000 + b,
            n_jobs=min(6, N_CORES),
            bootstrap=True,
        )
        rf_b.fit(X, y)
        preds[:, b] = rf_b.predict(Xp)

    fine["Yield_mean_t_ha"] = np.nanmean(preds, axis=1)
    fine["Yield_sd_t_ha"] = np.nanstd(preds, axis=1)
    fine["P_fail"] = np.nanmean(preds < YIELD_FAIL_THRESHOLD, axis=1)

    fine.to_csv(os.path.join(OUT_DIR, "yield_baseline_0p1deg_with_uncertainty.csv"), index=False)
    msg("Saved baseline + uncertainty + P_fail.")


# ==============================================================================
# 15) CMIP6 IMPACTS (fallback)
# ==============================================================================

msg("Computing CMIP6 scenario impacts (fallback)...")

base_DD35 = fine["DD35"] if "DD35" in fine.columns else pd.Series(np.zeros(len(fine)))
base_HDW  = fine["HDW"]  if "HDW"  in fine.columns else pd.Series(np.zeros(len(fine)))
base_DD35 = impute_median(base_DD35)
base_HDW  = impute_median(base_HDW)

# cartesian join
yield_cc = fine.assign(_k=1).merge(CMIP6.assign(_k=1), on="_k").drop(columns="_k")
yield_cc["DD35_f"] = base_DD35.to_numpy() + yield_cc["dT"].to_numpy() * 30.0
yield_cc["HDW_f"]  = base_HDW.to_numpy()  + yield_cc["dT"].to_numpy() * 10.0
yield_cc["Yield_CC_t_ha"] = yield_cc["Yield_pred_t_ha"] - 0.015*(yield_cc["DD35_f"]/30.0) - 0.05*(yield_cc["HDW_f"]/10.0)
yield_cc["Fail_CC"] = yield_cc["Yield_CC_t_ha"] < YIELD_FAIL_THRESHOLD

yield_cc.to_csv(os.path.join(OUT_DIR, "yield_cmip6_scenarios.csv"), index=False)
msg("Saved CMIP6 scenario yield impacts.")


# ==============================================================================
# 16) EDA (maps, distributions, importance, correlation)
# ==============================================================================

msg("Saving EDA outputs...")

# 16.1 Baseline yield map (scatter)
plt.figure()
plt.scatter(fine["lon"], fine["lat"], s=1, c=fine["Yield_pred_t_ha"])
plt.title("Baseline maize yield (0.1°) - ML (coords + climate + soil)")
plt.xlabel("lon"); plt.ylabel("lat")
plt.colorbar(label="t/ha")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "EDA_map_yield_baseline.png"), dpi=160)
plt.close()

# 16.2 Histogram + density (simple)
plt.figure()
plt.hist(fine["Yield_pred_t_ha"].to_numpy(dtype=float), bins=60)
plt.title("Yield histogram (downscaled 0.1°)")
plt.xlabel("t/ha"); plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "EDA_hist_yield_baseline.png"), dpi=160)
plt.close()

# 16.3 Training points
if len(train_df) > 0:
    plt.figure()
    plt.scatter(train_df["lon"], train_df["lat"], s=20, c=train_df["Yield_max_t_ha"])
    plt.title("Coarse APSIM yield (training points)")
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.colorbar(label="t/ha")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "EDA_coarse_training_points.png"), dpi=160)
    plt.close()

# 16.4 RF importance (if model exists)
try:
    imp = pd.Series(rf.feature_importances_, index=ok_pred).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "EDA_rf_importance_table.csv"), header=["importance"])
    plt.figure(figsize=(7, 5))
    imp.iloc[:20][::-1].plot(kind="barh")
    plt.title("RF feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "EDA_rf_importance.png"), dpi=160)
    plt.close()
except Exception:
    pass

# 16.5 Correlation heatmap (coarse)
eda_cols = ["Yield_max_t_ha"] + ok_pred
eda_cols = [c for c in eda_cols if c in coarse_site.columns]
eda_df = coarse_site[eda_cols].apply(pd.to_numeric, errors="coerce")
keep = [c for c in eda_df.columns if np.isfinite(eda_df[c]).sum() >= 8]
eda_df = eda_df[keep]
if eda_df.shape[1] >= 3:
    C = eda_df.corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(C, aspect="auto")
    plt.xticks(range(len(C.columns)), C.columns, rotation=45, ha="right")
    plt.yticks(range(len(C.index)), C.index)
    plt.colorbar(label="corr")
    plt.title("Correlation heatmap (coarse: yield + predictors)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "EDA_correlation_heatmap_coarse.png"), dpi=160)
    plt.close()

# 16.6 P_fail map + Yield SD map (if uncertainty)
if "P_fail" in fine.columns:
    plt.figure()
    plt.scatter(fine["lon"], fine["lat"], s=1, c=fine["P_fail"])
    plt.title(f"Probability of yield failure (Yield < {YIELD_FAIL_THRESHOLD} t/ha)")
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.colorbar(label="P_fail")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "EDA_map_P_fail.png"), dpi=160)
    plt.close()

if "Yield_sd_t_ha" in fine.columns:
    plt.figure()
    plt.scatter(fine["lon"], fine["lat"], s=1, c=fine["Yield_sd_t_ha"])
    plt.title("Prediction uncertainty (SD, t/ha)")
    plt.xlabel("lon"); plt.ylabel("lat")
    plt.colorbar(label="SD")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "EDA_map_uncertainty_SD.png"), dpi=160)
    plt.close()

msg("ALL DONE ✅")
msg("Outputs saved to:", OUT_DIR)
