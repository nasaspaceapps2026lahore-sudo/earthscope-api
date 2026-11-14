"""EarthScope API (NASA Space Apps 2026 - Be an Earth System Trend Detective)

Serves the EarthScope website from the gridded NetCDF files in src/data/input/.
Ibrahim's original endpoints (/tempanomaly, /precipitation, /ndvi) still work as before.
The website uses the contract endpoints (see API_CONTRACT.md):

  /health  /variables  /regions  /hotspots
  /trend-grid?var=   /value-grid?var=
  /series?var=&lat=&lon=   (or &region=)
  /forecast?var=&horizon=&lat=&lon=   (or &region=)
  /scenarios?var=...

Run:   uv run fastapi dev --port 8000      then open http://localhost:8000/docs
"""
from __future__ import annotations

import datetime
import logging
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import earthscope_core as core

log = logging.getLogger("earthscope")

# ------------------------------------------------------------------ where the data lives
HERE = Path(__file__).resolve().parent


def _data_dir() -> Path:
    for p in (os.environ.get("EARTHSCOPE_DATA"), Path.cwd() / "src/data/input", HERE / "src/data/input", HERE / "data/input"):
        if p and Path(p).is_dir():
            return Path(p)
    return Path.cwd() / "src/data/input"


DATA_DIR = _data_dir()

# Every dataset the website knows. A dataset is served only if its file exists,
# so adding GRACE / land cover / aerosols later is just dropping the file in.
DATASETS: dict[str, dict] = {
    "ndvi": dict(files=["ndvi_2deg.nc"], vars=["ndvi", "NDVI"], start=2000, name="Vegetation (NDVI)", short="NDVI", unit="", decimals=3, landOnly=True,
                 source="MODIS vegetation index (NDVI), regridded to 2°", up="greening", down="browning", recent=10),
    "temp": dict(files=["temperature_2deg.nc", "temperature_data.nc"], vars=["temperature", "tempanomaly"], start=1980, name="Surface temperature", short="Temperature", unit="°C", decimals=2, landOnly=False,
                 source="NASA GISTEMP v4 (anomaly vs 1951–1980)", up="warming", down="cooling", recent=5),
    "precip": dict(files=["precipitation_2deg.nc"], vars=["precipitation", "precip"], start=2000, name="Precipitation", short="Precipitation", unit="mm/month", decimals=1, landOnly=False,
                   source="NASA GPM IMERG monthly, regridded to 2°", up="getting wetter", down="getting drier", recent=10),
    # optional, served when the file is present
    "grace": dict(files=["grace_2deg.nc"], vars=["lwe_thickness", "grace"], start=2002, name="Water storage", short="Water Storage", unit="cm", decimals=1, landOnly=True,
                  source="GRACE / GRACE-FO JPL mascons (equivalent water height)", up="gaining water", down="losing water", recent=5),
    "landcover": dict(files=["landcover_2deg.nc"], vars=["forest", "landcover"], start=2001, name="Land cover (forest share)", short="Land Cover", unit="% forest", decimals=1, landOnly=True,
                      source="MODIS MCD12Q1 land cover, forest classes", up="gaining forest", down="losing forest", recent=5),
    "aq": dict(files=["aod_2deg.nc"], vars=["aod", "AOD"], start=2000, name="Air quality (aerosols)", short="Air Quality", unit="AOD", decimals=3, landOnly=False,
               source="MODIS aerosol optical depth", up="getting hazier", down="getting cleaner", recent=10),
    "lst": dict(files=["lst_2deg.nc"], vars=["lst", "LST"], start=2000, name="Land surface temperature", short="Land Heat", unit="°C", decimals=2, landOnly=True,
                source="MODIS MOD11C3 (daytime)", up="heating up", down="cooling", recent=10),
}

REGIONS = [
    dict(id="indus", name="Indus plains", lat=30.5, lon=73.5, scene="farmland", blurb="Irrigated farmland over a heavily pumped aquifer."),
    dict(id="amazon", name="Amazon Rainforest", lat=-10.0, lon=-56.0, scene="forest", blurb="Forest edge in Mato Grosso and Pará."),
    dict(id="sahel", name="Sahel", lat=14.0, lon=2.0, scene="savanna", blurb="Semi-arid belt south of the Sahara."),
    dict(id="alaska", name="Arctic Alaska", lat=69.0, lon=-152.0, scene="tundra", blurb="Tundra on the North Slope."),
    dict(id="centralvalley", name="California Central Valley", lat=36.8, lon=-120.0, scene="farmland", blurb="Farm valley that leans on groundwater in drought."),
    dict(id="aral", name="Aral Sea basin", lat=45.0, lon=60.0, scene="desert", blurb="A shrunken lake and its dry seabed."),
    dict(id="mediterranean", name="Central Mediterranean", lat=38.0, lon=15.0, scene="coast", blurb="Sicily, southern Italy and the sea around them."),
    dict(id="murray", name="Murray–Darling basin", lat=-33.0, lon=145.0, scene="savanna", blurb="Australia’s main farming river system."),
    dict(id="horn", name="Horn of Africa", lat=6.0, lon=40.0, scene="savanna", blurb="Ethiopian and Somali drylands."),
    dict(id="greenland", name="Greenland ice sheet", lat=72.0, lon=-40.0, scene="ice", blurb="The largest ice mass in the north."),
    dict(id="sahara", name="Sahara Desert", lat=23.0, lon=12.0, scene="desert", blurb="The largest hot desert on Earth."),
    dict(id="europe", name="Western Europe", lat=48.0, lon=6.0, scene="temperate", blurb="Farmland and forest from France to Germany."),
]
REGION_BY_ID = {r["id"]: r for r in REGIONS}

# How each record is named in Trend Detective sentences: (label, (word when rising, word when falling))
PHRASES = {
    "ndvi": ("Vegetation (NDVI)", ("is greening", "is browning")),
    "temp": ("Temperature", ("is warming", "is cooling")),
    "precip": ("Rainfall", ("is rising", "is falling")),
    "grace": ("Water stored on and under the ground", ("is rising", "is falling")),
    "landcover": ("Forest cover", ("is growing", "is shrinking")),
    "aq": ("Haze (aerosols)", ("is rising", "is falling")),
    "lst": ("Land surface heat", ("is rising", "is falling")),
}

# Trend Detective candidates: (region, preferred pair, fallback pair if a dataset is missing, likely driver)
HOTSPOTS = [
    ("indus", ["ndvi", "grace"], ["ndvi", "precip"], "Irrigation from groundwater"),
    ("alaska", ["temp", "ndvi"], None, "Longer, warmer growing seasons"),
    ("centralvalley", ["ndvi", "grace"], ["ndvi", "precip"], "Pumping groundwater through droughts"),
    ("mediterranean", ["temp", "precip"], None, "A warming, drying climate"),
    ("sahel", ["ndvi", "precip"], None, "Rainfall recovery after the 1970s–80s droughts"),
    ("amazon", ["ndvi", "precip"], ["ndvi", "temp"], "Drought and forest clearing"),
    ("aral", ["grace", "lst"], ["precip", "temp"], "Water diverted for irrigation"),
]


# ------------------------------------------------------------------ loading
def _pick(names, options):
    for o in options:
        if o in names:
            return o
    return None


def _to_periods(times) -> pd.PeriodIndex:
    idx = times
    if hasattr(idx, "to_datetimeindex"):  # cftime calendars
        idx = idx.to_datetimeindex()
    return pd.DatetimeIndex(idx).to_period("M")


def find_file(key: str) -> Path | None:
    for name in DATASETS[key]["files"]:
        for p in (DATA_DIR / name, (DATA_DIR / name).with_suffix(".npz")):
            if p.exists():
                return p
    return None


def load_npz(key: str, path: Path) -> core.Cube:
    """Pre-cleaned grid (made by tools/make_npz.py): small enough for free hosting and GitHub uploads."""
    z = np.load(path, allow_pickle=False)
    months = pd.PeriodIndex(list(z["months"]), freq="M")
    return core.Cube(str(z["var"]), z["lat"].astype(float), z["lon"].astype(float), months, z["data"].astype(np.float32))


SOURCES: dict[str, tuple[Path, str]] = {}  # key -> (file, variable name actually used)


def load_cube(key: str) -> core.Cube | None:
    """Read one NetCDF file with xarray; all cleaning happens in core.prepare_cube (tested)."""
    spec = DATASETS[key]
    path = find_file(key)
    if path is None:
        return None
    if path.suffix == ".npz":
        c = load_npz(key, path)
        SOURCES[key] = (path, c.var)
        log.info("loaded %s from %s: %s..%s, %d×%d cells at %g°", key, path.name, c.months[0], c.months[-1], len(c.lat), len(c.lon), c.res)
        return c
    with xr.open_dataset(path) as ds:
        var = _pick(list(ds.data_vars), spec["vars"]) or next((v for v in ds.data_vars if ds[v].ndim == 3), None)
        if var is None:
            log.warning("%s: no 3-D variable found", path.name)
            return None
        da = ds[var]
        dims = list(da.dims)
        lat_n = _pick(dims, ("lat", "latitude", "y"))
        lon_n = _pick(dims, ("lon", "longitude", "x"))
        t_n = _pick(dims, ("time", "t", "date"))
        if not (lat_n and lon_n and t_n):
            log.warning("%s: expected time/lat/lon dimensions, found %s", path.name, dims)
            return None
        order = [dims.index(t_n), dims.index(lat_n), dims.index(lon_n)]
        c = core.prepare_cube(key, da.values, order, da[lat_n].values, da[lon_n].values,
                              _to_periods(da.indexes[t_n]), dict(da.attrs), spec["start"], log=log)
    SOURCES[key] = (path, var)
    log.info("loaded %s from %s (variable '%s'): %s..%s, %d×%d cells at %g°", key, path.name, var, c.months[0], c.months[-1], len(c.lat), len(c.lon), c.res)
    return c


CUBES: dict[str, core.Cube] = {}
_lock = threading.Lock()
_grids: dict[tuple, dict] = {}


def cube(key: str) -> core.Cube:
    if key not in CUBES:
        raise HTTPException(404, f"Unknown or unavailable variable '{key}'. Available: {', '.join(CUBES) or 'none'}")
    return CUBES[key]


def first_year(key: str) -> int:
    return max(DATASETS[key]["start"], int(CUBES[key].months[0].year))


def last_full_year(key: str) -> int:
    """Last calendar year with all 12 months. A part year (e.g. Jan-Sep) would skew seasonal records like rain."""
    last = CUBES[key].months[-1]
    return int(last.year) if last.month == 12 else int(last.year) - 1


def cached_grid(kind: str, key: str) -> dict:
    with _lock:
        if (kind, key) in _grids:
            return _grids[(kind, key)]
    c, spec = cube(key), DATASETS[key]
    if kind == "trend":
        g = core.trend_grid(c, first_year(key), target_res=4.0)
    else:
        g = core.value_grid(c, years_back=spec["recent"], label="mean anomaly" if key in ("temp", "grace") else "mean", decimals=spec["decimals"] + 1)
    with _lock:
        _grids[(kind, key)] = g
    return g


def _prewarm():
    for k in list(CUBES):
        for kind in ("trend", "value"):
            try:
                cached_grid(kind, k)
            except Exception:  # noqa: BLE001
                log.exception("pre-computing %s grid for %s failed", kind, k)
    log.info("grids ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    log.info("data folder: %s", DATA_DIR)
    for k in DATASETS:
        try:
            c = load_cube(k)
            if c is not None:
                CUBES[k] = c
        except Exception:  # noqa: BLE001
            log.exception("could not load %s", k)
    if not CUBES:
        log.error("no datasets loaded from %s", DATA_DIR)
    threading.Thread(target=_prewarm, daemon=True).start()
    yield
    for ds in _raw.values():
        ds.close()


app = FastAPI(title="EarthScope API", version="1.0.0", lifespan=lifespan,
              description="Data service for the EarthScope website. Contract: API_CONTRACT.md")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)


# ------------------------------------------------------------------ helpers
def _clean(x: float) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def locate(key: str, lat: float | None, lon: float | None, region: str | None):
    """Monthly values and a location dict for a point, a region (box average) or 'global'."""
    c = cube(key)
    if region:
        if region == "global":
            return c.global_mean(), {"lat": 0.0, "lon": 0.0, "name": "Global (area-weighted mean)"}
        r = REGION_BY_ID.get(region)
        if not r:
            raise HTTPException(404, f"Unknown region '{region}'")
        try:
            vals, info = c.box(r["lat"], r["lon"], half=1)
        except LookupError:
            raise HTTPException(404, f"No {DATASETS[key]['short']} data in {r['name']}")
        return vals, {"lat": r["lat"], "lon": r["lon"], "name": f"{r['name']} · {info['note']}"}
    if lat is None or lon is None:
        raise HTTPException(422, "Give lat and lon, or region")
    if not (-90 <= lat <= 90 and -180 <= lon <= 360):
        raise HTTPException(422, "lat must be -90..90 and lon -180..180")
    try:
        vals, info = c.point(lat, lon)
    except LookupError:
        raise HTTPException(404, f"No {DATASETS[key]['short']} data near {lat:.2f}, {lon:.2f}"
                                 + (" (this dataset covers land only)" if DATASETS[key]["landOnly"] else ""))
    return vals, {"lat": info["lat"], "lon": info["lon"], "name": info["note"]}


@lru_cache(maxsize=512)
def _series(key: str, lat, lon, region):
    vals, loc = locate(key, lat, lon, region)
    return vals, loc


def _r2(x):
    return None if x is None else round(float(x), 2)


def _version() -> str:
    files = [SOURCES[k][0] for k in CUBES if k in SOURCES]
    ts = max((f.stat().st_mtime for f in files if f.exists()), default=0)
    return datetime.date.fromtimestamp(ts).isoformat() if ts else "unknown"


# ------------------------------------------------------------------ contract endpoints
@app.get("/health")
def health():
    return {"status": "ok" if CUBES else "no-data", "data_version": _version(), "data_folder": str(DATA_DIR),
            "variables": {k: f"{c.months[0]}..{c.months[-1]}" for k, c in CUBES.items()},
            "grids_ready": sorted(f"{k}:{kind}" for kind, k in _grids)}


@app.get("/variables")
def variables():
    out = []
    for k in CUBES:
        s = DATASETS[k]
        out.append({"id": k, "name": s["name"], "short": s["short"], "unit": s["unit"], "decimals": s["decimals"], "landOnly": s["landOnly"],
                    "start": first_year(k), "source": s["source"], "up": s["up"], "down": s["down"]})
    return out


@app.get("/regions")
def regions():
    return REGIONS


@app.get("/trend-grid")
def trend_grid(var: str = Query(..., description="variable id, e.g. ndvi")):
    return cached_grid("trend", var)


@app.get("/value-grid")
def value_grid(var: str = Query(..., description="variable id, e.g. ndvi")):
    return cached_grid("value", var)


@app.get("/series")
def series(var: str, lat: float | None = None, lon: float | None = None, region: str | None = None):
    vals, loc = _series(var, _r2(lat), _r2(lon), region)
    c, s = CUBES[var], DATASETS[var]
    end = f"{last_full_year(var)}-12"
    pts = [p for p in core.to_month_list(c.months, vals, first_year(var), decimals=s["decimals"] + 2) if p["t"] <= end]
    if sum(p["v"] is not None for p in pts) < 24:
        raise HTTPException(404, f"Not enough {s['short']} data at this location")
    return {"variable": var, "location": loc, "source": s["source"], "version": _version(), "series": pts}


@app.get("/forecast")
def forecast(var: str, horizon: int = Query(10, ge=1, le=30), lat: float | None = None, lon: float | None = None, region: str | None = None):
    vals, loc = _series(var, _r2(lat), _r2(lon), region)
    c = CUBES[var]
    hist = [h for h in core.annual_means(c.months, vals, first_year(var)) if h[0] <= last_full_year(var)]
    try:
        out = core.forecast(hist, horizon)
    except ValueError as e:
        raise HTTPException(404, str(e))
    out["variable"] = var
    out["location"] = loc
    return out


@app.get("/scenarios")
def scenarios(var: str, lat: float | None = None, lon: float | None = None, region: str | None = None):
    # No NEX-GDDP-CMIP6 file yet: the contract says return null and the website shows a note.
    cube(var)
    return None


@app.get("/hotspots")
def hotspots():
    out = []
    for region, pair, fallback, driver in HOTSPOTS:
        use = pair if all(v in CUBES for v in pair) else fallback
        if not use or not all(v in CUBES for v in use):
            continue
        r = REGION_BY_ID[region]
        parts, score = [], 0.0
        for v in use:
            try:
                vals, _ = _series(v, None, None, region)
            except HTTPException:
                parts = []
                break
            ann = [a for a in core.annual_means(CUBES[v].months, vals, first_year(v)) if a[0] <= last_full_year(v)]
            t = core.mk_sen([a for a, _ in ann], [b for _, b in ann])
            s = DATASETS[v]
            label, (up, down) = PHRASES.get(v, (s["short"], ("is rising", "is falling")))
            if t["p"] < 0.05 and np.isfinite(t["slope"]):
                unit = f" {s['unit']}" if s["unit"] else ""
                ptxt = "< 0.001" if t["p"] < 0.001 else f"= {t['p']:.3f}"
                parts.append(f"{label} {up if t['slope'] > 0 else down} ({t['slope'] * 10:+.{s['decimals']}f}{unit} per decade, p {ptxt})")
                score += 1
            else:
                parts.append(f"{label} shows no clear trend (p = {t['p']:.2f})")
        if not parts:
            continue
        low = lambda x: x if x[:2].isupper() else x[0].lower() + x[1:]  # keep acronyms like NDVI
        summary = f"{parts[0]}, while {low(parts[1])}."
        out.append({"id": f"h-{region}", "region": region, "name": r["name"], "lat": r["lat"], "lon": r["lon"], "pair": use,
                    "driver": f"Possible driver: {driver}", "summary": summary, "_score": score})
    out.sort(key=lambda h: -h["_score"])
    if sum(h["_score"] > 0 for h in out) >= 3:  # only show places where something is actually changing
        out = [h for h in out if h["_score"] > 0]
    for h in out:
        h.pop("_score")
    return out


# ------------------------------------------------------------------ Ibrahim's original endpoints (unchanged URLs and keys)
_raw: dict[str, xr.Dataset] = {}


def _raw_ds(key: str) -> xr.Dataset:
    if key not in _raw:
        path = find_file(key)
        if path is None:
            raise HTTPException(404, f"No {key} file in {DATA_DIR} (expected {' or '.join(DATASETS[key]['files'])})")
        _raw[key] = xr.open_dataset(path)
    return _raw[key]


def _point(key: str, lat: float, lon: float, time: datetime.datetime) -> float | None:
    ds = _raw_ds(key)
    var = _pick(list(ds.data_vars), DATASETS[key]["vars"]) or next(v for v in ds.data_vars if ds[v].ndim == 3)
    data = ds[var].sel(time=pd.to_datetime(time), lat=lat, lon=lon, method="nearest")
    return _clean(float(data.values))


@app.get("/")
def root():
    return {"message": "EarthScope API", "docs": "/docs", "variables": list(CUBES)}


@app.get("/tempanomaly")
def tempanomaly(lat: float, lon: float, time: datetime.datetime):
    return {"temperature anomaly": _point("temp", lat, lon, time), "lat": lat, "lon": lon}


@app.get("/precipitation")
def precipitation(lat: float, lon: float, time: datetime.datetime):
    return {"preci": _point("precip", lat, lon, time), "lat": lat, "lon": lon}


@app.get("/ndvi")
def ndvi(lat: float, lon: float, time: datetime.datetime):
    return {"ndvi": _point("ndvi", lat, lon, time), "lat": lat, "lon": lon}
