"""EarthScope backend core: everything that is maths, nothing that is web.

Works on plain numpy arrays so it can be tested without the data files:
  Cube  - one variable on a regular lat/lon grid, one value per month
  series, box averages, global (area-weighted) means
  Mann-Kendall + Sen's slope on annual means with lag-1 autocorrelation
  correction (the same method as the website's js/stats.js)
  trend grid, value grid
  forecast ensemble with rolling-origin back-tests and split-conformal bands

Only numpy, pandas and the standard library are used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_MONTHS_PER_YEAR = 9  # same rule as js/stats.js annualMeans


# ---------------------------------------------------------------- data cube
@dataclass
class Cube:
    """Monthly gridded data. data has shape (time, lat, lon); lat and lon ascending, lon in -180..180."""
    var: str
    lat: np.ndarray
    lon: np.ndarray
    months: pd.PeriodIndex  # one entry per month, continuous, no gaps
    data: np.ndarray  # float32, NaN = missing
    res: float = field(init=False)

    def __post_init__(self):
        self.res = float(abs(np.median(np.diff(self.lat)))) if len(self.lat) > 1 else 1.0
        self.weights = np.cos(np.deg2rad(self.lat)).clip(0, None)  # area weight per latitude row

    # -------- indexing
    def nearest(self, lat: float, lon: float) -> tuple[int, int]:
        lon = ((lon + 180) % 360) - 180
        i = int(np.abs(self.lat - lat).argmin())
        d = np.abs(self.lon - lon)
        d = np.minimum(d, 360 - d)  # wrap at the date line
        return i, int(d.argmin())

    def _valid_cell(self, i: int, j: int) -> bool:
        return bool(np.isfinite(self.data[:, i, j]).sum() >= 24)

    def nearest_valid(self, lat: float, lon: float, max_ring: int = 2) -> tuple[int, int, int]:
        """Nearest cell with data. Coastal points on land-only variables fall in the sea, so look up to
        max_ring cells away. Returns (i, j, ring) or raises LookupError."""
        i0, j0 = self.nearest(lat, lon)
        if self._valid_cell(i0, j0):
            return i0, j0, 0
        nlat, nlon = len(self.lat), len(self.lon)
        for ring in range(1, max_ring + 1):
            best = None
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if max(abs(di), abs(dj)) != ring:
                        continue
                    i, j = i0 + di, (j0 + dj) % nlon
                    if 0 <= i < nlat and self._valid_cell(i, j):
                        dist = di * di + dj * dj
                        if best is None or dist < best[0]:
                            best = (dist, i, j)
            if best:
                return best[1], best[2], ring
        raise LookupError("no data near this point")

    # -------- series
    def point(self, lat: float, lon: float) -> tuple[np.ndarray, dict]:
        i, j, ring = self.nearest_valid(lat, lon)
        note = f"nearest {self.res:g}° cell" if ring == 0 else f"nearest {self.res:g}° cell with data ({ring} cell{'s' if ring > 1 else ''} away)"
        return self.data[:, i, j].astype(float), {"lat": round(float(self.lat[i]), 2), "lon": round(float(self.lon[j]), 2), "note": note}

    def box(self, lat: float, lon: float, half: int = 1) -> tuple[np.ndarray, dict]:
        """Area-weighted mean of a (2*half+1)^2 box of cells around a point."""
        i0, j0 = self.nearest(lat, lon)
        rows = [i for i in range(i0 - half, i0 + half + 1) if 0 <= i < len(self.lat)]
        cols = [(j0 + d) % len(self.lon) for d in range(-half, half + 1)]
        sub = self.data[:, rows][:, :, cols]
        w = np.broadcast_to(self.weights[rows][:, None], sub.shape[1:])
        out = _weighted_nanmean(sub, w)
        if np.isfinite(out).sum() < 24:  # e.g. a box that is all sea for NDVI
            return self.point(lat, lon)
        size = self.res * (2 * half + 1)
        return out, {"lat": round(float(self.lat[i0]), 2), "lon": round(float(self.lon[j0]), 2), "note": f"{size:g}° × {size:g}° box average"}

    def global_mean(self) -> np.ndarray:
        w = np.broadcast_to(self.weights[:, None], self.data.shape[1:])
        out = _weighted_nanmean(self.data, w)
        # months where most of the usual area is missing are not a fair global mean
        cover = (np.isfinite(self.data) * w).sum(axis=(1, 2))
        typical = np.nanmedian(np.where(cover > 0, cover, np.nan)) if np.any(cover > 0) else 0
        out[cover < 0.6 * typical] = np.nan
        return out

    # -------- annual means for every cell
    def annual_grid(self, start_year: int | None = None, coarsen: int = 1):
        data, lat, lon = self.data, self.lat, self.lon
        if coarsen > 1:
            data, lat, lon = coarsen_grid(data, lat, lon, coarsen, self.weights)
        years = np.array(sorted(set(self.months.year)))
        full = [y for y in years if (self.months.year == y).sum() == 12 and (start_year is None or y >= start_year)]
        ann = np.full((len(full), len(lat), len(lon)), np.nan, dtype=np.float32)
        for k, y in enumerate(full):
            block = data[self.months.year == y]
            n = np.isfinite(block).sum(axis=0)
            m = _nanmean0(block)
            ann[k] = np.where(n >= MIN_MONTHS_PER_YEAR, m, np.nan)
        return np.array(full), ann, lat, lon


def _nanmean0(a: np.ndarray) -> np.ndarray:
    """nanmean over axis 0 without the 'mean of empty slice' warning."""
    ok = np.isfinite(a)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, a, 0).sum(axis=0) / n, np.nan)


def _weighted_nanmean(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    num = np.where(ok, a, 0) * w
    den = ok * w
    s = den.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, num.sum(axis=(-2, -1)) / s, np.nan)


def coarsen_grid(data, lat, lon, f, weights):
    """Average f×f blocks of cells (area-weighted, NaN-aware)."""
    nl, nm = (len(lat) // f) * f, (len(lon) // f) * f
    d = data[:, :nl, :nm].reshape(data.shape[0], nl // f, f, nm // f, f)
    w = np.broadcast_to(weights[:nl].reshape(nl // f, f)[:, :, None, None], (nl // f, f, nm // f, f))
    ok = np.isfinite(d)
    num = (np.where(ok, d, 0) * w).sum(axis=(2, 4))
    den = (ok * w).sum(axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.25 * w.sum(axis=(1, 3)), num / den, np.nan).astype(np.float32)
    return out, lat[:nl].reshape(-1, f).mean(axis=1), lon[:nm].reshape(-1, f).mean(axis=1)


# ---------------------------------------------------------------- monthly series helpers
def to_month_list(months: pd.PeriodIndex, values: np.ndarray, start_year: int | None = None, decimals: int = 4):
    out = []
    for p, v in zip(months, values):
        if start_year and p.year < start_year:
            continue
        out.append({"t": f"{p.year:04d}-{p.month:02d}", "v": None if not np.isfinite(v) else round(float(v), decimals)})
    return out


def annual_means(months: pd.PeriodIndex, values: np.ndarray, start_year: int | None = None):
    """[(year, mean)] for years with at least 9 valid months (same rule as the website)."""
    df = pd.DataFrame({"y": months.year, "v": values})
    if start_year:
        df = df[df.y >= start_year]
    g = df.dropna().groupby("y")["v"]
    keep = g.count() >= MIN_MONTHS_PER_YEAR
    m = g.mean()[keep]
    return [(int(y), float(v)) for y, v in m.items()]


# ---------------------------------------------------------------- statistics
def p_two_sided(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _detrend(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 3:
        return x
    i = np.arange(n, dtype=float)
    b = ((i - i.mean()) * (x - x.mean())).sum() / ((i - i.mean()) ** 2).sum()
    return x - (x.mean() + b * (i - i.mean()))


def _lag1(x: np.ndarray) -> float:
    m = x.mean()
    den = ((x - m) ** 2).sum()
    return float(((x[1:] - m) * (x[:-1] - m)).sum() / den) if den else 0.0


def mk_sen(years, values) -> dict:
    """Mann-Kendall + Sen's slope on annual values, variance inflated for lag-1 autocorrelation.
    Mirrors S.seasonalTrend in the website's js/stats.js. slope is per year."""
    t = np.asarray(years, dtype=float)
    x = np.asarray(values, dtype=float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    n = len(x)
    if n < 4:
        return {"slope": float("nan"), "p": 1.0, "n": n, "z": 0.0, "lo": float("nan"), "hi": float("nan")}
    iu, ju = np.triu_indices(n, 1)
    s = float(np.sign(x[ju] - x[iu]).sum())
    r1 = min(0.9, max(0.0, _lag1(_detrend(x))))
    corr = (1 + r1) / (1 - r1)
    var = n * (n - 1) * (2 * n + 5) / 18 * corr
    z = 0.0 if s == 0 else (s - math.copysign(1, s)) / math.sqrt(var)
    slopes = np.sort((x[ju] - x[iu]) / (t[ju] - t[iu]))
    N, C = len(slopes), 1.96 * math.sqrt(var)
    lo = slopes[max(0, round((N - C) / 2) - 1)]
    hi = slopes[min(N - 1, round((N + C) / 2))]
    return {"slope": float(np.median(slopes)), "p": p_two_sided(z), "n": n, "z": z, "lo": float(lo), "hi": float(hi), "r1": r1}


def trend_grid(cube: Cube, start_year: int | None, target_res: float = 4.0, land_only: bool = False) -> dict:
    f = max(1, int(round(target_res / cube.res)))
    years, ann, lat, lon = cube.annual_grid(start_year, coarsen=f)
    cells = []
    for i in range(len(lat)):
        for j in range(len(lon)):
            x = ann[:, i, j]
            if np.isfinite(x).sum() < max(8, int(0.6 * len(years))):
                continue  # too little data for an honest trend (also drops sea cells for land-only data)
            r = mk_sen(years, x)
            if not np.isfinite(r["slope"]):
                continue
            cells.append({"lat": round(float(lat[i]), 2), "lon": round(float(lon[j]), 2), "slope": float(f"{r['slope'] * 10:.4g}"), "p": round(r["p"], 4)})
    return {"variable": cube.var, "resolution": round(cube.res * f, 4), "units": "per decade",
            "period": f"{int(years[0])}–{int(years[-1])}" if len(years) else "", "cells": cells}


def value_grid(cube: Cube, years_back: int = 10, label: str = "mean", decimals: int = 3) -> dict:
    full = sorted({y for y in set(cube.months.year) if (cube.months.year == y).sum() == 12})
    use = full[-years_back:] if full else sorted(set(cube.months.year))[-years_back:]
    block = cube.data[np.isin(cube.months.year, use)]
    n = np.isfinite(block).sum(axis=0)
    mean = _nanmean0(block)
    m = np.where(n >= max(6, 3 * len(use)), mean, np.nan)  # at least ~3 valid months a year
    cells = [{"lat": round(float(cube.lat[i]), 2), "lon": round(float(cube.lon[j]), 2), "value": round(float(m[i, j]), decimals)}
             for i in range(len(cube.lat)) for j in range(len(cube.lon)) if np.isfinite(m[i, j])]
    return {"variable": cube.var, "resolution": round(cube.res, 4), "period": f"{use[0]}–{use[-1]} {label}", "cells": cells}


# ---------------------------------------------------------------- forecasting
def _fc_climatology(y, h):
    return np.full(h, np.mean(y[-10:]))


def _fc_persistence(y, h):
    return np.full(h, y[-1])


def _fc_linear(y, h, window=30):
    w = y[-window:]
    t = np.arange(len(w), dtype=float)
    b, a = np.polyfit(t, w, 1)
    return a + b * (len(w) - 1 + np.arange(1, h + 1))


def _fc_sen(y, h, window=30):
    w = y[-window:]
    n = len(w)
    iu, ju = np.triu_indices(n, 1)
    b = np.median((w[ju] - w[iu]) / (ju - iu))
    a = np.median(w - b * np.arange(n))
    return a + b * (n - 1 + np.arange(1, h + 1))


def _fc_holt(y, h):
    """Damped-trend exponential smoothing; parameters picked by one-step in-sample error."""
    best = None
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for beta in (0.05, 0.15, 0.3):
            for phi in (0.8, 0.9, 0.98):
                lvl, tr, sse = y[0], y[1] - y[0], 0.0
                for v in y[1:]:
                    pred = lvl + phi * tr
                    sse += (v - pred) ** 2
                    new = alpha * v + (1 - alpha) * pred
                    tr = beta * (new - lvl) + (1 - beta) * phi * tr
                    lvl = new
                if best is None or sse < best[0]:
                    best = (sse, lvl, tr, phi)
    _, lvl, tr, phi = best
    return np.array([lvl + tr * sum(phi ** k for k in range(1, i + 1)) for i in range(1, h + 1)])


MODELS = [
    ("Climatology (last 10 yr)", _fc_climatology),
    ("Persistence (last year)", _fc_persistence),
    ("Linear trend (30 yr)", _fc_linear),
    ("Sen–Theil trend (30 yr)", _fc_sen),
    ("Damped Holt smoothing", _fc_holt),
]


def _conformal_q(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(scores)[k - 1])


def forecast(history: list[tuple[int, float]], horizon: int = 10, min_train: int = 12, skill_leads: int = 5) -> dict:
    """Weighted ensemble of simple models on annual means.
    Weights = 1 / back-test MAE for models that beat climatology (skill > 0).
    Bands = split-conformal: residuals of the ensemble back-tests scaled by sqrt(lead)."""
    years = np.array([y for y, _ in history])
    y = np.array([v for _, v in history], dtype=float)
    if len(y) < 12:
        raise ValueError("need at least 12 years of data to forecast")
    min_train = min(min_train, max(8, len(y) - 6))  # short records (e.g. NDVI from 2010) still get >= 6 back-tests
    horizon = int(max(1, min(30, horizon)))
    origins = list(range(min_train, len(y)))  # train on y[:o], predict y[o:]
    preds = {name: [] for name, _ in MODELS}
    for o in origins:
        for name, fn in MODELS:
            preds[name].append(fn(y[:o], horizon))

    def errors(name, max_lead):
        e = []
        for k, o in enumerate(origins):
            for L in range(1, max_lead + 1):
                if o + L - 1 < len(y):
                    e.append(abs(preds[name][k][L - 1] - y[o + L - 1]))
        return np.array(e)

    mae = {name: float(errors(name, skill_leads).mean()) for name, _ in MODELS}
    clim = mae[MODELS[0][0]]
    skill = {n: (1 - m / clim) if clim > 0 else 0.0 for n, m in mae.items()}
    raw = {n: (1 / mae[n] if skill[n] > 0 and mae[n] > 0 else 0.0) for n in mae}
    tot = sum(raw.values())
    weights = {n: (w / tot if tot else (1.0 if n == MODELS[0][0] else 0.0)) for n, w in raw.items()}

    ens = [sum(weights[n] * preds[n][k] for n in weights) for k in range(len(origins))]
    # normalised residuals by lead
    res = []  # (origin index, lead, residual)
    for k, o in enumerate(origins):
        for L in range(1, horizon + 1):
            if o + L - 1 < len(y):
                res.append((k, L, y[o + L - 1] - ens[k][L - 1]))
    res = np.array(res) if res else np.zeros((0, 3))
    score = np.abs(res[:, 2]) / np.sqrt(res[:, 1]) if len(res) else np.array([])
    # split: calibrate on the earlier origins, measure coverage on the later ones
    cut = len(origins) // 2
    cal, test = res[:, 0] < cut, res[:, 0] >= cut
    q90_cal = _conformal_q(score[cal], 0.9)
    coverage = float(np.mean(score[test] <= q90_cal)) if test.any() and np.isfinite(q90_cal) else None
    q90, q50 = _conformal_q(score, 0.9), _conformal_q(score, 0.5)

    final = {n: fn(y, horizon) for n, fn in MODELS}
    med = sum(weights[n] * final[n] for n in weights)
    leads = np.sqrt(np.arange(1, horizon + 1))
    r = lambda a: [float(f"{v:.5g}") for v in a]
    bt = []
    for k in range(len(origins)):
        if k % max(1, len(origins) // 6):
            continue  # a handful of back-tests is enough to draw
        o = origins[k]
        L = min(5, len(y) - o, horizon)
        if L >= 2:
            bt.append({"origin": int(years[o - 1]), "years": [int(years[o - 1]) + i for i in range(1, L + 1)], "pred": r(ens[k][:L])})
    return {
        "history": [{"year": int(a), "v": float(f"{b:.5g}")} for a, b in history],
        "years": [int(years[-1]) + i for i in range(1, horizon + 1)],
        "median": r(med), "p05": r(med - q90 * leads), "p25": r(med - q50 * leads), "p75": r(med + q50 * leads), "p95": r(med + q90 * leads),
        "models": [{"name": n, "weight": round(weights[n], 3), "mae": float(f"{mae[n]:.4g}"), "skill": round(skill[n], 3)} for n, _ in MODELS],
        "coverage90": None if coverage is None else round(coverage, 3),
        "backtests": bt,
        "method": "Weighted ensemble of 5 simple models on annual means; weights = 1 / back-test MAE for models that beat climatology; "
                  "90% and 50% bands from split-conformal calibration of rolling-origin back-test errors (scaled by √lead).",
    }


# ---------------------------------------------------------------- cleaning raw file data
_RATE_HR = ("mm/hr", "mm/h", "mmh-1", "mm/hour", "mmhr-1", "mmh^-1")
_RATE_DAY = ("mm/day", "mm/d", "mmd-1", "mmday-1")
_RATE_SEC = ("kgm-2s-1", "kg/m2/s", "kgm^-2s^-1")


def prepare_cube(key: str, values, order, lat, lon, periods: pd.PeriodIndex, attrs: dict, start_year: int, log=None) -> Cube:
    """Turn one variable read from a file into a clean monthly Cube.
    order: positions of (time, lat, lon) in the array's dimensions."""
    warn = (log.warning if log else (lambda *a: None))
    values = np.transpose(np.asarray(values), order).astype("float32")
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if np.nanmax(lon) > 180:  # 0..360 -> -180..180
        lon = ((lon + 180) % 360) - 180
    li, lo = np.argsort(lat), np.argsort(lon)
    values, lat, lon = values[:, li][:, :, lo], lat[li], lon[lo]

    keep = np.asarray(periods.year >= start_year)
    values, periods = values[keep], periods[keep]

    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1e20] = np.nan
    values[values <= -9000] = np.nan
    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is not None and np.isfinite(float(fill)):
        values[values == np.float32(fill)] = np.nan

    if periods.has_duplicates:  # daily / 8-day files -> monthly mean
        uniq = periods.unique().sort_values()
        values = np.stack([_nanmean0(values[np.asarray(periods == p)]) for p in uniq]).astype("float32")
        periods = uniq
    full = pd.period_range(periods.min(), periods.max(), freq="M")
    cube = np.full((len(full),) + values.shape[1:], np.nan, dtype="float32")
    cube[full.get_indexer(periods)] = values

    units = str(attrs.get("units") or attrs.get("units_in_file") or "").lower().replace(" ", "")
    days = full.days_in_month.to_numpy()[:, None, None].astype("float32")
    if key == "precip":  # store as mm/month
        if units in _RATE_HR:
            cube *= days * 24
        elif units in _RATE_DAY:
            cube *= days
        elif units in _RATE_SEC:
            cube *= days * 86400
        elif not units.startswith("mm/m"):
            # units label missing: judge by size (Earth's mean rain is ~0.11 mm/hr = ~2.7 mm/day = ~80 mm/month)
            mean = float(np.nanmean(cube))
            if mean < 1:
                warn("precip units '%s' unknown; values look like mm/hr, converting to mm/month", units)
                cube *= days * 24
            elif mean < 15:
                warn("precip units '%s' unknown; values look like mm/day, converting to mm/month", units)
                cube *= days
    if key == "ndvi":
        if np.nanmax(cube) > 1.5:  # MODIS stores NDVI x 10000
            cube *= 1e-4
        cube[cube < -0.29] = np.nan  # MODIS fill (-3000 x 1e-4)
    if key == "lst" and np.nanmean(cube) > 150:  # Kelvin -> °C
        cube -= 273.15
    return Cube(var=key, lat=lat, lon=lon, months=full, data=cube)
