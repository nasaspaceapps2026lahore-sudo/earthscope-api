# EarthScope API contract

The frontend needs **eight read-only GET endpoints** that return JSON. The backend can be a live server (FastAPI, Flask, Node) or a folder of static JSON files. Either works if the paths and shapes below match.

- All responses are `application/json`, UTF-8.
- If the backend runs on a different domain from the website, it must send `Access-Control-Allow-Origin: *` (CORS).
- Errors: return a non-200 status. The app shows "The data service answered <status>".
- Numbers are plain JSON numbers. A missing month is `null`, never `NaN` or `-9999`.
- Months are strings `"YYYY-MM"`.

Variable ids used everywhere: `ndvi`, `temp`, `landcover`, `precip`, `aq`, `grace`, `lst`.
Location is either `lat` + `lon` (decimal degrees, 2 decimals) **or** `region` (a region id).

---

## 1. `GET /variables`

```json
[
  {
    "id": "ndvi",
    "name": "Vegetation (NDVI)",
    "short": "NDVI",
    "unit": "",
    "decimals": 3,
    "landOnly": true,
    "start": 2001,
    "source": "MODIS MOD13C2 + VIIRS VNP13C2 v2 (stitched)",
    "up": "greening",
    "down": "browning"
  }
]
```

Optional. Anything returned overrides the same field in `js/config.js`.

## 2. `GET /regions`

```json
[
  { "id": "indus", "name": "Indus plains", "lat": 30.5, "lon": 73.5, "blurb": "Irrigated farmland over a heavily pumped aquifer." }
]
```

Region ids are referenced by `/series?region=` and `/hotspots`.

## 3. `GET /trend-grid?var=ndvi`

The precomputed global trend map that drives the 3D globe and the Explore map.

```json
{
  "variable": "ndvi",
  "resolution": 4,
  "units": "per decade",
  "cells": [
    { "lat": -56.0, "lon": -178.0, "slope": 0.0041, "p": 0.31 }
  ]
}
```

- `slope`: Sen's slope **per decade**, in the variable's unit.
- `p`: Mann-Kendall p-value on annual means with autocorrelation correction (uncorrected for multiple tests; the frontend applies Benjamini–Hochberg FDR).
- Cell centres on a regular grid. 2°–5° keeps the globe fast (about 1,000–8,000 cells). The frontend hides ocean cells for land-only variables.

## 3b. `GET /value-grid?var=ndvi`

The average level of each cell, drawn as the coloured layer on the satellite maps (the NDVI 0–1 legend).

```json
{ "variable": "ndvi", "resolution": 2, "period": "2015–2025 mean", "cells": [ { "lat": -59.0, "lon": -179.0, "value": 0.12 } ] }
```

For `temp` and `grace`, `value` is the recent anomaly. Variable ids now also include `landcover` (% forest, MODIS MCD12Q1) and `aq` (aerosol optical depth).

## 4. `GET /series?var=ndvi&lat=30.50&lon=73.50` (or `&region=indus`)

```json
{
  "variable": "ndvi",
  "location": { "lat": 30.5, "lon": 73.5, "name": "Indus plains" },
  "source": "MODIS MOD13C2 + VIIRS VNP13C2 v2",
  "version": "2026-11-15",
  "series": [
    { "t": "2001-01", "v": 0.4312 },
    { "t": "2001-02", "v": null }
  ]
}
```

- **Monthly**, in time order, without skipping months. Missing months are `null`. For example, GRACE has a gap from 2017-07 to 2018-05.
- Raw monthly values, not anomalies, except `temp` and `grace`, which are anomalies already: GISTEMP against 1951–1980, GRACE against its own baseline.
- The frontend computes the Mann-Kendall test, Sen's slope, the 95% range, the decomposition and the noise simulation itself, so the backend does not need to.
- For `lat`/`lon`, return the nearest grid cell, or an average of a small box around it, and say which in `location.name`.

## 5. `GET /forecast?var=temp&horizon=10&lat=..&lon=..` (or `&region=`)

```json
{
  "variable": "temp",
  "history": [ { "year": 1980, "v": 0.21 } ],
  "years":   [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035],
  "median":  [1.33, 1.35],
  "p05":     [1.15, 1.06],
  "p25":     [1.26, 1.24],
  "p75":     [1.40, 1.46],
  "p95":     [1.51, 1.64],
  "models": [
    { "name": "Climatology (last 10 yr)", "weight": 0.0,  "mae": 0.148, "skill": 0.0 },
    { "name": "SARIMAX + ENSO",           "weight": 0.31, "mae": 0.088, "skill": 0.41 },
    { "name": "LightGBM quantile",        "weight": 0.27, "mae": 0.095, "skill": 0.36 }
  ],
  "coverage90": 0.88,
  "backtests": [
    { "origin": 2005, "years": [2006, 2007, 2008, 2009, 2010], "pred": [0.61, 0.63, 0.64, 0.66, 0.68] }
  ],
  "method": "Weighted ensemble; weights = inverse back-test error; bands from split-conformal calibration."
}
```

- `history`: annual means of the observed record.
- `years`, `median`, `p05`, `p25`, `p75` and `p95` all have length `horizon`.
- `models[].skill` = 1 − MAE(model) / MAE(climatology). A model with skill ≤ 0 gets `weight: 0`, and the app shows it as dropped.
- `coverage90`: share of back-test outcomes that fell inside the 90% band.
- `backtests`: the rolling-origin forecasts of the ensemble, drawn on the back-test chart.

## 6. `GET /scenarios?var=temp&lat=..&lon=..` (or `&region=`)

Return `null` for variables without CMIP6 projections. The app then shows a note instead of the chart.

```json
{
  "variable": "temp",
  "unit": "°C vs 1995–2014",
  "models": 34,
  "years": [2015, 2020, 2025, 2100],
  "ssp245": { "median": [0.0, 0.11], "lo": [-0.17, -0.09], "hi": [0.17, 0.31] },
  "ssp585": { "median": [0.0, 0.14], "lo": [-0.27, -0.2],  "hi": [0.27, 0.48] }
}
```

`lo` and `hi` are the spread across NEX-GDDP-CMIP6 models, for example the 10th and 90th percentiles. For precipitation, use `"unit": "% vs 1995–2014"`.

## 7. `GET /hotspots`

The Trend Detective discoveries.

```json
[
  {
    "id": "h-indus",
    "region": "indus",
    "name": "Indus plains",
    "lat": 30.5,
    "lon": 73.5,
    "pair": ["ndvi", "grace"],
    "driver": "Groundwater pumping for irrigation",
    "summary": "Fields are getting greener while the water stored underground keeps falling."
  }
]
```

- `region` must exist in `/regions`. The app loads both variables in `pair` through `/series?region=`.
- Sort by importance; the first one opens by default.

## 8. `GET /health` (optional)

```json
{ "status": "ok", "data_version": "2026-11-15" }
```

---

## Static-file option (no server)

Write each response to a file and set `API_BASE` to the folder:

```
data/variables.json  data/regions.json  data/hotspots.json
data/trend-grid?var=ndvi
```

A static host cannot serve query strings as separate files. If you go static, tell the lead and the paths in `js/api.js` will change to `data/series/<var>/<region>.json`. That is a one-line change in `request()`.

---

## Check your backend automatically

```bash
node tools/check-backend.mjs https://your-backend.example.org
```

It calls every endpoint for every variable, validates each response against this contract, checks the CORS header and flags slow answers. It exits with code 1 if anything fails, so it can run in CI.

You can also point the live website at your backend without editing any file: open `https://<site>/?api=https://your-backend.example.org`. Any response that breaks the contract shows a clear message in the page and the exact problem in the browser console. `?api=reset` switches back.
