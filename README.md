# EarthScope API

The data service behind the EarthScope website. FastAPI app, built on Ibrahim's original `main.py`.

## Run it
```bash
uv sync                               # installs everything from pyproject.toml
uv run fastapi dev --port 8000        # http://localhost:8000/docs
```
Data files go in `src/data/input/` (or set `EARTHSCOPE_DATA=/path/to/folder`):

| File | Variable in file | Served as |
|---|---|---|
| `temperature_2deg.nc` (or `temperature_data.nc`) | `temperature` (or `tempanomaly`) | `temp` (GISTEMP anomaly, from 1980) |
| `precipitation_2deg.nc` | `precipitation` | `precip` (mm/hr in the file, converted to mm/month) |
| `ndvi_2deg.nc` | `ndvi` | `ndvi` (MOD13C2, 2010–2025) |
| `grace_2deg.nc` *(optional)* | `lwe_thickness` | `grace` (cm), unlocks the Indus groundwater story |
| `landcover_2deg.nc`, `aod_2deg.nc`, `lst_2deg.nc` *(optional)* | `forest`, `aod`, `lst` | `landcover`, `aq`, `lst` |

A dataset is served only if its file is there. The website hides tabs for anything missing.
Any monthly NetCDF with `time`/`lat`/`lon` (or `latitude`/`longitude`) works; 0–360 longitudes, descending latitudes, daily files and MODIS ×10000 NDVI are handled.

On start-up it loads the files (a few seconds) and pre-computes the trend and value grids in the background (`/health` shows when they are ready).

## Endpoints
Website contract (`API_CONTRACT.md`): `/health /variables /regions /hotspots /trend-grid /value-grid /series /forecast /scenarios`.
Original endpoints still work: `/tempanomaly`, `/precipitation`, `/ndvi` (now return `null` instead of crashing on missing values).

## What the maths is
- **Trends:** Mann-Kendall + Sen's slope on annual means (years with ≥ 9 months), variance inflated for lag-1 autocorrelation — the same method as the website (checked to agree to 1e-7). Trend grid at 4°, slope per decade.
- **Forecast:** 5 models (climatology, persistence, linear trend, Sen–Theil trend, damped Holt) back-tested from rolling origins; weights = 1/MAE for models that beat climatology; 50% and 90% bands from split-conformal calibration of back-test errors; `coverage90` measured on held-out origins. **Ibrahim: swap in your prediction model** by adding it to `MODELS` in `earthscope_core.py` — `fn(history_array, horizon) -> array` — and it is back-tested, weighted and banded automatically.
- **Hotspots:** the text is computed from the data each time (trend, direction, p-value), so it never claims something the numbers don't show.
- **Scenarios:** returns `null` until a NEX-GDDP-CMIP6 file is added.

## Test
```bash
uv run python tests/test_core.py                         # maths
node ../earthscope-web/tools/check-backend.mjs http://localhost:8000   # whole contract + CORS
```
Then open the website: `http://localhost:<site port>/index.html` talks to `localhost:8000` automatically.

## Deploy
`docker build -t earthscope-api . && docker run -p 8000:8000 -v $PWD/src/data/input:/app/src/data/input earthscope-api`
(Render, Railway, Fly.io or a Hugging Face Docker Space all work.) Then set `LIVE_API_BASE` in the website's `js/config.js`.

## Tested on the team's real files (14 Nov)
`temperature_2deg.nc` 1880–2026-09, `ndvi_2deg.nc` 2010–2025, `precipitation_2deg.nc` 2010–2025-09.
All 92 test responses pass the website contract. Headline result:
**Indus plains: NDVI +0.028 per decade (p < 0.001) while rainfall shows no trend (p = 0.96).**
