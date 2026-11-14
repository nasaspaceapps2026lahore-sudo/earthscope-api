"""Turn the NetCDF inputs into small pre-cleaned .npz grids for free hosting (each well under GitHub's 25 MB web-upload limit).

   uv run python tools/make_npz.py      # writes src/data/input/*_2deg.npz next to the .nc files

The server prefers the .nc file when both exist, and uses the .npz on its own when only that is deployed.
"""
from pathlib import Path

import numpy as np

import main

for key in ("temp", "ndvi", "precip"):
    nc = next((main.DATA_DIR / n for n in main.DATASETS[key]["files"] if (main.DATA_DIR / n).exists()), None)
    if nc is None:
        continue
    c = main.load_cube(key)
    out = nc.with_suffix(".npz")
    np.savez_compressed(out, data=c.data.astype(np.float32), lat=c.lat.astype(np.float32), lon=c.lon.astype(np.float32),
                        months=np.array([str(p) for p in c.months]), var=np.array(c.var))
    print(f"{out.name}: {out.stat().st_size / 1e6:.1f} MB")
