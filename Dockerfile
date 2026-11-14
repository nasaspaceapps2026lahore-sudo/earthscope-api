FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
RUN uv pip install --system "fastapi[standard]>=0.162.0" h5netcdf h5py netcdf4 numpy pandas xarray
COPY main.py earthscope_core.py API_CONTRACT.md ./
COPY src ./src
EXPOSE 8000
CMD fastapi run main.py --host 0.0.0.0 --port ${PORT:-8000}
