# BDCM — reproducible CPU runtime (PyTorch + Hydra + Matplotlib).
# Editable install keeps `results/` and Hydra paths rooted at /app (see bdcm.config._repo_root).

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Runtime libs for PyTorch (OpenMP) and common scientific wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer cache: install deps before copying the rest of the tree
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install PyTorch from the CPU index first. Default PyPI `torch` on aarch64/x86_64
# often pulls CUDA + cuDNN (~800MB+), which makes builds very slow for a CPU-only image.
RUN pip install --upgrade pip setuptools wheel \
    && pip install "torch>=2.11.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e .

RUN useradd --create-home --uid 10001 --shell /bin/bash bdcm \
    && mkdir -p /app/results /app/outputs \
    && chown -R bdcm:bdcm /app

USER bdcm

# Default: fast sanity preset (headless-safe; no GUI)
ENTRYPOINT ["python", "-m", "bdcm.experiments"]
CMD ["experiment=sanity", "scm=1", "variant=simple"]
