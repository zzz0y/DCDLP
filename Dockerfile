FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG PYG_WHEEL_INDEX=https://data.pyg.org/whl/torch-2.3.1+cu121.html
ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONPATH=/workspace/src \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir --upgrade "pip<25.2" && \
    python -m pip install --no-cache-dir \
      pyg_lib==0.4.0+pt23cu121 \
      torch_scatter==2.1.2+pt23cu121 \
      torch_sparse==0.6.18+pt23cu121 \
      torch_cluster==1.6.3+pt23cu121 \
      torch_spline_conv==1.2.2+pt23cu121 \
      -f "${PYG_WHEEL_INDEX}"

COPY docker/requirements-linux.lock /tmp/requirements-linux.lock
RUN python -m pip install --no-cache-dir -r /tmp/requirements-linux.lock && \
    python -m pip check

# Install a fallback copy of the package for image-level validation. At runtime
# /workspace is the bind-mounted experiment directory and PYTHONPATH makes that
# source tree authoritative, so results and code stay under the requested host path.
WORKDIR /opt/dcdlp-build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps . && \
    rm -rf /opt/dcdlp-build /tmp/requirements-linux.lock

WORKDIR /workspace
CMD ["bash", "scripts/run_full_experiments.sh"]
