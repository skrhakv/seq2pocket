# Seq2Pocket inference image.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.4 wheels; also runs on CPU-only hosts).
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt

WORKDIR /app
COPY run_seq2pocket.py SASA.py /app/
COPY tutorial/finetuning_utils.py /app/tutorial/finetuning_utils.py

# Replace Biopython's SASA with the repo's version (populates atom.sasa_points,
# required by the surface-point clustering).
RUN cp /app/SASA.py "$(python -c 'import Bio.PDB.SASA as m; print(m.__file__)')"

VOLUME /models
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && ln -s /entrypoint.sh /usr/local/bin/seq2pocket
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
