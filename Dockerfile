# Ask IIT Ropar: self-contained image (CPU). The search index is built into the image.
#   docker build -t ask-iitropar .
#   docker run --rm -p 7860:7860 -v "$PWD/.env:/app/.env:ro" -v "$PWD/logs:/app/logs" \
#       -e OLLAMA_BASE_URL=http://host.docker.internal:11434 ask-iitropar
# Also runs as-is on a Hugging Face Docker Space (non-root user 1000, port 7860).
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch first (the default Linux wheel pulls several GB of CUDA libraries)
COPY requirements-app.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements-app.txt

COPY data/faculty.csv data/staff.csv data/departments.csv data/committee_members.csv data/calendar.csv \
     data/chunks.jsonl data/corpus.jsonl data/
# Only what indexing needs, so code changes elsewhere don't trigger a 10-minute re-index.
COPY app/__init__.py app/config.py app/index.py app/
RUN mkdir -p logs index && chown -R user /app

USER user
# Downloads the embedding model and builds the Chroma index (~10 min on CPU), so start-up is fast.
RUN python -m app.index

COPY --chown=user app app
COPY --chown=user scraper scraper
COPY --chown=user static static
COPY --chown=user prompts prompts

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["python", "-m", "app"]
