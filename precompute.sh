uv run --dev notebooks/precompute.py && \
    gcloud storage cp 'data/*' gs://lyrics-analyzer/dashboard-data/
