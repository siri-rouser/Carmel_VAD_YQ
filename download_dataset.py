from huggingface_hub import snapshot_download

snapshot_download(
    "HannaLicht/overhead-traffic-anomalies",
    repo_type="dataset",
    local_dir="OTA"
)
