import json
import os
from datetime import datetime

LOG_FILE = "logs/rag_logs.jsonl"
os.makedirs("logs", exist_ok=True)


def convert(obj):
    # handle numpy types
    try:
        return float(obj)
    except:
        return str(obj)


def log_event(data: dict):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        **data
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry, default=convert) + "\n")