#!/usr/bin/env python

import os
import shutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Africa/Johannesburg")
HISTORY_ROOT = os.path.join("data", "history")

# Recommended default retention window
RETENTION_DAYS = 180


def is_date_folder(name: str) -> bool:
    # Expect YYYY-MM-DD
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main():
    if not os.path.isdir(HISTORY_ROOT):
        print(f"No history folder found at {HISTORY_ROOT}. Nothing to prune.")
        return

    cutoff_date = (datetime.now(LOCAL_TZ).date() - timedelta(days=RETENTION_DAYS))
    print(f"Pruning history older than {RETENTION_DAYS} days (cutoff: {cutoff_date.isoformat()})")

    removed = 0
    kept = 0

    for name in sorted(os.listdir(HISTORY_ROOT)):
        path = os.path.join(HISTORY_ROOT, name)
        if not os.path.isdir(path):
            continue
        if name == "index.csv":
            continue
        if not is_date_folder(name):
            continue

        folder_date = datetime.strptime(name, "%Y-%m-%d").date()
        if folder_date < cutoff_date:
            shutil.rmtree(path)
            removed += 1
            print(f"REMOVED: {path}")
        else:
            kept += 1

    print(f"Done. Kept {kept} day-folders. Removed {removed} day-folders.")


if __name__ == "__main__":
    main()