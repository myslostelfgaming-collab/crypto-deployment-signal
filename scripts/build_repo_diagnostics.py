#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(".")
DATA_DIR = REPO_ROOT / "data"
DIAG_DIR = DATA_DIR / "diagnostics"

WATCH_FILES = [
    "data/hourly_signal.json",
    "data/hourly_report.json",
    "data/ai_handoff_v1.json",
    "data/features/features_v1.csv",
    "data/labels/labels_v1.csv",
    "data/labels/labels_v2.csv",
    "data/model/model_dataset_v1.csv",
    "data/model/similarity_forecast_v1.json",
    "data/model/similarity_forecast_v2.json",
    "data/model/similarity_forecast_v2_multi.json",
    "data/model/similarity_forecast_v2_ETH-USDT.json",
    "data/model/similarity_forecast_v2_BTC-USDT.json",
    "data/model/predictions_v1.csv",
    "data/model/performance_summary_v1.json",
    "data/forecast/baseline_probs_v1.csv",
    "data/forecast/baseline_summary_v1.json",
    "data/model/actionability_v1.json",
]

ASSETS = ["ETH-USDT", "BTC-USDT", "ETH-BTC"]
HORIZONS = [24, 48, 168, 336]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def file_size_human(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 ** 2):.2f} MB"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_csv_rows(path: Path) -> Dict[str, Any]:
    result = {
        "exists": path.exists(),
        "rows_excluding_header": 0,
        "columns": [],
        "latest_timestamp_guess": None,
        "error": None,
    }

    if not path.exists():
        return result

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            result["columns"] = reader.fieldnames or []
            rows = 0
            latest_guess = None

            timestamp_candidates = [
                "created_at_utc",
                "as_of_utc",
                "timestamp_utc",
                "ts_utc",
                "entry_ts_utc",
                "target_ts_utc",
                "date",
            ]

            for row in reader:
                rows += 1
                for col in timestamp_candidates:
                    val = row.get(col)
                    if val:
                        latest_guess = val

            result["rows_excluding_header"] = rows
            result["latest_timestamp_guess"] = latest_guess

    except Exception as e:
        result["error"] = str(e)

    return result


def preview_csv(path: Path, max_rows: int = 5) -> Dict[str, Any]:
    result = {
        "exists": path.exists(),
        "columns": [],
        "tail_rows": [],
        "error": None,
    }

    if not path.exists():
        return result

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            result["columns"] = reader.fieldnames or []
            buffer: List[Dict[str, Any]] = []

            for row in reader:
                buffer.append(dict(row))
                if len(buffer) > max_rows:
                    buffer.pop(0)

            result["tail_rows"] = buffer

    except Exception as e:
        result["error"] = str(e)

    return result


def build_file_sizes() -> Dict[str, Any]:
    files = []

    for rel in WATCH_FILES:
        path = REPO_ROOT / rel
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0

        info = {
            "path": rel,
            "exists": exists,
            "size_bytes": size_bytes,
            "size_human": file_size_human(size_bytes) if exists else None,
        }

        if rel.endswith(".csv"):
            info["csv"] = count_csv_rows(path)

        files.append(info)

    # Also list the largest files under data/.
    largest = []
    if DATA_DIR.exists():
        for path in DATA_DIR.rglob("*"):
            if path.is_file():
                size_bytes = path.stat().st_size
                largest.append(
                    {
                        "path": str(path).replace("\\", "/"),
                        "size_bytes": size_bytes,
                        "size_human": file_size_human(size_bytes),
                    }
                )

    largest.sort(key=lambda x: x["size_bytes"], reverse=True)

    return {
        "schema": "file_sizes_v1",
        "generated_at_utc": utc_now(),
        "watched_files": files,
        "largest_data_files_top20": largest[:20],
    }


def extract_hourly_report_status() -> Dict[str, Any]:
    report = read_json_safe(DATA_DIR / "hourly_report.json")

    if report is None:
        return {
            "exists": False,
            "error": "Missing or unreadable data/hourly_report.json",
        }

    market_state = report.get("market_state", {})
    integrity = report.get("integrity", {})
    source_files = report.get("meta", {}).get("source_files", {})

    return {
        "exists": True,
        "generated_at_utc": report.get("meta", {}).get("generated_at_utc"),
        "published_at_utc": market_state.get("published_at_utc"),
        "published_at_local": market_state.get("published_at_local"),
        "timezone": market_state.get("timezone"),
        "source_files": source_files,
        "integrity": integrity,
        "assets_present": {
            "eth_usdt": bool(market_state.get("eth_usdt")),
            "btc_usdt": bool(market_state.get("btc_usdt")),
            "eth_btc": bool(market_state.get("eth_btc")),
        },
    }


def prediction_summary() -> Dict[str, Any]:
    path = DATA_DIR / "model" / "predictions_v1.csv"

    summary: Dict[str, Any] = {
        "schema": "prediction_summary_v1",
        "generated_at_utc": utc_now(),
        "path": str(path).replace("\\", "/"),
        "exists": path.exists(),
        "total_rows": 0,
        "by_asset": {},
        "by_status": {},
        "by_horizon": {},
        "latest_created_at_utc": None,
        "latest_entry_ts_utc": None,
        "error": None,
    }

    if not path.exists():
        return summary

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                summary["total_rows"] += 1

                asset = row.get("asset") or "UNKNOWN"
                status = row.get("status") or "UNKNOWN"
                horizon = row.get("horizon_h") or "UNKNOWN"

                summary["by_asset"].setdefault(asset, 0)
                summary["by_asset"][asset] += 1

                summary["by_status"].setdefault(status, 0)
                summary["by_status"][status] += 1

                summary["by_horizon"].setdefault(horizon, 0)
                summary["by_horizon"][horizon] += 1

                if row.get("created_at_utc"):
                    summary["latest_created_at_utc"] = row.get("created_at_utc")

                if row.get("entry_ts_utc"):
                    summary["latest_entry_ts_utc"] = row.get("entry_ts_utc")

    except Exception as e:
        summary["error"] = str(e)

    return summary


def model_readiness() -> Dict[str, Any]:
    model_path = DATA_DIR / "model" / "model_dataset_v1.csv"
    predictions = prediction_summary()

    readiness: Dict[str, Any] = {
        "schema": "model_readiness_v1",
        "generated_at_utc": utc_now(),
        "method_note": (
            "Readiness is diagnostic only. It checks whether data/prediction rows exist. "
            "It does not yet prove forecast accuracy."
        ),
        "assets": {},
        "model_dataset": count_csv_rows(model_path),
    }

    # Read model dataset coverage if available.
    coverage: Dict[str, Dict[str, Any]] = {}

    if model_path.exists():
        try:
            with model_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    asset = row.get("asset") or "UNKNOWN"
                    coverage.setdefault(
                        asset,
                        {
                            "rows": 0,
                            "label_counts_by_horizon": {str(h): 0 for h in HORIZONS},
                        },
                    )

                    coverage[asset]["rows"] += 1

                    for h in HORIZONS:
                        field = f"close_change_pct_{h}"
                        if row.get(field) not in (None, ""):
                            coverage[asset]["label_counts_by_horizon"][str(h)] += 1

        except Exception as e:
            readiness["model_dataset"]["coverage_error"] = str(e)

    # Prediction coverage.
    pred_by_asset = predictions.get("by_asset", {})

    for asset in ASSETS:
        asset_cov = coverage.get(
            asset,
            {
                "rows": 0,
                "label_counts_by_horizon": {str(h): 0 for h in HORIZONS},
            },
        )

        labels_by_h = asset_cov.get("label_counts_by_horizon", {})

        horizon_status = {}
        for h in HORIZONS:
            label_count = int(labels_by_h.get(str(h), 0) or 0)

            if label_count >= 100:
                status = "ready"
            elif label_count >= 30:
                status = "limited"
            elif label_count > 0:
                status = "early"
            else:
                status = "not_ready"

            horizon_status[str(h)] = {
                "label_rows": label_count,
                "status": status,
            }

        readiness["assets"][asset] = {
            "model_rows": asset_cov.get("rows", 0),
            "prediction_rows": pred_by_asset.get(asset, 0),
            "horizons": horizon_status,
            "active_prediction_asset": pred_by_asset.get(asset, 0) > 0,
        }

    return readiness


def latest_rows_preview() -> Dict[str, Any]:
    previews = {}

    for rel in WATCH_FILES:
        if rel.endswith(".csv"):
            previews[rel] = preview_csv(REPO_ROOT / rel, max_rows=5)

    return {
        "schema": "latest_rows_preview_v1",
        "generated_at_utc": utc_now(),
        "previews": previews,
    }


def repo_status(
    file_sizes_payload: Dict[str, Any],
    readiness_payload: Dict[str, Any],
    prediction_payload: Dict[str, Any],
) -> Dict[str, Any]:
    hourly = extract_hourly_report_status()

    watched_missing = [
        f["path"]
        for f in file_sizes_payload.get("watched_files", [])
        if not f.get("exists")
    ]

    largest = file_sizes_payload.get("largest_data_files_top20", [])
    largest_file = largest[0] if largest else None

    return {
        "schema": "repo_status_v1",
        "generated_at_utc": utc_now(),
        "summary": {
            "hourly_report_available": hourly.get("exists", False),
            "watched_files_missing_count": len(watched_missing),
            "prediction_rows_total": prediction_payload.get("total_rows", 0),
            "prediction_assets": prediction_payload.get("by_asset", {}),
            "largest_data_file": largest_file,
        },
        "hourly_report_status": hourly,
        "missing_watched_files": watched_missing,
        "model_readiness_brief": readiness_payload.get("assets", {}),
    }


def main() -> None:
    ensure_dirs()

    file_sizes_payload = build_file_sizes()
    prediction_payload = prediction_summary()
    readiness_payload = model_readiness()
    preview_payload = latest_rows_preview()
    status_payload = repo_status(
        file_sizes_payload=file_sizes_payload,
        readiness_payload=readiness_payload,
        prediction_payload=prediction_payload,
    )

    write_json(DIAG_DIR / "file_sizes.json", file_sizes_payload)
    write_json(DIAG_DIR / "prediction_summary.json", prediction_payload)
    write_json(DIAG_DIR / "model_readiness.json", readiness_payload)
    write_json(DIAG_DIR / "latest_rows_preview.json", preview_payload)
    write_json(DIAG_DIR / "repo_status.json", status_payload)

    print("Repo diagnostics built:")
    print(f"- {DIAG_DIR / 'repo_status.json'}")
    print(f"- {DIAG_DIR / 'file_sizes.json'}")
    print(f"- {DIAG_DIR / 'model_readiness.json'}")
    print(f"- {DIAG_DIR / 'prediction_summary.json'}")
    print(f"- {DIAG_DIR / 'latest_rows_preview.json'}")


if __name__ == "__main__":
    main()
