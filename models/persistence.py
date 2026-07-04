from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROCESSED_DIR / "models"
LAST_CONFIG_PATH = PROCESSED_DIR / "last_model_config.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _signature_blob(model_name: str, params: dict[str, Any] | None, data_hash: str | None) -> str:
    payload = {
        "model_name": model_name,
        "params": params or {},
        "data_hash": data_hash or "",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_signature(model_name: str, params: dict[str, Any] | None = None, data_hash: str | None = None) -> str:
    return hashlib.sha1(_signature_blob(model_name, params, data_hash).encode("utf-8")).hexdigest()


def model_path(model_name: str, signature: str, ticker: str | None = None) -> Path:
    parts = [_slug(model_name)]
    if ticker:
        parts.append(_slug(ticker.upper()))
    parts.append(signature[:12])
    return MODEL_DIR / ("__".join(parts) + ".joblib")


def save_model(model: Any, model_name: str, params: dict[str, Any] | None = None, data_hash: str | None = None, ticker: str | None = None) -> Path:
    signature = config_signature(model_name, params, data_hash)
    path = model_path(model_name, signature, ticker)
    joblib.dump({"model": model, "signature": signature, "model_name": model_name, "params": params or {}, "data_hash": data_hash}, path)
    return path


def load_model(model_name: str, params: dict[str, Any] | None = None, data_hash: str | None = None, ticker: str | None = None):
    signature = config_signature(model_name, params, data_hash)
    path = model_path(model_name, signature, ticker)
    if not path.exists():
        return None
    payload = joblib.load(path)
    return payload.get("model")


def save_last_config(config: dict[str, Any]) -> Path:
    LAST_CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return LAST_CONFIG_PATH


def load_last_config() -> dict[str, Any] | None:
    if not LAST_CONFIG_PATH.exists():
        return None
    try:
        return json.loads(LAST_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
