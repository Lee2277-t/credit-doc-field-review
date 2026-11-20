"""Configuration access. All thresholds, schemas and paths come from YAML."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "scenario.yaml"


@lru_cache(maxsize=4)
def load_config(path: Optional[str] = None) -> Dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def doc_types(config: Optional[Dict] = None) -> Dict:
    return (config or load_config())["doc_types"]


def field_specs(doc_type: str, config: Optional[Dict] = None) -> List[Dict]:
    config = config or load_config()
    return list(config["doc_types"].get(doc_type, {}).get("fields", []))


def field_spec(doc_type: str, path: str, config: Optional[Dict] = None) -> Optional[Dict]:
    for spec in field_specs(doc_type, config):
        if spec["path"] == path:
            return spec
    return None


def all_field_labels(doc_type: str, config: Optional[Dict] = None) -> List[str]:
    labels = []
    for spec in field_specs(doc_type, config):
        if "label" in spec:
            labels.append(spec["label"])
        labels.extend(spec.get("aliases", []))
    return sorted(set(labels), key=len, reverse=True)


def risk_threshold(spec: Dict, config: Optional[Dict] = None) -> float:
    config = config or load_config()
    overrides = config["thresholds"].get("field_overrides", {}) or {}
    if spec["path"] in overrides:
        return float(overrides[spec["path"]])
    return float(config["thresholds"]["risk_tiers"][spec["risk"]])


def table_spec(doc_type: str, table_name: str, config: Optional[Dict] = None) -> Optional[Dict]:
    config = config or load_config()
    return config["doc_types"].get(doc_type, {}).get("tables", {}).get(table_name)


def resolve_path(path: str, config: Optional[Dict] = None) -> Path:
    config = config or load_config()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def evaluation_only_splits(config: Optional[Dict] = None) -> Set[str]:
    """SCENARIO 第 108 行：留出集标签只许评估脚本读，调参侧一律拒读。"""
    config = config or load_config()
    names = (config.get("paths") or {}).get("evaluation_only_splits") or []
    return {str(name) for name in names}
