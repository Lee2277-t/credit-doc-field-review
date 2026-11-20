"""Fit the confidence calibrator on the dev split.

Samples are grouped by document for cross-validation so that pages from the
same document never leak across folds. The artifact stores the logistic model,
the isotonic mapping fitted on out-of-fold predictions, and sample counts.

``--probe`` additionally fits a model that includes synthetic negatives
(near-miss values / shifted grounding / clearly wrong values) and writes a
reliability curve, so the ceiling of the current feature set is measurable
instead of assumed.

``--review`` folds the human review ledger (``out/<split>/review/*.review.jsonl``)
back into the sample pool: ``accept`` says the automatic value was right, so the
field is a positive; ``correct`` means the reviewer had to retype it, so the same
feature vector is a negative. Features are read from the stored field payload,
i.e. the exact ``signals`` the decision was made with, so no re-parse is needed.
The gate ``calibration.learn_from_review`` ships switched off, and a batch
smaller than ``review_min_samples`` is refused: a handful of corrections must not
silently move a production threshold.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .config import (ROOT, evaluation_only_splits, field_spec, load_config,
                     resolve_path)
from .evaluate import grounding_ok, match_options, values_match
from .extract import extract_document
from .parse import parse_document
from .reviews import action_vocab, read_review_records
from .textutil import decimal_str, now_iso, parse_date, parse_decimal, parse_int
from .trust import (TRUST_VERSION, assess, feature_names, field_features,
                    heuristic_confidence, normalize_path)

CALIBRATE_VERSION = "0.3.0"


def _source_for(samples_dir: Path, doc_id: str) -> Optional[Path]:
    if not samples_dir.is_dir():
        return None
    for candidate in samples_dir.iterdir():
        if candidate.is_file() and candidate.stem == doc_id:
            return candidate
    return None


def load_split(split: str, config: Dict):
    """Parse (or reuse) each labelled document before extracting.

    The parse cache is only valid for the file it was built from: after
    ``docpipe.synth`` rewrites the samples, a cache keyed by document id still
    holds the previous layout. Reading it here would train the calibrator on
    boxes and rows that no longer exist, so the source hash is checked via
    ``parse_document`` instead of trusting the file by name.
    """
    parsed_dir = ROOT / "work" / "parsed"
    samples_dir = resolve_path(config["paths"]["samples"], config) / split
    docs, parsed_by_id = [], {}
    for label_path in sorted((resolve_path(config["paths"]["labels"], config)
                              / split).glob("*.json")):
        doc_id = label_path.stem
        source = _source_for(samples_dir, doc_id)
        if source is not None:
            parsed = parse_document(source, doc_id,
                                    dpi=int(config["parsing"].get("render_dpi", 300)),
                                    config=config, out_dir=parsed_dir)
        else:
            parsed_path = parsed_dir / f"{doc_id}.json"
            if not parsed_path.exists():
                continue
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        docs.append(extract_document(parsed, config))
        parsed_by_id[doc_id] = parsed
    return docs, parsed_by_id


def _flip_last_char(text: str) -> Optional[str]:
    """One-character change that keeps length and shape (near-miss labels)."""
    if not text:
        return None
    last = text[-1]
    if last.isdigit():
        return text[:-1] + str((int(last) + 1) % 10)
    if "a" <= last.lower() <= "y":
        return text[:-1] + chr(ord(last) + 1)
    return text[:-1] + ("9" if last != "9" else "8")


def _other_enum(value, spec: Dict, config: Dict) -> Optional[str]:
    table = ((config.get("enums") or {}).get(spec.get("enum")) or {})
    for name in table:
        if str(name) != str(value):
            return str(name)
    return None


def _near_miss(value, spec: Dict, config: Dict, rng: random.Random) -> Optional[str]:
    value_type = spec.get("type", "string")
    text = "" if value is None else str(value)
    if value_type == "decimal":
        number = parse_decimal(text)
        if number is None:
            return None
        delta = Decimal("0.03") if number >= 0 else Decimal("-0.03")
        return decimal_str(number + delta)
    if value_type == "date":
        parsed = parse_date(text)
        return (parsed + timedelta(days=rng.choice([1, 2, 3]))).isoformat() if parsed else None
    if value_type == "integer":
        number = parse_int(text)
        return str(number + rng.choice([1, 2])) if number is not None else None
    if value_type == "enum":
        return _other_enum(text, spec, config)
    return _flip_last_char(text)


def _wrong_value(value, spec: Dict, config: Dict, rng: random.Random) -> Optional[str]:
    value_type = spec.get("type", "string")
    text = "" if value is None else str(value)
    if value_type == "decimal":
        number = parse_decimal(text)
        if number is None:
            return None
        wrong = (number * Decimal("2.37")).quantize(Decimal("0.01"))
        if wrong == number:
            wrong += Decimal("123.45")
        return decimal_str(wrong)
    if value_type == "date":
        parsed = parse_date(text)
        return (parsed + timedelta(days=rng.choice([30, 45, 60]))).isoformat() if parsed else None
    if value_type == "integer":
        number = parse_int(text)
        return str(number * 7 + 3) if number is not None else None
    if value_type == "enum":
        return _other_enum(text, spec, config)
    wrong = "示例错误值"
    return wrong if wrong != text else wrong + "甲"


def _misgrounded(pred: Dict, rng: random.Random) -> Optional[Dict]:
    grounding = copy.deepcopy(pred.get("grounding") or {})
    bbox = list(grounding.get("bbox") or [])
    if not bbox:
        return None
    dx = rng.choice([38.0, 46.0, 55.0])
    dy = rng.choice([52.0, 61.0, 70.0])
    grounding["bbox"] = [round(bbox[0] + dx, 2), round(bbox[1] + dy, 2),
                         round(bbox[2] + dx, 2), round(bbox[3] + dy, 2)]
    # 坐标仍然"存在"（grounding_present 保持 1），只是指错了位置；
    # 这正是现有特征看不见、但真实核对能发现的错误。
    perturbed = copy.deepcopy(pred)
    perturbed["grounding"] = grounding
    return perturbed


def _perturbed_field(kind: str, entry: Dict, config: Dict,
                     rng: random.Random) -> Optional[Dict]:
    pred, spec = entry["pred"], entry["spec"]
    if kind == "misgrounded":
        return _misgrounded(pred, rng)
    builder = _near_miss if kind == "near_miss" else _wrong_value
    value = builder(entry["gt"].get("value"), spec, config, rng)
    if value is None:
        return None
    perturbed = copy.deepcopy(pred)
    perturbed["value"] = value
    return perturbed


def _synthetic_negatives(positives: List[Dict], config: Dict) -> List[Dict]:
    """Label-aware perturbations that look fine to the current signals.

    Each negative shares its source document id, so GroupKFold keeps it in the
    same fold as the document it came from and no synthetic error leaks into a
    test fold.
    """
    cfg = (config.get("calibration") or {}).get("synthetic_negatives") or {}
    if not cfg.get("enabled", True) or not positives:
        return []
    target = int(cfg.get("target", 120))
    max_per_field = max(1, int(cfg.get("max_per_field", 2)))
    rng = random.Random(int(cfg.get("seed", 20261025)))
    shares = {
        "near_miss": float(cfg.get("near_miss_share", 0.45)),
        "misgrounded": float(cfg.get("misgrounded_share", 0.35)),
        "wrong_value": float(cfg.get("wrong_value_share", 0.20)),
    }
    total = sum(shares.values()) or 1.0
    kinds: List[str] = []
    for kind, share in shares.items():
        kinds.extend([kind] * max(1, int(round(target * share / total))))
    while len(kinds) < target:
        kinds.append("near_miss")
    rng.shuffle(kinds)

    pool = list(positives)
    rng.shuffle(pool)
    samples: List[Dict] = []
    per_field: Counter = Counter()
    cursor = 0
    attempts = 0
    max_attempts = target * 15
    match = match_options(config)
    while len(samples) < target and attempts < max_attempts:
        entry = pool[cursor % len(pool)]
        cursor += 1
        attempts += 1
        key = (entry["doc"]["document_id"], entry["path"])
        if per_field[key] >= max_per_field:
            continue
        kind = kinds[len(samples) % len(kinds)]
        perturbed = _perturbed_field(kind, entry, config, rng)
        if perturbed is None:
            continue
        spec, gt = entry["spec"], entry["gt"]
        correct = values_match(spec, gt.get("value"), perturbed.get("value"), match)
        grounded = grounding_ok(gt, perturbed.get("grounding"), match)
        if correct and grounded:
            continue  # 扰动没有越过判等，不能当负例
        samples.append({
            "document_id": entry["doc"]["document_id"],
            "doc_type": entry["doc"]["doc_type"],
            "path": entry["path"],
            "group_path": normalize_path(entry["path"]),
            "risk": perturbed.get("risk", ""),
            "y": 0,
            "correct": bool(correct),
            "grounded": bool(grounded),
            "features": field_features(perturbed, entry["doc"]["doc_type"], config),
            "heuristic": heuristic_confidence(perturbed, config, entry["doc"]["doc_type"]),
            "synthetic": kind,
        })
        per_field[key] += 1
    return samples


def build_samples(split: str, config: Dict, augment: bool = False) -> List[Dict]:
    # SCENARIO 第 108 行：留出集标签只许评估脚本读。在这儿拦，是因为拟合与调参
    # 都从这里取样本——放行一次，后面所有 holdout 指标都不再可信，而且不会有任何报错。
    if split in evaluation_only_splits(config):
        raise SystemExit(
            f"标注集 {split} 是留出集：调参只许在开发集上拟合（`paths.evaluation_only_splits`），"
            f"否则留出指标会被自己刷高。评估请走 `python -m docpipe.evaluate --labels …`。")
    docs, parsed_by_id = load_split(split, config)
    # assess() fills format/cross-field signals before features are taken.
    assess(docs, parsed_by_id, config, cross_document=True)
    samples: List[Dict] = []
    positives: List[Dict] = []
    match = match_options(config)
    for doc in docs:
        label_path = resolve_path(config["paths"]["labels"], config) / split / f"{doc['document_id']}.json"
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        for path, gt in labels["fields"].items():
            pred = doc["fields"].get(path)
            if pred is None:
                continue
            spec = field_spec(doc["doc_type"], normalize_path(path), config) or {}
            correct = values_match(spec, gt["value"], pred.get("value"), match)
            grounded = grounding_ok(gt, pred.get("grounding"), match)
            sample = {
                "document_id": doc["document_id"],
                "doc_type": doc["doc_type"],
                "path": path,
                "group_path": normalize_path(path),
                "risk": pred.get("risk", ""),
                "y": 1 if (correct and grounded) else 0,
                "correct": bool(correct),
                "grounded": bool(grounded),
                "features": field_features(pred, doc["doc_type"], config),
                "heuristic": heuristic_confidence(pred, config, doc["doc_type"]),
                "source": "label",
            }
            samples.append(sample)
            if sample["y"] == 1:
                positives.append({"doc": doc, "path": path, "gt": gt,
                                  "pred": pred, "spec": spec})
    if augment:
        samples.extend(_synthetic_negatives(positives, config))
    return samples


def review_sample_dirs(config: Dict, override: Optional[List[str]] = None) -> List[Path]:
    """复核台账所在的结果目录；开关关着就一个都不扫。

    ``--review`` 的显式目录不看开关（人当场要求，就当场算），但 ``review_min_samples``
    这一关对两条路都生效：几条修正不足以移动生产阈值。
    """
    if override:
        return [Path(item) if Path(item).is_absolute() else ROOT / item for item in override]
    options = config.get("calibration") or {}
    if not options.get("learn_from_review"):
        return []
    return [resolve_path(item, config) for item in (options.get("review_dirs") or [])]


def _stored_documents(out_dir: Path) -> Dict[str, Dict]:
    """结果目录里存下来的字段载荷——复核记录要靠它还原当时的特征向量。"""
    documents: Dict[str, Dict] = {}
    if not out_dir.is_dir():
        return documents
    for item in sorted(out_dir.iterdir()):
        path = item / f"{item.name}.json"
        if not item.is_dir() or not path.exists():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("fields"):
            documents[item.name] = document
    return documents


def _anchor_text(value) -> str:
    """台账锚点的可比形态：null 与空串算同一个值。"""
    return "" if value is None else str(value)


def _drift_kind(record: Dict, field: Dict) -> str:
    """这条人工判决对着的还是当前这版产物吗——只拿台账自己带的两个锚点跟产物对。

    人在复核台上签的是**当时那一版**的值与自动判决；结果目录后来重跑过（换解析器、改阈值、
    重新 OCR），字段就换了值、也换了判决，可台账还是原来那一行。这时把 accept 当成「这一版
    特征是对的」折进校准池，学的是一条没人看过的对照。`serve.append_review` 每次都写
    `original_value` 与 `auto_decision`，所以缺键（老台账、手写的）不判——看不见证据不等于有矛盾。
    """
    parts = []
    if "original_value" in record and "value" in field:
        if _anchor_text(record["original_value"]) != _anchor_text(field["value"]):
            parts.append("value")
    recorded = str(record.get("auto_decision") or "")
    now = str(field.get("decision") or "")
    if recorded and now and recorded != now:
        parts.append("decision")
    return "+".join(parts)


def review_samples(dirs: List[Path], config: Dict) -> List[Dict]:
    """人工终局判定 → 校准样本：accept 说明自动值是对的（y=1），correct 说明得重录（y=0）。

    同一份特征向量、两种结局，正是校准器要学的那个对照。取数只用结果目录里已经
    存下来的字段，不重跑解析与 OCR；同一（文档，字段）跨目录也只认最后一次判定。
    每条样本带一个 `drift` 标签（`""` 或 `value`/`decision`/`value+decision`），
    折不折由 `merge_review_samples` 按 `calibration.review_drift` 定，这里只如实标注。
    """
    latest: Dict[Tuple[str, str], Tuple[Dict, Dict, str]] = {}
    vocab = action_vocab(config)
    for directory in dirs:
        documents = _stored_documents(directory)
        for document_id, document in documents.items():
            records = read_review_records(directory, document_id)
            for field_path, record in records.items():
                field = (document.get("fields") or {}).get(field_path)
                action = record.get("action")
                if not field or str(action) not in vocab["actions"]:
                    continue
                key = (document_id, field_path)
                previous = latest.get(key)
                if previous and str(previous[1].get("timestamp") or "") >= str(
                        record.get("timestamp") or ""):
                    continue
                latest[key] = (field, record, document.get("doc_type", ""))

    samples: List[Dict] = []
    for (document_id, field_path), (field, record, doc_type) in sorted(latest.items()):
        grounded = bool(field.get("grounding"))
        correct = str(record.get("action")) in set(vocab["confirming"])
        samples.append({
            "document_id": document_id,
            "doc_type": doc_type,
            "path": field_path,
            "group_path": normalize_path(field_path),
            "risk": field.get("risk", ""),
            "y": 1 if (correct and grounded) else 0,
            "correct": bool(correct),
            "grounded": bool(grounded),
            "features": field_features(field, doc_type, config),
            "heuristic": heuristic_confidence(field, config, doc_type),
            "source": "review",
            "review_action": record.get("action"),
            "auto_decision": record.get("auto_decision") or field.get("decision", ""),
            "reviewed_at": record.get("timestamp"),
            "drift": _drift_kind(record, field),
        })
    return samples


def merge_review_samples(samples: List[Dict], reviewed: List[Dict],
                        config: Dict) -> Tuple[List[Dict], Dict]:
    """标注永远优先：同一（文档，字段）已有标注就不再拿复核记录当第二份真值。

    不足 ``review_min_samples`` 就整批不接（``enabled=false`` 记进产物），因为几条
    修正会把可靠性曲线拉成噪声——那正是"用样本刷数字"的形态。
    """
    options = config.get("calibration") or {}
    minimum = int(options.get("review_min_samples", 20))
    requested = str(options.get("review_drift", "drop"))
    policy = requested if requested in ("drop", "keep") else "drop"
    labeled = {(item["document_id"], item["path"]) for item in samples}
    unmatched = [item for item in reviewed if (item["document_id"], item["path"]) not in labeled]
    drifted = [item for item in unmatched if item.get("drift")]
    kept = [item for item in unmatched if policy == "keep" or not item.get("drift")]
    stats = {
        "records": len(reviewed),
        "shadowed_by_labels": len(reviewed) - len(unmatched),
        "drift": len(drifted),
        "drift_value": sum(1 for item in drifted if "value" in item["drift"]),
        "drift_decision": sum(1 for item in drifted if "decision" in item["drift"]),
        "drift_dropped": len(drifted) if policy == "drop" else 0,
        "drift_kept": len(drifted) if policy == "keep" else 0,
        "drift_policy": policy,
        "usable": len(kept),
        "accepted": sum(1 for item in kept if item["y"] == 1),
        "corrected": sum(1 for item in kept if item["y"] == 0),
        "min_samples": minimum,
        "enabled": bool(kept) and len(kept) >= minimum,
    }
    return (samples + kept if stats["enabled"] else samples), stats


def _source_of(item: Dict) -> str:
    if item.get("synthetic"):
        return "synthetic"
    return str(item.get("source") or "label")


def _split_by_source(samples: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {"label": [], "review": [], "synthetic": []}
    for item in samples:
        buckets[_source_of(item)].append(item)
    return buckets


def _reliability_points(probabilities: np.ndarray, labels: np.ndarray,
                        bins: List[float]) -> List[Dict]:
    points: List[Dict] = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high >= bins[-1]:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())
        if not count:
            continue
        points.append({
            "bin": [round(float(low), 4), round(float(high), 4)],
            "count": count,
            "mean_confidence": round(float(probabilities[mask].mean()), 4),
            "observed_accuracy": round(float(labels[mask].mean()), 4),
        })
    return points


def _reliability_block(probabilities: np.ndarray, labels: np.ndarray,
                       config: Dict) -> Dict:
    bins = [float(value) for value in
            (config.get("calibration", {}).get("reliability_bins")
             or [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01])]
    points = _reliability_points(probabilities, labels, bins)
    count = len(probabilities)
    ece = sum(point["count"] / count *
              abs(point["observed_accuracy"] - point["mean_confidence"])
              for point in points) if count else 0.0
    brier = float(np.mean((probabilities - labels) ** 2)) if count else 0.0
    return {
        "samples": count,
        "ece": round(float(ece), 4),
        "brier": round(brier, 4),
        "points": points,
    }


def _write_reliability_svg(block: Dict, path: Path, title: str) -> None:
    """Dependency-free reliability diagram (predicted vs observed)."""
    width, height, pad = 460, 460, 56
    span = width - 2 * pad

    def px(value: float) -> float:
        return pad + value * span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="15">{title}</text>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#888"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#888"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{pad}" '
        'stroke="#bbb" stroke-dasharray="4 4"/>',
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12">预测置信度</text>',
        f'<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" '
        'text-anchor="middle" font-family="sans-serif" font-size="12">实际准确率</text>',
    ]
    for point in block.get("points") or []:
        cx = px(float(point["mean_confidence"]))
        cy = px(1.0 - float(point["observed_accuracy"]))
        radius = 4 + 10 * (point["count"] / max(1, block.get("samples") or 1)) ** 0.5
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="#2563eb" '
            f'fill-opacity="0.55" stroke="#1e40af"/>'
            f'<text x="{cx:.1f}" y="{cy - radius - 5:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="10">{point["count"]}</text>')
    parts.append(f'<text x="{pad}" y="{height - pad + 22}" font-family="sans-serif" '
                 f'font-size="11">ECE={block.get("ece")} · Brier={block.get("brier")} '
                 f'· n={block.get("samples")}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _isotonic_points(x: np.ndarray, y: np.ndarray) -> Dict:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if len(np.unique(xs)) < 2:
        return {}
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    fitted = model.fit_transform(xs, ys)
    grid_x, grid_y = [], []
    for value, yhat in zip(xs, fitted):
        value = round(float(value), 6)
        yhat = round(float(yhat), 6)
        if grid_x and grid_x[-1] == value:
            grid_y[-1] = yhat
        else:
            grid_x.append(value)
            grid_y.append(yhat)
    return {"x": grid_x, "y": grid_y}


SUPPORTED_METHODS = {"logistic_regression_isotonic"}


def calibration_method(config: Dict) -> str:
    """校准方法名。未实现的名字报错并列出可选值，不静默换一种算法。"""
    method = str(config["calibration"].get("method", "logistic_regression_isotonic"))
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"未实现的校准方法 {method!r}（calibration.method）：可选 {sorted(SUPPORTED_METHODS)}")
    return method


def require_document_grouping(config: Dict) -> None:
    """SCENARIO §6.2：折外分数必须按文档分组，同一文档的字段不得跨折。"""
    if not bool(config["calibration"].get("group_by_document", True)):
        raise ValueError(
            "calibration.group_by_document 不支持关闭：关掉等于让同一文档的字段跨折泄漏，"
            "与 SCENARIO §6.2 的按文档分组要求相反")


def _fit_model(samples: List[Dict], config: Dict) -> Dict:
    require_document_grouping(config)
    X = np.array([item["features"] for item in samples], dtype=float)
    y = np.array([item["y"] for item in samples], dtype=int)
    groups = np.array([item["document_id"] for item in samples])

    model = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0)
    model.fit(X, y)

    n_splits = min(int(config["calibration"].get("n_splits", 5)), len(set(groups)))
    oof = np.array([item["heuristic"] for item in samples], dtype=float)
    fold_scores = []
    if len(set(y)) >= 2 and n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(set(y[train_idx])) < 2:
                continue
            fold = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0)
            fold.fit(X[train_idx], y[train_idx])
            oof[test_idx] = fold.predict_proba(X[test_idx])[:, 1]
            fold_scores.append({
                "train_docs": sorted(set(groups[train_idx])),
                "test_docs": sorted(set(groups[test_idx])),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
            })

    return {
        "model": model,
        "oof": oof,
        "y": y,
        "fold_scores": fold_scores,
        "isotonic": _isotonic_points(oof, y),
    }


def _artifact(samples: List[Dict], fit: Dict, split: str, config: Dict,
              review: Optional[Dict] = None,
              review_dirs: Optional[List[Path]] = None) -> Dict:
    buckets = _split_by_source(samples)
    synthetic = buckets["synthetic"]
    synthetic_counts = Counter(item["synthetic"] for item in synthetic)
    field_samples = Counter(item["group_path"] for item in samples)
    group_samples = Counter(item["doc_type"] for item in samples)
    oof, labels = fit["oof"], fit["y"]
    masks = {name: np.array([_source_of(item) == name for item in samples])
             for name in buckets}
    reliability = {
        "all": _reliability_block(oof, labels, config),
        "label_only": _reliability_block(oof[masks["label"]], labels[masks["label"]], config),
    }
    # 复核样本天生偏难（机器自己拿不准才去问人），把它的曲线单列出来，
    # 才看得清"曲线变好"是学到了东西还是只是把难例混进了总体。
    for name in ("review", "synthetic"):
        if masks[name].any():
            reliability[f"{name}_only"] = _reliability_block(
                oof[masks[name]], labels[masks[name]], config)
    return {
        "calibrate_version": CALIBRATE_VERSION,
        "trust_version": TRUST_VERSION,
        "method": calibration_method(config),
        "split": split,
        "trained_at": now_iso(),
        "features": feature_names(config),
        "coef": [round(float(value), 8) for value in fit["model"].coef_[0]],
        "intercept": round(float(fit["model"].intercept_[0]), 8),
        "isotonic": fit["isotonic"],
        "sample_grouping": config["calibration"].get("sample_grouping", "field_auto"),
        "n_samples": int(len(samples)),
        "sample_sources": {name: len(items) for name, items in sorted(buckets.items())},
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "negative_sources": {
            "label": int(sum(1 for item in buckets["label"] if item["y"] == 0)),
            "review": int(sum(1 for item in buckets["review"] if item["y"] == 0)),
            "synthetic": dict(sorted(synthetic_counts.items())),
        },
        "review_loop": {
            "gate": bool((config.get("calibration") or {}).get("learn_from_review")),
            "dirs": [str(item) for item in (review_dirs or [])],
            **(review or {}),
        },
        "reliability": reliability,
        "by_doc_type": {
            name: {"samples": int(count), "negatives": int(sum(1 for item in samples
                                                                if item["doc_type"] == name and item["y"] == 0))}
            for name, count in sorted(group_samples.items())
        },
        "field_samples": dict(sorted(field_samples.items())),
        "group_samples": dict(group_samples),
        "cv_folds": fit["fold_scores"],
    }


def _class_balance(samples: List[Dict]) -> Tuple[int, int]:
    positives = sum(1 for item in samples if item["y"] == 1)
    return positives, len(samples) - positives


def _require_two_classes(positives: int, negatives: int, what: str) -> None:
    """只有一个类别的样本池拟合不出概率：宁可不写产物，也不发一个常数模型。

    标注集全对是好事，但校准器在那里学不到任何东西。合法的补法只有两条：
    ``--review`` 把人工判决折进来（那是真实的负样本），或 ``--augment`` 只在探针里
    看合成分布。把阈值调低、把负样本编出来都不算——见 SCENARIO 的诚实弃权条款。
    """
    if positives and negatives:
        return
    if negatives:
        raise SystemExit(
            f"{what}：{negatives} 条样本全是负例、0 条正例，拟合不出概率。"
            "复核台账里的 accept 才是正例，用 --review 把判决折进来；"
            "不许把正样本编出来凑数。")
    raise SystemExit(
        f"{what}：{positives} 条样本全是正例、0 条负例，拟合不出概率。"
        "用 --review 折入复核台账里的人工判决（correct 就是负例），"
        "或用 --augment 只跑探针看合成分布；不许用调阈值代替真实负样本。")


def train(split: str, config: Dict, augment: bool = False,
          review_override: Optional[List[str]] = None) -> Dict:
    samples = build_samples(split, config, augment=augment)
    dirs = review_sample_dirs(config, review_override)
    merged, review = merge_review_samples(samples, review_samples(dirs, config), config)
    samples = merged
    if len(samples) < int(config["calibration"].get("min_total_samples", 60)):
        raise SystemExit(f"not enough calibration samples ({len(samples)})")
    _require_two_classes(*_class_balance(samples),
                         f"标注集 {split}" + ("+复核台账" if review["enabled"] else ""))
    return _artifact(samples, _fit_model(samples, config), split, config,
                     review=review, review_dirs=dirs)


def _decision_stats(probabilities: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.90) -> Dict:
    accepted = probabilities >= threshold
    count = int(accepted.sum())
    errors = int((labels[accepted] == 0).sum())
    return {
        "threshold": threshold,
        "auto_accept_rate": round(count / len(probabilities), 4) if len(probabilities) else 0.0,
        "accepted_error_rate": round(errors / count, 4) if count else 0.0,
        "mean_confidence": round(float(probabilities.mean()), 4) if len(probabilities) else 0.0,
    }


def _predict(fit: Dict, features: List[List[float]]) -> np.ndarray:
    """Apply logistic + isotonic exactly like ``Calibrator.predict`` does."""
    probabilities = fit["model"].predict_proba(np.array(features, dtype=float))[:, 1]
    iso = fit["isotonic"] or {}
    xs, ys = iso.get("x") or [], iso.get("y") or []
    out: List[float] = []
    for probability in probabilities:
        value = float(probability)
        if xs and ys:
            if value <= xs[0]:
                value = ys[0]
            elif value >= xs[-1]:
                value = ys[-1]
            else:
                for index in range(1, len(xs)):
                    if value <= xs[index]:
                        left, right = xs[index - 1], xs[index]
                        ratio = (value - left) / (right - left) if right > left else 1.0
                        value = ys[index - 1] + ratio * (ys[index] - ys[index - 1])
                        break
        out.append(value)
    return np.array(out)


class _ShippedModel:
    """出厂产物里的 logistic 系数伪装成 sklearn 模型，只为让探针继续算生产分。"""

    def __init__(self, coef: List[float], intercept: float):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(matrix, dtype=float) @ self.coef + self.intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - probability, probability], axis=1)


def _shipped_fit(config: Dict) -> Optional[Dict]:
    """标注集里一个负例都不剩时，探针仍要能回答"生产模型对新样本给什么分"。

    生产模型这时只能取出厂产物、不能重拟合：拿全正例的池子拟合出来的是常数模型，
    那种"探针结果"是假的。
    """
    path = Path(str(config["calibration"].get("artifact") or "work/calibration.json"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not artifact.get("coef"):
        return None
    return {
        "model": _ShippedModel(artifact["coef"], artifact.get("intercept") or 0.0),
        "isotonic": artifact.get("isotonic") or {},
        "oof": None,
        "y": None,
        "fold_scores": [],
    }


def probe(split: str, config: Dict) -> Dict:
    """Compare production calibration with a synthetic-negative-augmented model.

    Answers the P0-2 question directly: do near-miss / misgrounded / wrong-value
    negatives make confidence probability-readable, or do they only show that
    today's signals cannot separate them?
    """
    samples = build_samples(split, config, augment=True)
    real = [item for item in samples if not item.get("synthetic")]
    synthetic = [item for item in samples if item.get("synthetic")]
    minimum = int(config["calibration"].get("min_total_samples", 60))
    if len(real) < minimum:
        raise SystemExit(f"not enough real calibration samples ({len(real)})")
    real_mask = np.array([not item.get("synthetic") for item in samples])
    positives, negatives = _class_balance(real)
    if positives and negatives:
        base_fit, production_source = _fit_model(real, config), "refit"
    else:
        base_fit = _shipped_fit(config)
        if base_fit is None:
            _require_two_classes(positives, negatives, f"探针 标注集（{split}，不含合成负例）")
            raise SystemExit("探针需要出厂产物，但 work/calibration.json 不在。")
        production_source = "shipped_artifact"
        base_fit["oof"] = _predict(base_fit, [item["features"] for item in real])
        base_fit["y"] = np.array([item["y"] for item in real], dtype=int)
    aug_fit = _fit_model(samples, config)
    threshold = float(config["thresholds"].get("global_min_confidence", 0.90))
    base_stats = _decision_stats(base_fit["oof"], base_fit["y"], threshold)
    aug_real_stats = _decision_stats(aug_fit["oof"][real_mask], aug_fit["y"][real_mask],
                                     threshold)
    aug_synth_stats = _decision_stats(aug_fit["oof"][~real_mask], aug_fit["y"][~real_mask],
                                      threshold)
    # 生产模型（不带合成负例训练）遇到"看起来正常但内容错"的样本会给出什么分：
    # 这是"能不能当概率读"最直接的证据。
    synth_labels = np.zeros(len(synthetic), dtype=int)
    prod_on_synth = _predict(base_fit, [item["features"] for item in synthetic])
    prod_synth_stats = _decision_stats(prod_on_synth, synth_labels, threshold)
    base_reliability = _reliability_block(base_fit["oof"], base_fit["y"], config)
    aug_reliability = _reliability_block(aug_fit["oof"][real_mask],
                                         aug_fit["y"][real_mask], config)
    aug_synth_reliability = _reliability_block(aug_fit["oof"][~real_mask],
                                               aug_fit["y"][~real_mask], config)
    prod_synth_reliability = _reliability_block(prod_on_synth, synth_labels, config)
    same_features = sum(1 for item in synthetic
                        if any(tuple(item["features"]) == tuple(other["features"])
                               for other in real))
    readable = prod_synth_stats["auto_accept_rate"] < 0.5
    return {
        "probe_version": CALIBRATE_VERSION,
        "split": split,
        "trained_at": now_iso(),
        "threshold": threshold,
        "real_samples": len(real),
        "production_source": production_source,
        "synthetic_samples": len(synthetic),
        "synthetic_by_kind": dict(sorted(Counter(
            item["synthetic"] for item in synthetic).items())),
        # 负例与真实正例共享完全相同的特征向量，说明现有信号看不见这类错误。
        "synthetic_with_identical_features": same_features,
        "verdict": "probability_readable" if readable else "still_ordinal",
        "reason": (
            "合成负例与真实正例的特征向量完全相同：现有信号（OCR 置信、规则命中、"
            "表格分、坐标存在性）看不见'值错但形似'与'坐标错位但存在'。"
            "生产模型会给这些负例高置信，因此置信度仍只能当排序，不能当概率。"
            "要升级需先补值复核/抽取通路一致性特征（P1-2），而不是只加负例。"
        ) if not readable else "生产模型已能在这类负例上给出低置信。",
        "production": {"decision": base_stats, "reliability": base_reliability},
        "production_on_synthetic": {
            "decision": prod_synth_stats,
            "reliability": prod_synth_reliability,
        },
        "augmented": {
            "decision_on_real": aug_real_stats,
            "decision_on_synthetic": aug_synth_stats,
            "reliability_on_real": aug_reliability,
            "reliability_on_synthetic": aug_synth_reliability,
        },
    }


def _print_probe(result: Dict) -> None:
    def line(name: str, block: Dict, stats: Dict) -> str:
        return (f"{name:<24} accept={stats['auto_accept_rate']:.3f} "
                f"err={stats['accepted_error_rate']:.3f} "
                f"ECE={block.get('ece')} Brier={block.get('brier')} "
                f"n={block.get('samples')}")
    print(json.dumps({
        "real_samples": result["real_samples"],
        "synthetic_samples": result["synthetic_samples"],
        "synthetic_by_kind": result["synthetic_by_kind"],
        "synthetic_with_identical_features": result["synthetic_with_identical_features"],
        "production_source": result["production_source"],
        "verdict": result["verdict"],
    }, ensure_ascii=False, indent=2))
    print(line("production / real", result["production"]["reliability"],
               result["production"]["decision"]))
    print(line("production / synthetic", result["production_on_synthetic"]["reliability"],
               result["production_on_synthetic"]["decision"]))
    print(line("augmented / real", result["augmented"]["reliability_on_real"],
               result["augmented"]["decision_on_real"]))
    print(line("augmented / synthetic", result["augmented"]["reliability_on_synthetic"],
               result["augmented"]["decision_on_synthetic"]))
    print("reason:", result["reason"])


def _print_artifact(artifact: Dict, out_path: Path) -> None:
    print(json.dumps({
        "samples": artifact["n_samples"],
        "sample_sources": artifact["sample_sources"],
        "review_loop": artifact["review_loop"],
        "positives": artifact["positives"],
        "negatives": artifact["negatives"],
        "negative_sources": artifact["negative_sources"],
        "folds": len(artifact["cv_folds"]),
        "isotonic_points": len(artifact["isotonic"].get("x", [])),
        "reliability": {name: {"ece": block.get("ece"), "brier": block.get("brier"),
                               "samples": block.get("samples")}
                        for name, block in artifact["reliability"].items()},
        "by_doc_type": artifact["by_doc_type"],
    }, ensure_ascii=False, indent=2))
    print("written:", out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit confidence calibration from dev labels")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", default=None)
    parser.add_argument("--augment", action="store_true",
                        help="ship an artifact trained with synthetic negatives")
    parser.add_argument("--probe", action="store_true",
                        help="write synthetic-negative probe and reliability diagrams")
    parser.add_argument("--review", action="store_true",
                        help="fold the human review ledger into the sample pool")
    parser.add_argument("--review-dir", dest="review_dir", action="append", default=[],
                        help="result directory to read review records from (repeatable)")
    args = parser.parse_args(argv)
    config = load_config()
    override = list(args.review_dir)
    if args.review and not override:
        override = [str(resolve_path(item, config))
                    for item in (config["calibration"].get("review_dirs") or ["out/user"])]
    # 样本池只剩一个类别时 train 如实拒训（见 _require_two_classes）。探针是诊断，
    # 不该被这一步连坐——照样出曲线，但退出码非零，别让人以为产物已经重训过。
    refusal: Optional[str] = None
    try:
        artifact = train(args.split, config, augment=args.augment,
                         review_override=override or None)
    except SystemExit as error:
        if not args.probe:
            raise
        refusal, artifact = str(error), None
    if artifact is not None:
        out_path = Path(args.out) if args.out else ROOT / config["calibration"]["artifact"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_artifact(artifact, out_path)
    else:
        print(f"[校准未写入] {refusal}")
    if args.probe:
        result = probe(args.split, config)
        work_dir = resolve_path(config["paths"]["work"], config)
        probe_path = work_dir / "calibration_probe.json"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        eval_dir = work_dir / "eval"
        _write_reliability_svg(
            result["production"]["reliability"],
            eval_dir / f"reliability_{args.split}.svg",
            f"校准可靠性（生产模型 · {args.split}）")
        _write_reliability_svg(
            result["production_on_synthetic"]["reliability"],
            eval_dir / f"reliability_{args.split}_synthetic_errors.svg",
            f"生产模型在合成负例上的可靠性（{args.split}）")
        _write_reliability_svg(
            result["augmented"]["reliability_on_synthetic"],
            eval_dir / f"reliability_{args.split}_synthetic.svg",
            f"合成负例上的可靠性（增广模型 · {args.split}）")
        _print_probe(result)
        print("probe written:", probe_path)
    if refusal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
