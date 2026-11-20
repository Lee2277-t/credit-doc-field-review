"""Compare extracted fields with ground-truth labels.

Ground truth is read only from labels/<split>/; samples are never consulted
for values during extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, field_spec, load_config
from .reviews import action_vocab, read_review_records, review_status
from .textutil import normalize_code, normalize_text, parse_date, parse_decimal

MONEY_TYPES = {"decimal"}
DATE_TYPES = {"date"}
# 收集哪些流水列（按配置的内容键取用）；顺序无关匹配的键表在 evaluation.match 里。
TXN_COLUMNS = ("txn_date", "amount", "balance", "direction")
DEFAULT_TXN_KEYS = ("txn_date", "amount", "balance")
TXN_RE = re.compile(r"^bs\.transactions\[(\d+)\]\.(\w+)$")


def match_options(config: Optional[Dict]) -> Dict:
    """SCENARIO §10.1 的三条匹配规则：值、溯源、交易行内容键。"""
    return ((config or {}).get("evaluation") or {}).get("match") or {}


def transaction_keys(config: Optional[Dict]) -> tuple:
    """§10.1 只按 (txn_date, amount, balance) 做顺序无关匹配，direction 不进内容键。

    配置里写了不认识的列名就当场报错：默默忽略会让所有行凑不齐、整份文档的交易
    匹配凭空变成「没有可比的标注」，而且不会有任何告警。
    """
    keys = tuple(match_options(config).get("transaction_keys") or DEFAULT_TXN_KEYS)
    unknown = [key for key in keys if key not in TXN_COLUMNS]
    if unknown or not keys:
        raise ValueError(f"evaluation.match.transaction_keys 含未知列 {unknown}（只认 {TXN_COLUMNS}）")
    return keys


def _compact(text) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _norm_string(text) -> str:
    return _compact(normalize_text(text))


def values_match(spec: Dict, gt_value, pred_value, options: Optional[Dict] = None) -> bool:
    """§10.1 值正确：按 4.2 归一化后完全相等。小数容差默认 0，放宽要显式写配置。"""
    if pred_value in (None, ""):
        return gt_value in (None, "")
    if gt_value in (None, ""):
        return pred_value in (None, "")
    value_type = spec.get("type", "string")
    if value_type == "decimal":
        left, right = parse_decimal(gt_value), parse_decimal(pred_value)
        if left is None or right is None:
            return False
        tolerance = Decimal(str((options or {}).get("decimal_tolerance", 0.0)))
        return abs(left - right) <= tolerance
    if value_type == "date":
        return parse_date(gt_value) == parse_date(pred_value)
    if value_type == "integer":
        try:
            return int(str(gt_value)) == int(str(pred_value))
        except (TypeError, ValueError):
            return False
    if spec.get("risk") == "identity":
        return normalize_code(gt_value) == normalize_code(pred_value)
    return _norm_string(gt_value) == _norm_string(pred_value)


def _iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bbox_values(bbox) -> Optional[tuple]:
    """框只要能拆成四个数才算有坐标；形状不对时评估要记「没溯源」，不是崩。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None


def _center_inside(outer: List[float], inner: List[float]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def grounding_ok(gt: Dict, pred: Optional[Dict], options: Optional[Dict] = None) -> bool:
    """§10.1 溯源正确：预测 bbox 中心点在标注 bbox 内，或与标注 bbox 的 IoU ≥ min_iou。

    旧口径是「与标注框有任何正重叠就算命中」——同一份页面上框更大、更歪的预测也会被
    记成溯源正确，`grounding_hit_rate` 因此比契约说的高。
    """
    if not pred or not pred.get("bbox"):
        return False
    if pred.get("page") != gt.get("page"):
        return False
    gt_box = _bbox_values(gt.get("bbox"))
    pred_box = _bbox_values(pred["bbox"])
    if gt_box is None or pred_box is None:
        return False
    opts = options or {}
    if bool(opts.get("center_inside", True)) and _center_inside(gt_box, pred_box):
        return True
    return _iou(gt_box, pred_box) >= float(opts.get("min_iou", 0.5))


def _grounding_present(grounding: Optional[Dict]) -> bool:
    """§6.5 溯源缺失率的分子判据：缺页码、缺 bbox，或 bbox 宽/高不为正。

    这里只看产物自己有没有坐标，不与标注比（那是 `grounding_ok` 的活）。两件事
    分开是因为「坐标画不出来」与「坐标画错地方」要落在两张表里。
    """
    if not isinstance(grounding, dict):
        return False
    if grounding.get("page") in (None, ""):
        return False
    bbox = _bbox_values(grounding.get("bbox"))
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0


def normalized_value(spec: Dict, value):
    if value in (None, ""):
        return None
    value_type = spec.get("type", "string")
    if value_type == "decimal":
        return parse_decimal(value)
    if value_type == "date":
        return parse_date(value)
    if spec.get("risk") == "identity":
        return normalize_code(value)
    return normalize_text(value)


def transaction_rows(fields: Dict[str, Dict], doc_type: str, config: Dict,
                     getter, keys: tuple = DEFAULT_TXN_KEYS) -> Dict[int, Dict]:
    rows: Dict[int, Dict] = {}
    for path, item in fields.items():
        match = TXN_RE.match(path)
        if not match:
            continue
        index, column = int(match.group(1)), match.group(2)
        if column not in keys:
            continue
        spec = field_spec(doc_type, f"bs.transactions[i].{column}", config) or {}
        rows.setdefault(index, {})[column] = normalized_value(spec, getter(item))
    return rows


def transaction_comparison(labels: Dict, pred_fields: Dict, doc_type: str,
                           config: Dict) -> Optional[Dict]:
    """§10.1：按 row_index 匹配一次，再按配置的内容键（默认日期/金额/余额）顺序无关匹配一次。"""
    keys = transaction_keys(config)
    gt_rows = transaction_rows(labels["fields"], doc_type, config,
                               lambda item: item.get("value"), keys)
    pred_rows = transaction_rows(pred_fields, doc_type, config,
                                 lambda item: item.get("value"), keys)

    def complete(row: Dict) -> bool:
        return all(row.get(key) is not None for key in keys)

    gt_complete = {index: row for index, row in gt_rows.items() if complete(row)}
    if not gt_complete:
        return None
    order_ok = 0
    gt_tuples: Counter = Counter()
    pred_tuples: Counter = Counter()
    for index, row in gt_complete.items():
        key = tuple(row.get(column) for column in keys)
        gt_tuples[key] += 1
        pred_row = pred_rows.get(index)
        if pred_row and complete(pred_row) and tuple(pred_row.get(column) for column in keys) == key:
            order_ok += 1
    for row in pred_rows.values():
        if complete(row):
            pred_tuples[tuple(row.get(column) for column in keys)] += 1
    content_matched = sum((gt_tuples & pred_tuples).values())
    return {
        "keys": list(keys),
        "gt_rows": len(gt_complete),
        "pred_rows": sum(1 for row in pred_rows.values() if complete(row)),
        "order_sensitive_correct": order_ok,
        "content_matched": content_matched,
        "content_match_rate": round(content_matched / len(gt_complete), 4),
    }


def error_category(record: Dict, config: Optional[Dict] = None) -> str:
    validation = ((config or load_config()).get("validation") or {})
    min_table_score = float(validation.get("min_table_score", 0.5))
    low_ocr = float(validation.get("low_ocr_threshold", 0.6))
    reasons = set(record.get("reasons") or [])
    signals = record.get("signals") or {}
    extractor = record.get("extractor_id") or ""
    if record.get("pred") is None:
        return "MISSING"
    if "GROUNDING_MISSING" in reasons:
        return "GROUNDING"
    if "UNKNOWN_DOC_TYPE" in reasons:
        return "DOC_TYPE"
    if "LOW_OCR" in reasons or float(signals.get("ocr_conf") or 0.0) < low_ocr:
        return "OCR"
    if "TEMPLATE_MISMATCH" in reasons or (
            extractor.startswith(("table_cell", "table_row"))
            and float(signals.get("table_score") if signals.get("table_score") is not None else 1.0)
            < min_table_score):
        return "TABLE_ALIGNMENT"
    if "AMBIGUOUS_UNIT" in reasons:
        return "UNIT"
    if "AMBIGUOUS_DATE" in reasons:
        return "DATE"
    if float(signals.get("rule_exact") if signals.get("rule_exact") is not None else 1.0) < 1:
        return "RULE_MISMATCH"
    return "OTHER"


def compare_document(doc_id: str, split: str, extracted: Dict, labels: Dict,
                     config: Dict) -> Dict:
    doc_type = labels["doc_type"]
    stats = Counter()
    errors = []
    by_risk = defaultdict(Counter)
    match = match_options(config)
    for path, gt in labels["fields"].items():
        spec = field_spec(doc_type, re.sub(r"\[\d+\]", "[i]", path), config)
        pred = extracted["fields"].get(path)
        risk = (spec or {}).get("risk", "unknown")
        stats["gt"] += 1
        by_risk[risk]["gt"] += 1
        if not pred:
            stats["missing"] += 1
            errors.append({"path": path, "gt": gt["value"], "pred": None, "reason": "missing"})
            continue
        correct = values_match(spec or {}, gt["value"], pred.get("value"), match)
        if correct:
            stats["value_correct"] += 1
            by_risk[risk]["value_correct"] += 1
        else:
            errors.append({"path": path, "gt": gt["value"], "pred": pred.get("value"),
                           "raw": pred.get("raw_text"), "status": pred.get("status")})
        grounded = grounding_ok(gt, pred.get("grounding"), match)
        if grounded:
            stats["grounded"] += 1
        if correct and grounded:
            stats["correct_and_grounded"] += 1
            by_risk[risk]["correct_and_grounded"] += 1
    return {
        "document_id": doc_id,
        "doc_type": doc_type,
        "engine": extracted.get("engine"),
        "stats": dict(stats),
        "by_risk": {risk: dict(counter) for risk, counter in by_risk.items()},
        "errors": errors[:80],
    }


def summarize(results: List[Dict]) -> Dict:
    total = Counter()
    by_type = defaultdict(Counter)
    by_risk = defaultdict(Counter)
    for item in results:
        for key, value in item["stats"].items():
            total[key] += value
            by_type[item["doc_type"]][key] += value
        for risk, counter in item["by_risk"].items():
            for key, value in counter.items():
                by_risk[risk][key] += value
    def rates(counter):
        return {
            "gt": counter.get("gt", 0),
            "value_accuracy": round(counter.get("value_correct", 0) / counter["gt"], 4)
            if counter.get("gt") else None,
            "grounding_rate": round(counter.get("grounded", 0) / counter["gt"], 4)
            if counter.get("gt") else None,
            "correct_and_grounded_rate": round(counter.get("correct_and_grounded", 0) / counter["gt"], 4)
            if counter.get("gt") else None,
            "missing": counter.get("missing", 0),
        }
    return {
        "overall": rates(total),
        "by_doc_type": {name: rates(counter) for name, counter in by_type.items()},
        "by_risk": {name: rates(counter) for name, counter in by_risk.items()},
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Optional[List[float]]:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


# (指标, 方向, 门槛键, 分母键, 区间键, 母体键)：分母=这项指标实际数出来的样本，
# 母体=这一级里到底有没有这类字段。两者分开，才分得清「没抽到」和「这一级压根不该有」。
CLAIM_METRICS = (
    # §6.5 第一行：「自动接受率 = accept 的**必填**字段数 / 全部**必填**字段数」。分母不是
    # 全部字段——可选格（支行名、上期数）本来就不参与这条门槛，摊进分母是把门槛稀释掉。
    ("auto_accept_rate", "min", "min_auto_accept_rate", "required_field_count",
     "auto_accept_rate_ci", "required_field_count"),
    ("accepted_error_rate", "max", "max_accepted_error_rate", "accept_count", "accepted_error_ci",
     "field_count"),
    ("monetary_accepted_error_rate", "max", "max_monetary_accepted_error_rate",
     "monetary_accept_count", "monetary_accepted_error_ci", "monetary_field_count"),
    ("accepted_grounding_missing", "max", "max_accepted_grounding_missing", "accept_count", None,
     "field_count"),
    ("grounding_hit_rate", "min", "min_grounding_hit_rate", "field_count", "grounding_hit_rate_ci",
     "field_count"),
)
CLAIM_STATUS_RANK = ("not_met", "no_sample", "provisional", "met", "not_applicable")


def claim_targets(config: Dict) -> Dict:
    """§6.5 的门槛与样本下限，全部读 `acceptance` 那一节，代码里不留第二个数。"""
    gates = config.get("acceptance") or {}
    confidence = float(gates.get("claim_confidence", 0.95))
    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return {
        "min_auto_accept_rate": float(gates.get("min_auto_accept_rate", 0.60)),
        "max_accepted_error_rate": float(gates.get("max_accepted_error_rate", 0.02)),
        "max_monetary_accepted_error_rate": float(
            gates.get("max_monetary_accepted_error_rate", 0.01)),
        "max_accepted_grounding_missing": int(
            gates.get("max_accepted_grounding_missing", 0)),
        "min_grounding_hit_rate": float(gates.get("min_grounding_hit_rate", 0.98)),
        "min_accept_for_claim": int(gates.get("min_accept_for_claim", 100)),
        "claim_confidence": confidence,
        "z": round(z, 4),
    }


SCOPE_SYNTHETIC = "synthetic_only"
SCOPE_REAL = "real_calibrated"


def code_digest() -> Optional[str]:
    """代码版本：`docpipe/*.py` 逐文件摘要后再整体摘要，按路径排序保证可复现。"""
    digest = hashlib.sha256()
    files = sorted((ROOT / "docpipe").rglob("*.py"))
    if not files:
        return None
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            return None
        digest.update(b"\0")
    return digest.hexdigest()


def config_digest(config: Dict) -> str:
    """配置哈希：整份配置按排序键序列化，动任何一个阈值都会变。"""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_manifest_digest(split: str, config: Dict) -> Optional[str]:
    from . import samplefreeze

    try:
        return hashlib.sha256(samplefreeze.manifest_path(split, config).read_bytes()).hexdigest()
    except OSError:
        return None


def report_provenance(split: str, config: Dict, real_samples: int = 0) -> Dict:
    """SCENARIO §9.3/§9.4：每次评估记录代码版本、配置哈希、样本清单哈希，并把
    合成集的局限写进报告。

    `real_samples` 由调用方如实给出（CLI `--real-samples`，默认 0）：评估脚本不去
    扫用户目录数真实件——数不出来就说 0，宁可把范围说小，不许把合成集指标说成
    真实业务准确率。
    """
    opts = (config.get("evaluation") or {}).get("provenance") or {}
    min_real = int(opts.get("min_real_samples", 20))
    limitations = [str(item) for item in (opts.get("limitations") or []) if str(item)]
    real = max(0, int(real_samples))
    calibrated = min_real > 0 and real >= min_real
    return {
        "provenance": {
            "split": split,
            "code_sha256": code_digest(),
            "config_sha256": config_digest(config),
            "sample_manifest_sha256": sample_manifest_digest(split, config),
            "real_samples": real,
            "min_real_samples": min_real,
        },
        "claim_scope": SCOPE_REAL if calibrated else SCOPE_SYNTHETIC,
        "production_ready": calibrated,
        "limitations": limitations,
    }


def decision_claim(stats: Dict, targets: Dict) -> Dict:
    """SCENARIO §6.5：分母不足 `min_accept_for_claim` 时只作暂定结论，不许宣称达标。

    每个指标按**自己的分母**判样本量：金额错值率的分母是金额类已接受字段，一份 10 件的批次里
    它可能只有 12 个，这时候整体达标说明不了金额那一维。所以 `no_sample`（分母为 0，点值 0.0
    是空位不是成绩）和 `provisional`（分母太小）都不算 `met`，总状态取最坏的一项。

    `not_applicable` 是第三种情况，也是原来最容易搞错的一种：按风险分组的 `free_text` 里根本没有
    金额字段，金额错值率对它不是「样本不够」而是「不该由它表态」。这类行排在判决阶梯的最末，
    不会把 348 格全对、本该 met 的分组拖成 no_sample；只有整级全是空位时才判出 not_applicable
    （这一级确实没什么可说）。母体键为 0 才算不适用；有金额字段却一个都没自动接受，
    仍是 no_sample（那一维确实不知道）。
    """
    floor = int(targets["min_accept_for_claim"])
    rows = []
    for name, kind, key, denominator, ci_key, population in CLAIM_METRICS:
        value = stats.get(name)
        count = int(stats.get(denominator) or 0)
        limit = targets[key]
        if int(stats.get(population) or 0) <= 0:
            status = "not_applicable"
        elif value is None or count <= 0:
            status = "no_sample"
        elif (value >= limit) if kind == "min" else (value <= limit):
            # 过了线还要看分母够不够：不够只能算暂定，点值漂亮也不许写成达标。
            status = "met" if count >= floor else "provisional"
        else:
            # 没过线就是没过，跟样本量无关——小样本豁免的是「宣称达标」，不是「承认失败」。
            status = "not_met"
        rows.append({"name": name, "value": value, "target": f"{kind}={limit}",
                     "denominator_name": denominator, "denominator": count,
                     "population_name": population, "population": int(stats.get(population) or 0),
                     "ci": stats.get(ci_key) if ci_key else None, "status": status})
    return {
        "status": min((row["status"] for row in rows), key=CLAIM_STATUS_RANK.index),
        "min_accept_for_claim": floor,
        "confidence": targets["claim_confidence"],
        "accept_count": int(stats.get("accept_count") or 0),
        "metrics": rows,
        "not_applicable": [row["name"] for row in rows if row["status"] == "not_applicable"],
        "note": "分母不足 min_accept_for_claim 时只作暂定结论；看区间下界，不看裸点值；"
                "本组没有的字段类型计 not_applicable，不拖累判决",
    }


def _decision_stats(records: List[Dict], targets: Optional[Dict] = None,
                    config: Optional[Dict] = None) -> Dict:
    total = len(records)
    accepted = [item for item in records if item["decision"] == "accept"]
    # §6.5 第一行的分母：必填字段。全部字段那一版仍照发（`field_accept_rate`），
    # 它是「这批件整体放行比例」，与「必填格达标没有」不是同一个数。
    required_records = [item for item in records if item.get("required")]
    required_accepted = [item for item in required_records if item["decision"] == "accept"]
    reviewed = [item for item in records if item["decision"] == "review"]
    monetary = [item for item in records if item["risk"] == "monetary"]
    monetary_accepted = [item for item in monetary if item["decision"] == "accept"]
    accepted_errors = [item for item in accepted if not item["correct"]]
    monetary_errors = [item for item in monetary_accepted if not item["correct"]]
    grounded = sum(1 for item in records if item["grounded"])
    accept_grounding_missing = sum(1 for item in accepted if not item["grounded"])
    correct_accepted = sum(1 for item in accepted if item["correct"])
    # §6.5 的「溯源缺失率」分母是全部字段、不是已接受字段：只报已接受那一路，
    # 一份坐标全丢、整份进复核的文档会显示 0% 缺失。
    grounding_missing = sum(1 for item in records if not item.get("grounding_present"))
    # §10.2 第 1 项的「平均置信度」：只对真有置信度的格子求均值，另报覆盖率——
    # 三个格子撑起来的均值与 1495 个格子的均值不是同一句话。
    confidences = [float(item["confidence"]) for item in records
                   if isinstance(item.get("confidence"), (int, float))
                   and not isinstance(item.get("confidence"), bool)]

    manual = [item for item in records if item.get("review_status")]
    vocab = action_vocab(config)
    corrected = ([item for item in manual
                  if item["review_status"] == vocab["corrected_state"]]
                 if vocab["corrected_state"] else [])
    manual_accept = ([item for item in manual
                      if item["review_status"] == vocab["confirming_state"]]
                     if vocab["confirming_state"] else [])
    unresolved = [item for item in reviewed if not item.get("review_status")]
    z = float(targets["z"]) if targets else 1.96

    confusion = {
        "correct_accept": correct_accepted,
        "wrong_accept": len(accepted_errors),
        "over_review": sum(1 for item in unresolved if item["correct"]),
        "unresolved_review": sum(1 for item in unresolved if not item["correct"]),
        "reviewed_corrected": len(corrected),
        "reviewed_accept_correct": sum(1 for item in manual_accept if item["correct"]),
        "reviewed_accept_wrong": sum(1 for item in manual_accept if not item["correct"]),
    }

    categories: Dict[str, Counter] = defaultdict(Counter)
    for item in records:
        if item["correct"]:
            continue
        category = error_category(item, config)
        categories[category]["total"] += 1
        categories[category][item["decision"]] += 1

    stats = {
        "field_count": total,
        "accept_count": len(accepted),
        "review_count": len(reviewed),
        "required_field_count": len(required_records),
        "required_accept_count": len(required_accepted),
        "auto_accept_rate": (round(len(required_accepted) / len(required_records), 4)
                             if required_records else 0.0),
        "auto_accept_rate_ci": wilson_interval(len(required_accepted), len(required_records), z),
        "field_accept_rate": round(len(accepted) / total, 4) if total else 0.0,
        "review_rate": round(len(reviewed) / total, 4) if total else 0.0,
        "accepted_error_count": len(accepted_errors),
        "accepted_error_rate": round(len(accepted_errors) / len(accepted), 4) if accepted else 0.0,
        "accepted_error_ci": wilson_interval(len(accepted_errors), len(accepted), z),
        "monetary_field_count": len(monetary),
        "monetary_accept_count": len(monetary_accepted),
        "monetary_accepted_error_count": len(monetary_errors),
        "monetary_accepted_error_rate": round(len(monetary_errors) / len(monetary_accepted), 4)
        if monetary_accepted else 0.0,
        "monetary_accepted_error_ci": wilson_interval(len(monetary_errors),
                                                      len(monetary_accepted), z),
        "accepted_grounding_missing": accept_grounding_missing,
        "grounding_missing_count": grounding_missing,
        "grounding_missing_rate": round(grounding_missing / total, 4) if total else None,
        "grounding_hit_rate": round(grounded / total, 4) if total else None,
        "grounding_hit_rate_ci": wilson_interval(grounded, total, z),
        "accepted_correct_rate": round(correct_accepted / len(accepted), 4) if accepted else None,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "confidence_coverage": round(len(confidences) / total, 4) if total else None,
        "reviewed_count": len(manual),
        "reviewed_accept_count": len(manual_accept),
        "reviewed_corrected_count": len(corrected),
        "pending_review_count": sum(1 for item in unresolved
                                    if item["decision"] == "review"),
        "correction_rate": round(len(corrected) / len(reviewed), 4) if reviewed else 0.0,
        "confusion": confusion,
        "error_categories": [
            {"category": name, "total": counter.get("total", 0),
             "accepted": counter.get("accept", 0), "review": counter.get("review", 0)}
            for name, counter in sorted(categories.items())
        ],
    }
    if targets:
        stats["claim"] = decision_claim(stats, targets)
    return stats


def evaluate_predictions(pred_dir: Path, labels_dir: Path, config: Dict,
                         include_records: bool = False, real_samples: int = 0) -> Dict:
    """Decision-level evaluation of an out/ directory against labels.

    ``include_records`` adds the flat per-field records so callers can regroup
    them (the hard set attributes by degradation). ``real_samples`` is what the
    caller honestly knows about real desensitised documents (SCENARIO §9.4).
    """
    documents, records = [], []
    targets = claim_targets(config)
    match = match_options(config)
    for label_path in sorted(labels_dir.glob("*.json")):
        doc_id = label_path.stem
        pred_path = pred_dir / doc_id / f"{doc_id}.json"
        if not pred_path.exists():
            print(f"missing prediction for {doc_id}, skipped")
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        reviews = read_review_records(pred_dir, doc_id)
        doc_records = []
        for path, gt in labels["fields"].items():
            spec = field_spec(labels["doc_type"], re.sub(r"\[\d+\]", "[i]", path), config) or {}
            field = pred["fields"].get(path)
            record = reviews.get(path)
            status = review_status(record, config) if record else None
            if field is None:
                doc_records.append({
                    "document_id": doc_id, "doc_type": labels["doc_type"],
                    "path": path, "field_path": re.sub(r"\[\d+\]", "[i]", path),
                    "risk": spec.get("risk", "unknown"),
                    "value_type": spec.get("type"), "decision": "review",
                    "correct": False, "grounded": False, "review_status": status,
                    "required": bool(spec.get("required")),
                    "grounding_present": False,
                    "gt": gt["value"], "pred": None, "reasons": ["MISSING_PREDICTION"],
                    "signals": {}, "extractor_id": None,
                })
                continue
            correct = values_match(spec, gt["value"], field.get("value"), match)
            grounded = grounding_ok(gt, field.get("grounding"), match)
            doc_records.append({
                "document_id": doc_id, "doc_type": labels["doc_type"],
                "path": path, "field_path": re.sub(r"\[\d+\]", "[i]", path),
                "risk": spec.get("risk", "unknown"),
                "value_type": spec.get("type"),
                "decision": field.get("decision"),
                "correct": bool(correct), "grounded": bool(grounded),
                "review_status": status,
                "required": bool(spec.get("required")),
                "grounding_present": _grounding_present(field.get("grounding")),
                "confidence": field.get("confidence"),
                "gt": gt["value"], "pred": field.get("value"),
                "reasons": field.get("review_reasons") or [],
                "signals": field.get("signals") or {},
                "extractor_id": field.get("extractor_id"),
            })
        records.extend(doc_records)
        # §6.5 的「文档全自动率」：全部必填字段都 accept 的文档才算一份。一个必填格
        # 进了复核，整份文档就还要人看一眼——按字段数平摊会把这件事平均掉。
        required_items = [item for item in doc_records if item.get("required")]
        fully_auto = bool(required_items) and all(item["decision"] == "accept"
                                                  for item in required_items)
        document_errors = []
        for item in doc_records:
            if item["correct"]:
                continue
            document_errors.append({**item, "category": error_category(item, config)})
        documents.append({
            "document_id": doc_id,
            "doc_type": labels["doc_type"],
            "page_count": int(pred.get("source", {}).get("page_count") or 0),
            "timing": pred.get("timing"),
            "transaction_match": transaction_comparison(labels, pred["fields"],
                                                        labels["doc_type"], config),
            "required_field_count": len(required_items),
            "document_fully_auto": fully_auto,
            **_decision_stats(doc_records, targets, config),
            "errors": document_errors[:80],
        })

    def grouped(key: str) -> List[Dict]:
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for item in records:
            buckets[item[key]].append(item)
        return sorted(({"name": name, **_decision_stats(items, targets, config)}
                       for name, items in buckets.items()),
                      key=lambda row: row["name"])

    transaction_docs = [doc["transaction_match"] for doc in documents if doc.get("transaction_match")]
    transaction_summary = None
    if transaction_docs:
        gt_rows = sum(item["gt_rows"] for item in transaction_docs)
        content_matched = sum(item["content_matched"] for item in transaction_docs)
        order_ok = sum(item["order_sensitive_correct"] for item in transaction_docs)
        transaction_summary = {
            "keys": list(transaction_keys(config)),
            "documents": len(transaction_docs),
            "gt_rows": gt_rows,
            "pred_rows": sum(item["pred_rows"] for item in transaction_docs),
            "order_sensitive_correct": order_ok,
            "order_sensitive_rate": round(order_ok / gt_rows, 4) if gt_rows else None,
            "content_matched": content_matched,
            "content_match_rate": round(content_matched / gt_rows, 4) if gt_rows else None,
        }

    total_seconds = sum(float((doc.get("timing") or {}).get("total_seconds") or 0.0)
                        for doc in documents)
    total_pages = sum(int(doc.get("page_count") or 0) for doc in documents)
    latency = {
        "documents": len(documents),
        "pages": total_pages,
        "total_seconds": round(total_seconds, 3),
        "avg_document_seconds": round(total_seconds / len(documents), 3) if documents else None,
        "avg_page_seconds": round(total_seconds / total_pages, 3) if total_pages else None,
        "note": "measured during docpipe.run; cached parses reduce parse_seconds (use --force for cold latency)",
    }

    summary = {
        **_decision_stats(records, targets, config),
        # §6.5 表格的最后一行的「记录，不设第一阶段门槛」：这个数此前根本没算。
        "document_fully_auto_count": sum(1 for doc in documents if doc["document_fully_auto"]),
        "document_auto_rate": (round(sum(1 for doc in documents if doc["document_fully_auto"])
                                     / len(documents), 4) if documents else None),
        "by_doc_type": grouped("doc_type"),
        # §10.2 第 2 项的「按字段路径」：按 schema 路径（数组下标折成 `[i]`）分组，
        # 否则一条流水表的 168 行交易会变成 168 个各看一眼的分组。
        "by_field_path": grouped("field_path"),
        "by_risk": grouped("risk"),
        "transactions": transaction_summary,
        "latency": latency,
    }
    result = {"summary": summary, "documents": documents,
              **report_provenance(labels_dir.name, config, real_samples)}
    if include_records:
        result["records"] = records
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate extraction or decisions against labels")
    parser.add_argument("--pred", default=None, help="out/ directory produced by docpipe.run")
    parser.add_argument("--labels", default=None, help="label directory, e.g. labels/holdout")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--extracted", default=None)
    parser.add_argument("--real-samples", type=int, default=0,
                        help="本次评估里真实脱敏样本的份数（SCENARIO §9.4，默认 0，不去扫用户目录）")
    args = parser.parse_args(argv)
    config = load_config()

    if args.pred and args.labels:
        result = evaluate_predictions(Path(args.pred), Path(args.labels), config,
                                      real_samples=args.real_samples)
        summary = result["summary"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        for row in summary["by_doc_type"]:
            print(f"{row['name']:<26} accept={row['auto_accept_rate']} "
                  f"n_required={row['required_field_count']} all={row['field_accept_rate']} "
                  f"err={row['accepted_error_rate']} monetary_err={row['monetary_accepted_error_rate']} "
                  f"claim={(row.get('claim') or {}).get('status')}")
        claim = summary.get("claim") or {}
        print(f"claim={claim.get('status')} accepted={claim.get('accept_count')} "
              f"floor={claim.get('min_accept_for_claim')} "
              f"confidence={claim.get('confidence')} "
              f"error_ci={summary.get('accepted_error_ci')}")
        for row in (claim.get("metrics") or []):
            print(f"  {row['name']:<30} {row['value']} vs {row['target']} "
                  f"n={row['denominator']} ci={row['ci']} -> {row['status']}")
        print(f"manual reviewed={summary['reviewed_count']} "
              f"corrected={summary['reviewed_corrected_count']} "
              f"pending={summary['pending_review_count']} "
              f"correction_rate={summary['correction_rate']}")
        print(f"confidence avg={summary['avg_confidence']} "
              f"coverage={summary['confidence_coverage']} "
              f"fields={summary['field_count']}")
        print(f"accept required={summary['required_accept_count']}/"
              f"{summary['required_field_count']} rate={summary['auto_accept_rate']} "
              f"all={summary['accept_count']}/{summary['field_count']} "
              f"rate_all={summary['field_accept_rate']}")
        print(f"grounding missing={summary['grounding_missing_count']} "
              f"rate={summary['grounding_missing_rate']} "
              f"documents fully_auto={summary['document_fully_auto_count']} "
              f"rate={summary['document_auto_rate']}")
        for row in summary.get("by_field_path") or []:
            if row["confusion"]["wrong_accept"] or row["confusion"]["unresolved_review"]:
                print(f"  path {row['name']:<34} n={row['field_count']} "
                      f"accept={row['accept_count']} wrong={row['confusion']['wrong_accept']} "
                      f"review={row['review_count']} "
                      f"unresolved={row['confusion']['unresolved_review']}")
        if summary.get("transactions"):
            txn = summary["transactions"]
            print(f"transactions rows={txn['gt_rows']} order_sensitive={txn['order_sensitive_rate']} "
                  f"content={txn['content_match_rate']}")
        if summary.get("latency"):
            lat = summary["latency"]
            print(f"latency total={lat['total_seconds']}s doc_avg={lat['avg_document_seconds']}s "
                  f"page_avg={lat['avg_page_seconds']}s")
        prov = result["provenance"]
        print(f"provenance code={str(prov['code_sha256'])[:12]} "
              f"config={str(prov['config_sha256'])[:12]} "
              f"samples={str(prov['sample_manifest_sha256'])[:12]} "
              f"split={prov['split']}")
        print(f"scope={result['claim_scope']} production_ready={result['production_ready']} "
              f"real_samples={prov['real_samples']}/{prov['min_real_samples']}")
        for line in result["limitations"]:
            print(f"  limitation: {line}")
        name = Path(args.labels).name
        out_path = ROOT / "work" / "eval" / f"decisions_{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", out_path)
        return

    extracted_dir = Path(args.extracted) if args.extracted else ROOT / "work" / "extracted"
    results = []
    for label_path in sorted((ROOT / "labels" / args.split).glob("*.json")):
        doc_id = label_path.stem
        extracted_path = extracted_dir / f"{doc_id}.json"
        if not extracted_path.exists():
            continue
        extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        results.append(compare_document(doc_id, args.split, extracted, labels, config))
    summary = summarize(results)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    for name, rates in summary["by_doc_type"].items():
        print(f"{name:<26} acc={rates['value_accuracy']} grounded={rates['grounding_rate']} "
              f"both={rates['correct_and_grounded_rate']} missing={rates['missing']}")
    for name, rates in summary["by_risk"].items():
        print(f"  risk {name:<18} acc={rates['value_accuracy']} both={rates['correct_and_grounded_rate']}")
    out_path = ROOT / "work" / "eval" / f"extract_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "documents": results,
               **report_provenance(args.split, config, args.real_samples)}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scope={payload['claim_scope']} production_ready={payload['production_ready']} "
          f"real_samples={payload['provenance']['real_samples']}"
          f"/{payload['provenance']['min_real_samples']}")
    for line in payload["limitations"]:
        print(f"  limitation: {line}")
    print("written:", out_path)



if __name__ == "__main__":
    main()
