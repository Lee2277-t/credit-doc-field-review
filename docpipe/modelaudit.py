"""SCENARIO §6.2「融合与校准」的产物侧对账。

§6.2 的六条规则里，前四条描述训练过程，后两条（小样本压上限 0.94、禁止把原始
softmax / OCR 置信度当最终 `confidence`）描述的是**每一个交付出去的数**。这两条在
代码里只有一处执行点：`trust.decide_field` 算完就把数写进产物，之后没有人回看这个
数是不是那个模型给的。换一个出口覆写 `confidence`、把 `calibration_samples` 换成
组样本数、拿原始分数冒充校准概率，产物上都只留下一个 0～0.99 的数，既有八问全绿。

这个模块只读校准产物与结果目录，逐格把 `confidence` 按产物自带的 coef /
intercept / isotonic / 样本数**独立重算一遍**，再把「按文档分组的 5 折」「小样本
上限」从产物自身读出来对账。

两类判定刻意分开：

- `PROBLEMS`：重算值与交付值对不上、特征向量与当前配置不同形、isotonic 不单调、
  basis 不在词表、交付样本数与产物对不上——这些是「这一格凭什么」，错了就是错了。
- `NOTES`：产物版本比代码旧、折数不足 5、折外覆盖不满、字段级样本不足走了组样本
  兜底——出厂产物本来如此，它们解释数字的来源，不当作缺陷。

口径：特征向量复用 `trust.field_features`（那是「特征怎么定义」的唯一出处），
**模型运算与样本数规则在本模块里独立重写**（不复用 `Calibrator.predict` /
`samples_for`），否则自己跟自己永远对得上。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .calibrate import CALIBRATE_VERSION
from .config import ROOT, load_config, resolve_path
from .trust import TRUST_VERSION, feature_names, field_features, heuristic_confidence, normalize_path

MODULE_VERSION = "0.1.0"

DEFAULTS = {
    "confidence_tolerance": 0.0005,
    "min_isotonic_points": 2,
    "oof_coverage_target": 1.0,
    "min_delivered_fields": 100,
}

ARTIFACT_MISSING = "artifact_missing"
THIN_CORPUS = "thin_corpus"
FEATURES_MISSING = "features_missing"
FEATURE_ORDER = "feature_order"
COEF_SHAPE = "coef_shape"
ISOTONIC_SHAPE = "isotonic_shape"
ISOTONIC_NOT_MONOTONE = "isotonic_not_monotone"
BASIS_UNKNOWN = "basis_unknown"
BASIS_MISMATCH = "basis_mismatch"
SAMPLES_MISMATCH = "samples_mismatch"
CONFIDENCE_MISSING = "confidence_missing"
CONFIDENCE_UNEXPLAINED = "confidence_unexplained"

VERSION_SKEW = "version_skew"
METHOD_SKEW = "method_skew"
FOLD_COUNT_BELOW_TARGET = "fold_count_below_target"
OOF_COVERAGE_BELOW_TARGET = "oof_coverage_below_target"
SAMPLE_COUNTING_GAP = "sample_counting_gap"

DECLARED_PROBLEMS = frozenset({
    ARTIFACT_MISSING, THIN_CORPUS, FEATURES_MISSING, FEATURE_ORDER, COEF_SHAPE,
    ISOTONIC_SHAPE, ISOTONIC_NOT_MONOTONE, BASIS_UNKNOWN,
    BASIS_MISMATCH, SAMPLES_MISMATCH, CONFIDENCE_MISSING, CONFIDENCE_UNEXPLAINED,
})
DECLARED_NOTES = frozenset({
    VERSION_SKEW, METHOD_SKEW, FOLD_COUNT_BELOW_TARGET, OOF_COVERAGE_BELOW_TARGET,
    SAMPLE_COUNTING_GAP,
})

# `trust.decide_field` 只会写这三个 basis；多出来一个说明有别的代码路径在写
# `confidence`，而这正是本模块要抓的东西。
BASES = ("calibrated", "calibrated_low_samples", "heuristic")


def problem(code: str, path: str, detail: str) -> Dict:
    if code not in DECLARED_PROBLEMS:
        raise ValueError(f"未登记的问题码 {code!r}：先加进 DECLARED_PROBLEMS 再报")
    return {"code": code, "path": path, "detail": detail}


def note(code: str, path: str, detail: str) -> Dict:
    if code not in DECLARED_NOTES:
        raise ValueError(f"未登记的说明码 {code!r}：先加进 DECLARED_NOTES 再报")
    return {"code": code, "path": path, "detail": detail}


def options(config: Dict) -> Dict:
    """`calibration.artifact_contract` 的取值；缺键退回 DEFAULTS（与出厂 YAML 逐字一致）。"""
    block = ((config.get("calibration") or {}).get("artifact_contract") or {})
    merged = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in block:
            merged[key] = block[key]
    return merged


def current_versions() -> Dict[str, str]:
    return {"calibrate_version": CALIBRATE_VERSION, "trust_version": TRUST_VERSION}


def interpolate(xs: List[float], ys: List[float], value: float) -> float:
    """isotonic 映射的线性插值；两端夹住。独立实现，不借 `Calibrator.predict`。"""
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            left, right = xs[index - 1], xs[index]
            if right <= left:
                return ys[index]
            ratio = (value - left) / (right - left)
            return ys[index - 1] + ratio * (ys[index] - ys[index - 1])
    return ys[-1]


def predict(artifact: Dict, features: List[float]) -> float:
    """产物里的模型：sigmoid(coef·x + intercept) → isotonic → 夹在 [0, 0.99]。"""
    coef = [float(item) for item in artifact.get("coef") or []]
    intercept = float(artifact.get("intercept") or 0.0)
    z = intercept + sum(c * x for c, x in zip(coef, features))
    z = max(-30.0, min(30.0, z))
    probability = 1.0 / (1.0 + math.exp(-z))
    curve = artifact.get("isotonic") or {}
    xs = [float(item) for item in curve.get("x") or []]
    ys = [float(item) for item in curve.get("y") or []]
    if xs and ys and len(xs) == len(ys):
        probability = interpolate(xs, ys, probability)
    return max(0.0, min(0.99, probability))


def sample_count(artifact: Dict, field: Dict, doc_type: str, config: Dict) -> Dict:
    """按产物记的 `sample_grouping` 复算「这一格用了多少条样本」。

    §6.2 第 5 条说的是**字段**的有效样本 < 30 就压上限；出厂 `field_auto` 在小样本
    字段上换用组样本数，于是上限分支走不到——这件事如实记成 `sample_counting_gap`。
    """
    calibration = config.get("calibration") or {}
    min_samples = int(calibration.get("min_samples_per_field", 30))
    grouping = str(artifact.get("sample_grouping")
                   or calibration.get("sample_grouping") or "field_auto")
    path = normalize_path(str(field.get("path") or ""))
    field_samples = int((artifact.get("field_samples") or {}).get(path, 0))
    group_samples = int((artifact.get("group_samples") or {}).get(doc_type, 0))
    if grouping == "doc_type":
        return {"samples": group_samples, "mode": "group",
                "field_samples": field_samples, "min_samples": min_samples}
    if grouping == "field_auto" and field_samples < min_samples:
        return {"samples": group_samples, "mode": "group",
                "field_samples": field_samples, "min_samples": min_samples}
    return {"samples": field_samples, "mode": "field",
            "field_samples": field_samples, "min_samples": min_samples}


def expected_confidence(artifact: Optional[Dict], field: Dict, doc_type: str,
                        config: Dict) -> Dict:
    if artifact is None:
        return {
            "confidence": heuristic_confidence(field, config, doc_type),
            "basis": "heuristic",
            "samples": 0,
            "mode": "heuristic",
            "field_samples": 0,
            "min_samples": int((config.get("calibration") or {}).get("min_samples_per_field", 30)),
        }
    counted = sample_count(artifact, field, doc_type, config)
    calibration = config.get("calibration") or {}
    value = predict(artifact, field_features(field, doc_type, config))
    basis = "calibrated"
    if counted["samples"] < counted["min_samples"]:
        value = min(value, float(calibration.get("low_sample_confidence_cap", 0.94)))
        basis = "calibrated_low_samples"
    return {
        "confidence": round(max(0.0, min(0.99, value)), 3),
        "basis": basis,
        "samples": counted["samples"],
        "mode": counted["mode"],
        "field_samples": counted["field_samples"],
        "min_samples": counted["min_samples"],
    }


def oof_coverage(folds: List[Dict]) -> Dict:
    """§6.2 第 3 条：折外分数要覆盖每份文档。被跳过的折会让一批文档只剩启发式。"""
    documents = set()
    covered = set()
    for fold in folds or []:
        documents.update(fold.get("train_docs") or [])
        documents.update(fold.get("test_docs") or [])
        covered.update(fold.get("test_docs") or [])
    ratio = (len(covered) / len(documents)) if documents else 0.0
    return {
        "documents": len(documents),
        "covered": len(covered),
        "ratio": round(ratio, 4),
        "uncovered": sorted(documents - covered),
    }


def artifact_problems(artifact: Optional[Dict], config: Dict,
                      opts: Optional[Dict] = None) -> Dict:
    """校准产物自身受不受 §6.2 约束：形状、方法、单调性、版本、折数与覆盖。"""
    opts = opts or options(config)
    details: List[Dict] = []
    notes: List[Dict] = []
    if not isinstance(artifact, dict) or not artifact.get("coef"):
        details.append(problem(ARTIFACT_MISSING, "artifact",
                               "校准产物缺失或没有 coef：所有字段只能落到启发式置信度"))
        return {"details": details, "notes": notes, "coverage": {}}

    expected_names = feature_names(config)
    names = artifact.get("features")
    if names is None:
        details.append(problem(FEATURES_MISSING, "artifact.features",
                               "产物没有记 features：没法核对特征顺序，也没法核对 coef 宽度"))
        width = len(artifact.get("coef") or [])
    else:
        names = [str(item) for item in names]
        width = len(names)
        if names != expected_names:
            index = next((position for position, pair in enumerate(zip(names, expected_names))
                          if pair[0] != pair[1]), min(len(names), len(expected_names)))
            got = names[index] if index < len(names) else "（缺）"
            want = expected_names[index] if index < len(expected_names) else "（多）"
            details.append(problem(
                FEATURE_ORDER, "artifact.features",
                f"第 {index} 维产物是 {got}、当前配置要 {want}；产物 {len(names)} 维、"
                f"当前 {len(expected_names)} 维。配置里的 risk_tiers / doc_types 改过就要重训，"
                f"否则 coef 会按位置错配"))
    if len(artifact.get("coef") or []) != width:
        details.append(problem(
            COEF_SHAPE, "artifact.coef",
            f"coef {len(artifact.get('coef') or [])} 个、特征向量 {width} 维："
            f"zip() 会静默截断，多出来的特征拿不到系数"))

    # 产物里的方法名由 `calibrate._artifact` 直接抄配置写回，所以「与配置不同」只可能
    # 是别的版本写下的——与版本偏斜同类，如实报、不追责。
    configured_method = str((config.get("calibration") or {}).get("method") or "")
    method = str(artifact.get("method") or "")
    if method != configured_method:
        notes.append(note(METHOD_SKEW, "artifact.method",
                          f"产物方法={method!r}、当前配置={configured_method!r}："
                          f"本版本按配置回写方法名，对不上说明产物是别的版本写下的"))

    curve = artifact.get("isotonic") or {}
    xs = [float(item) for item in curve.get("x") or []]
    ys = [float(item) for item in curve.get("y") or []]
    if len(xs) != len(ys) or len(xs) < int(opts["min_isotonic_points"]):
        details.append(problem(ISOTONIC_SHAPE, "artifact.isotonic",
                               f"x={len(xs)} y={len(ys)} 点，至少要 "
                               f"{int(opts['min_isotonic_points'])} 对"))
    else:
        if any(right < left for left, right in zip(xs, xs[1:])):
            details.append(problem(ISOTONIC_SHAPE, "artifact.isotonic", "isotonic.x 必须非递减"))
        if any(right < left for left, right in zip(ys, ys[1:])):
            details.append(problem(ISOTONIC_NOT_MONOTONE, "artifact.isotonic",
                                   "isotonic.y 递减：映射不再保序，置信度会被压反"))

    versions = current_versions()
    skew = {key: {"artifact": artifact.get(key), "code": versions[key]}
            for key in versions if artifact.get(key) != versions[key]}
    if skew:
        notes.append(note(VERSION_SKEW, "artifact.version",
                          "产物版本与当前代码不同（如实报、不追责）："
                          + "，".join(f"{key} 产物={item['artifact']} 代码={item['code']}"
                                      for key, item in sorted(skew.items()))))

    folds = artifact.get("cv_folds") or []
    target = int((config.get("calibration") or {}).get("n_splits", 5))
    if len(folds) < target:
        notes.append(note(FOLD_COUNT_BELOW_TARGET, "artifact.cv_folds",
                          f"记下 {len(folds)} 折、配置要 {target} 折：训练侧会跳过"
                          f"「训练集只剩一类」的折，这不是缺陷，但折外分数因此不是 5 折"))
    coverage = oof_coverage(folds)
    if coverage and coverage["ratio"] < float(opts["oof_coverage_target"]):
        notes.append(note(
            OOF_COVERAGE_BELOW_TARGET, "artifact.cv_folds",
            f"折外覆盖 {coverage['covered']}/{coverage['documents']} 份文档"
            f"（{coverage['ratio']:.2f}），未覆盖的 {coverage['uncovered']} 这些文档的折外"
            f"分数停在启发式，isotonic 是在混合样本上拟合的"))
    return {"details": details, "notes": notes, "coverage": coverage}


def field_problems(document: Dict, artifact: Optional[Dict], config: Dict,
                   opts: Optional[Dict] = None) -> Dict:
    """逐格重算 confidence，并与交付值、basis、样本数三样对账。"""
    opts = opts or options(config)
    details: List[Dict] = []
    counted = {"fields": 0, "low_sample_on_group": 0, "low_sample_paths": set()}
    document_id = str(document.get("document_id") or "")
    doc_type = str(document.get("doc_type") or "")
    for path, field in sorted((document.get("fields") or {}).items()):
        if not isinstance(field, dict):
            continue
        where = f"{document_id}.{path}"
        if field.get("confidence") is None:
            details.append(problem(CONFIDENCE_MISSING, where, "交付物里没有 confidence"))
            continue
        counted["fields"] += 1
        expected = expected_confidence(artifact, field, doc_type, config)
        basis = str(field.get("confidence_basis") or "")
        if basis not in BASES:
            details.append(problem(BASIS_UNKNOWN, where,
                                   f"confidence_basis={basis!r} 不在 {list(BASES)} 里"))
        elif basis != expected["basis"]:
            details.append(problem(BASIS_MISMATCH, where,
                                   f"产物说 basis={basis}，按产物自己重算应当是 "
                                   f"{expected['basis']}（样本 {expected['samples']} / "
                                   f"门槛 {expected['min_samples']}）"))
        delivered_samples = int(field.get("calibration_samples") or 0)
        if delivered_samples != expected["samples"]:
            details.append(problem(SAMPLES_MISMATCH, where,
                                   f"交付 samples={delivered_samples}、产物重算="
                                   f"{expected['samples']}（{expected['mode']} 口径）"))
        gap = abs(float(field["confidence"]) - expected["confidence"])
        if gap > float(opts["confidence_tolerance"]):
            details.append(problem(CONFIDENCE_UNEXPLAINED, where,
                                   f"交付 confidence={field['confidence']}、按产物重算="
                                   f"{expected['confidence']}（差 {gap:.4f}）：这个数不是"
                                   f"产物里的模型给的"))
        if expected["mode"] == "group" and expected["field_samples"] < expected["min_samples"]:
            counted["low_sample_on_group"] += 1
            counted["low_sample_paths"].add(normalize_path(path))
    return {"details": details, "counted": counted}


def load_documents(out_dir: Path) -> List[Dict]:
    documents: List[Dict] = []
    for path in sorted(Path(out_dir).iterdir()):
        payload = path / f"{path.name}.json"
        if not path.is_dir() or not payload.exists():
            continue
        try:
            document = json.loads(payload.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and document.get("fields"):
            documents.append(document)
    return documents


def audit(documents: List[Dict], artifact: Optional[Dict], config: Dict,
          opts: Optional[Dict] = None, *, allow_uncalibrated: bool = False) -> Dict:
    opts = opts or options(config)
    uncalibrated = allow_uncalibrated and artifact is None
    base = {"details": [], "notes": []} if uncalibrated else artifact_problems(artifact, config, opts)
    scoring_artifact = None if uncalibrated else (artifact or {})
    details = list(base["details"])
    notes = list(base["notes"])
    totals = {"documents": 0, "fields": 0, "low_sample_on_group": 0, "low_sample_paths": 0}
    low_paths = set()
    for document in documents:
        totals["documents"] += 1
        result = field_problems(document, scoring_artifact, config, opts)
        details.extend(result["details"])
        totals["fields"] += result["counted"]["fields"]
        totals["low_sample_on_group"] += result["counted"]["low_sample_on_group"]
        low_paths |= result["counted"]["low_sample_paths"]
    totals["low_sample_paths"] = len(low_paths)
    if totals["fields"] < int(opts["min_delivered_fields"]):
        details.append(problem(
            THIN_CORPUS, "fields",
            f"只读到 {totals['fields']} 格、下限 {int(opts['min_delivered_fields'])}："
            f"「0 违规」可能只是「0 文件」"))
    if totals["low_sample_on_group"]:
        notes.append(note(
            SAMPLE_COUNTING_GAP, "calibration.sample_grouping",
            f"{totals['low_sample_paths']} 条字段路径的字段级样本不足门槛，"
            f"{totals['low_sample_on_group']} 格按 field_auto 换用组样本数、因此不触发 "
            f"0.94 上限——这是出厂取舍（避免冷启动字段全被压进复核队列），如实报出"))
    problems = Counter(item["code"] for item in details)
    note_counts = Counter(item["code"] for item in notes)
    return {
        "module_version": MODULE_VERSION,
        "artifact": {key: (artifact or {}).get(key) for key in
                     ("calibrate_version", "trust_version", "method", "split",
                      "trained_at", "sample_grouping")},
        "current_versions": current_versions(),
        "options": opts,
        "totals": totals,
        "coverage": base.get("coverage") or {},
        "problems": dict(sorted(problems.items())),
        "problems_total": len(details),
        "notes": dict(sorted(note_counts.items())),
        "notes_total": len(notes),
        "details": details,
        "note_details": notes,
    }


def artifact_path(config: Dict) -> Path:
    """`calibration.artifact` 的绝对路径：与 `trust.Calibrator` 读的是同一个文件。"""
    path = Path(str((config.get("calibration") or {}).get("artifact")
                    or "work/calibration.json"))
    return path if path.is_absolute() else ROOT / path


def load_artifact(path: Path) -> Optional[Dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SCENARIO §6.2 融合与校准的产物侧对账（只读产物，不改判决）")
    parser.add_argument("--out", default=None, help="结果目录，默认 paths.out/dev")
    parser.add_argument("--artifact", default=None, help="校准产物，默认 calibration.artifact")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    config = load_config()
    out_dir = Path(args.out) if args.out else resolve_path(config["paths"]["out"], config) / "dev"
    source = Path(args.artifact) if args.artifact else artifact_path(config)
    artifact = load_artifact(source)
    documents = load_documents(out_dir)
    payload = audit(documents, artifact, config)
    payload["artifact_path"] = str(source)
    payload["out"] = str(out_dir)
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / "model_contract.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    totals = payload["totals"]
    print(f"产物={artifact_path} 结果={out_dir} 文档={totals['documents']} "
          f"字段={totals['fields']} 产物={payload['artifact']}")
    print(f"问题={payload['problems_total']} 明细={payload['problems']}")
    print(f"如实报={payload['notes_total']} 明细={payload['notes']}")
    for item in payload["details"][:20]:
        print(f"  {item['code']}: {item['path']} {item['detail']}")
    for item in payload["note_details"]:
        print(f"  [说明] {item['code']}: {item['detail']}")
    print("report:", json_path)
    return 0 if payload["problems_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
