"""SCENARIO §8.1 单文档 JSON 契约审计。

`summaryaudit` 管的是交付目录与汇总账本，这一层管每一份 `<id>/<id>.json` 自己——
键树、字段对象、金额写法、置信度精度、判决与原因码、时间戳、UTF-8 无 BOM、
「必填为空必须带 REQUIRED_EMPTY」，以及 §6.1 的七个置信度信号（键齐、值域、扩展登记）。
契约从 SCENARIO 现读，扩展键在配置里登记，代码里不写第二份允许清单。
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

SECTION = re.compile(r"###\s*8\.1\s*单文档 JSON.*?```json\s*\n(.*?)```", re.S)
REASONS_SECTION = re.compile(r"###\s*6\.4\s*硬性弃权.*?(?=\n###\s*6\.5)", re.S)
REASON_ROW = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]+)`\s*\|", re.M)
SIGNALS_SECTION = re.compile(r"###\s*6\.1\s*置信度信号.*?(?=\n###\s*6\.2)", re.S)
SIGNAL_ROW = re.compile(r"^\|\s*`([a-z][a-z0-9_]+)`\s*\|\s*([^|\n]+?)\s*\|", re.M)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
DECISIONS = ("accept", "review")


def section81_spec(text: str) -> Dict:
    match = SECTION.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def reason_codes(text: str) -> List[str]:
    section = REASONS_SECTION.search(text)
    if not section:
        return []
    return sorted(set(REASON_ROW.findall(section.group(0))))


def _field_spec(spec: Dict) -> Dict:
    for item in (spec.get("fields") or {}).values():
        if isinstance(item, dict):
            return item
    return {}


def _extra_keys(config: Dict) -> set:
    output = config.get("output", {}) or {}
    return set(output.get("document_extra_keys") or [])


def _allowed_reasons(config: Dict, spec_codes: List[str]) -> List[str]:
    validation = config.get("validation", {}) or {}
    extension = [str(item) for item in (validation.get("review_reason_extensions") or [])]
    return sorted(set(spec_codes) | set(extension))


def stale_reason_extensions(config: Dict, spec_codes: List[str]) -> List[Dict]:
    validation = config.get("validation", {}) or {}
    extension = [str(item) for item in (validation.get("review_reason_extensions") or [])]
    return [{"code": "reason_extension_stale", "detail": item}
            for item in extension if item in spec_codes]


def signal_spec(text: str) -> Dict[str, Dict]:
    """§6.1 的信号表：名字 → 允许的取值（区间或枚举，`空` 表示允许 null）。

    `ocr_conf` 那类的域写成「0～1」，`rule_exact` 写成「0/0.5/1」——两种都从表里
    现读，代码里不另存一份信号清单或值域。
    """
    section = SIGNALS_SECTION.search(text)
    if not section:
        return {}
    spec: Dict[str, Dict] = {}
    for name, domain in SIGNAL_ROW.findall(section.group(0)):
        values = sorted({float(item) for item in NUMBER.findall(domain)})
        if not values:
            continue
        spec[name] = {
            "values": values,
            "range": "～" in domain or "~" in domain,
            "null": "空" in domain,
        }
    return spec


def signal_extensions(config: Dict) -> List[str]:
    validation = config.get("validation", {}) or {}
    return [str(item) for item in (validation.get("signal_extensions") or [])]


def stale_signal_extensions(config: Dict, specs: Dict[str, Dict]) -> List[Dict]:
    return [{"code": "signal_extension_stale", "detail": name}
            for name in signal_extensions(config) if name in specs]


def signal_allowed(value, rule: Dict) -> bool:
    if value is None:
        return bool(rule["null"])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    low, high = rule["values"][0], rule["values"][-1]
    if rule["range"]:
        return low - 1e-9 <= value <= high + 1e-9
    return any(abs(value - item) <= 1e-9 for item in rule["values"])


def signal_problems(path: str, field: Dict, specs: Dict[str, Dict],
                    extensions: Sequence[str] = ()) -> List[Dict]:
    signals = field.get("signals")
    if not isinstance(signals, dict):
        return [{"code": "signals_shape", "detail": path}]
    problems = []
    for name, rule in specs.items():
        if name not in signals:
            problems.append({"code": "signal_missing", "detail": f"{path}.{name}"})
        elif not signal_allowed(signals[name], rule):
            problems.append({"code": "signal_out_of_domain",
                             "detail": f"{path}.{name}={signals[name]!r}"})
    for name in signals:
        if name not in specs and name not in extensions:
            problems.append({"code": "signal_extra", "detail": f"{path}.{name}"})
    return problems


def _paths(node, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    if isinstance(node, dict) and node:
        found: List[Tuple[str, ...]] = []
        for key, value in node.items():
            found.extend(_paths(value, prefix + (key,)))
        return found
    return [prefix]


def spec_leaves(spec: Dict) -> List[Tuple[str, ...]]:
    if not spec:
        return []
    leaves = []
    for key, value in spec.items():
        if key == "fields":
            continue
        leaves.extend(_paths(value, (key,)))
    return leaves


def _dig(payload, path: Sequence[str]) -> Tuple[bool, object]:
    node = payload
    for step in path:
        if not isinstance(node, dict) or step not in node:
            return False, None
        node = node[step]
    return True, node


def key_problems(doc: Dict, spec: Dict, extras: set) -> List[Dict]:
    problems = []
    for path in spec_leaves(spec):
        ok, _ = _dig(doc, path)
        if not ok and ".".join(path) not in extras:
            problems.append({"code": "key_missing", "detail": ".".join(path)})
    for key in doc:
        if spec and key not in spec and key not in extras:
            problems.append({"code": "key_extra", "detail": key})
    for group in ("source", "document_quality", "validation_summary", "summary"):
        value = doc.get(group)
        if not isinstance(value, dict) or not isinstance(spec.get(group), dict):
            continue
        for key in value:
            dotted = f"{group}.{key}"
            if key not in spec[group] and dotted not in extras:
                problems.append({"code": "key_extra", "detail": dotted})
    return problems


def field_problems(doc: Dict, spec: Dict, extras: set, reasons: List[str],
                   signals: Optional[Dict[str, Dict]] = None,
                   signal_ext: Sequence[str] = ()) -> List[Dict]:
    problems = []
    wanted = _field_spec(spec)
    grounding_keys = set((wanted.get("grounding") or {}).keys())
    field_extra = {item.split(".", 2)[2] for item in extras if item.startswith("fields.*.")}
    for path, field in (doc.get("fields") or {}).items():
        if not isinstance(field, dict):
            problems.append({"code": "field_shape", "detail": path})
            continue
        if signals:
            problems.extend(signal_problems(path, field, signals, signal_ext))
        for key in wanted:
            if key not in field:
                problems.append({"code": "field_key_missing", "detail": f"{path}.{key}"})
        for key in field:
            if wanted and key not in wanted and key not in field_extra:
                problems.append({"code": "field_key_extra", "detail": f"{path}.{key}"})
        grounding = field.get("grounding")
        if isinstance(grounding, dict):
            for key in grounding_keys:
                if key not in grounding:
                    problems.append({"code": "grounding_key_missing",
                                     "detail": f"{path}.grounding.{key}"})
        value = field.get("value")
        if isinstance(value, str) and not value.strip():
            problems.append({"code": "empty_string_value", "detail": path})
        if field.get("value_type") == "decimal" and value is not None:
            if not isinstance(value, str):
                problems.append({"code": "decimal_not_string", "detail": path})
        confidence = field.get("confidence")
        if isinstance(confidence, (int, float)) and abs(confidence * 1000 - round(confidence * 1000)) > 1e-6:
            problems.append({"code": "confidence_precision",
                             "detail": f"{path}={confidence}"})
        if field.get("decision") not in DECISIONS:
            problems.append({"code": "decision_invalid",
                             "detail": f"{path}={field.get('decision')}"})
        for reason in field.get("review_reasons") or []:
            if reasons and reason not in reasons:
                problems.append({"code": "reason_unknown", "detail": f"{path}={reason}"})
        if value is None and field.get("required") is True:
            if "REQUIRED_EMPTY" not in (field.get("review_reasons") or []):
                problems.append({"code": "null_reason_missing", "detail": path})
    return problems


def document_problems(doc: Dict, spec: Dict, config: Dict,
                      reasons: Optional[List[str]] = None, bom: bool = False,
                      signals: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    extras = _extra_keys(config)
    problems: List[Dict] = []
    if bom:
        problems.append({"code": "bom_present", "detail": "UTF-8 with BOM"})
    if not spec:
        problems.append({"code": "spec_unread", "detail": "SCENARIO §8.1"})
        return problems
    problems.extend(key_problems(doc, spec, extras))
    problems.extend(field_problems(doc, spec, extras, reasons or [], signals,
                                  signal_extensions(config)))

    stamp = doc.get("extraction_timestamp")
    offset = (config.get("output", {}) or {}).get("timestamp_offset", "+08:00")
    if isinstance(stamp, str) and not stamp.endswith(offset):
        problems.append({"code": "timestamp_offset", "detail": stamp})

    fields = doc.get("fields") or {}
    summary = doc.get("summary") or {}
    accept = sum(1 for item in fields.values()
                 if isinstance(item, dict) and item.get("decision") == "accept")
    review = sum(1 for item in fields.values()
                 if isinstance(item, dict) and item.get("decision") == "review")
    for key, want in (("field_count", len(fields)), ("accept_count", accept),
                      ("review_count", review)):
        if summary.get(key) != want:
            problems.append({"code": "summary_mismatch",
                             "detail": f"summary.{key}={summary.get(key)} 应={want}"})
    return problems


def document_files(out_dir: Path) -> List[Path]:
    files = []
    for path in sorted(out_dir.glob("*/*.json")):
        if path.stem == path.parent.name:
            files.append(path)
    return files


def audit(out_dir: Path, config: Optional[Dict] = None,
          scenario_text: Optional[str] = None, document: Optional[Dict] = None,
          reasons: Optional[List[str]] = None) -> Dict:
    config = config or {}
    text = scenario_text
    if text is None:
        text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    spec = section81_spec(text)
    spec_codes = reason_codes(text)
    signals = signal_spec(text)
    codes = reasons if reasons is not None else _allowed_reasons(config, spec_codes)
    problems: List[Dict] = []

    if not signals:
        problems.append({"code": "signal_spec_unread", "detail": "SCENARIO §6.1"})
    if document is not None:
        problems.extend(document_problems(document, spec, config, codes,
                                          signals=signals))
        return {"documents": 1, "fields": len(document.get("fields") or {}),
                "keys": len(spec_leaves(spec)), "signals": len(signals),
                "problems": problems}

    problems.extend(stale_reason_extensions(config, spec_codes))
    problems.extend(stale_signal_extensions(config, signals))
    files = document_files(out_dir)
    if not files:
        problems.append({"code": "documents_missing", "detail": str(out_dir)})
    fields = 0
    for path in files:
        raw = path.read_bytes()
        try:
            doc = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append({"code": "document_unreadable", "detail": f"{path.name}: {exc}"})
            continue
        fields += len(doc.get("fields") or {})
        if doc.get("document_id") != path.stem:
            problems.append({"code": "document_id_mismatch",
                             "detail": f"{path.stem} != {doc.get('document_id')}"})
        problems.extend(document_problems(doc, spec, config, codes,
                                          bom=raw[:3] == b"\xef\xbb\xbf",
                                          signals=signals))
    return {"documents": len(files), "fields": fields,
            "keys": len(spec_leaves(spec)), "signals": len(signals),
            "problems": problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit SCENARIO §8.1 document JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "scenario.yaml"))
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    result = audit(Path(args.out), config)
    print(f"文档 {result['documents']} 份 / 字段 {result['fields']} 个 / "
          f"契约叶子键 {result['keys']} / 信号 {result.get('signals')} / "
          f"问题 {len(result['problems'])} 条")
    for item in result["problems"][:20]:
        print(f"  ! {item['code']}: {item['detail']}")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
