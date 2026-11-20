"""新增一类文档的接入契约审计。

`doc_types` 下加一个类型，真正要动的地方分散在五处：检测关键词、字段 schema、表格模板、
抽取器名字、合成构建器/样本清单，外加 `cross_document.rules.*.paths` 里对它的引用。
漏掉任何一处，流水线都不会报错——最坏的情况是整份文档被悄悄按另一种类型处理。

这里把这张清单变成可复查的账：每个类型一条记录，每个问题一个独立码，互不连坐。
口径：
- 只读配置、两张注册表（`extract.KNOWN_EXTRACTORS`、`synth.SYNTH_BUILDERS`）与
  `samples/manifest.json`；不解析 PDF、不读 `out/`、不写交付物。
- 样本与构建器只对 `evaluation.doc_type_contract.require_synth` 点名的类型强制。
  发票/缴款书目前只有 schema、抽取与单测夹具，没有合成版面：如实记 `not_required`，
  既不凭空判红，也不假装它们跑过端到端。
- `min_required_fields` 来自配置；检测门槛直接用 `parsing.doc_type.min_score`，
  能拿到多少分是关键词权重的算术和，不另设阈值。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, doc_types, field_specs, load_config

MODULE_VERSION = "0.1.0"
DEFAULTS = {
    "min_required_fields": 3,
    "require_synth": ["BUSINESS_LICENSE", "BANK_STATEMENT",
                      "INCOME_STATEMENT", "CREDIT_REPORT_ENTERPRISE"],
}

EMPTY_KEYWORDS = "empty_keywords"
BAD_WEIGHT = "bad_weight"
UNDETECTABLE = "undetectable"
EMPTY_FIELDS = "empty_fields"
THIN_REQUIRED = "thin_required"
PATH_PREFIX = "path_prefix"
MISSING_LABEL = "missing_label"
UNKNOWN_RISK = "unknown_risk"
UNKNOWN_ENUM = "unknown_enum"
EXTRACTOR_UNKNOWN = "extractor_unknown"
EXTRACTOR_PATH_MISMATCH = "extractor_path_mismatch"
ARRAY_MISMATCH = "array_mismatch"
DUPLICATE_PATH = "duplicate_path"
TABLE_UNKNOWN = "table_unknown"
TABLE_NO_HEADERS = "table_no_headers"
COLUMN_UNKNOWN = "column_unknown"
ROW_LABEL_UNKNOWN = "row_label_unknown"
NO_BUILDER = "no_builder"
NO_SAMPLES = "no_samples"
UNKNOWN_MANIFEST_TYPE = "unknown_manifest_type"
MANIFEST_MISSING = "manifest_missing"
DEAD_CROSS_PATH = "dead_cross_path"
DECLARED_PROBLEMS = frozenset({
    EMPTY_KEYWORDS, BAD_WEIGHT, UNDETECTABLE, EMPTY_FIELDS, THIN_REQUIRED,
    PATH_PREFIX, MISSING_LABEL, UNKNOWN_RISK, UNKNOWN_ENUM, EXTRACTOR_UNKNOWN,
    EXTRACTOR_PATH_MISMATCH, ARRAY_MISMATCH, DUPLICATE_PATH, TABLE_UNKNOWN,
    TABLE_NO_HEADERS, COLUMN_UNKNOWN, ROW_LABEL_UNKNOWN, NO_BUILDER, NO_SAMPLES,
    UNKNOWN_MANIFEST_TYPE, MANIFEST_MISSING, DEAD_CROSS_PATH,
})


def options(config: Dict) -> Dict:
    section = (config.get("evaluation") or {}).get("doc_type_contract") or {}
    return {
        "min_required_fields": int(section.get("min_required_fields", 3)),
        "require_synth": list(section.get("require_synth", [
            "BUSINESS_LICENSE", "BANK_STATEMENT", "INCOME_STATEMENT",
            "CREDIT_REPORT_ENTERPRISE"])),
    }


def _known_extractors() -> set:
    from .extract import KNOWN_EXTRACTORS
    return set(KNOWN_EXTRACTORS)


def _synth_builders() -> set:
    from .synth import SYNTH_BUILDERS
    return set(SYNTH_BUILDERS)


def _column_names(table: Dict, spec: Dict) -> set:
    names = set(table.get("headers") or [])
    names |= set(table.get("optional_headers") or [])
    names |= set((table.get("aliases") or {}).keys())
    names |= set((table.get("column_patterns") or {}).keys())
    names |= set((spec.get("column_enum") or {}).keys())
    names |= set(spec.get("fallback_columns") or [])
    names |= set((spec.get("direction_columns") or {}).keys())
    return names


def _check_field(doc_type: str, spec: Dict, tables: Dict, context: Dict,
                 problems: List[Dict]) -> None:
    def add(code: str, path: str, detail: str) -> None:
        problems.append({"code": code, "doc_type": doc_type,
                         "path": path, "detail": detail})

    path = str(spec.get("path") or "")
    risk = spec.get("risk")
    if risk not in context["risks"]:
        add(UNKNOWN_RISK, path, f"risk={risk!r} 不在 thresholds.risk_tiers")
    if spec.get("type") == "enum" and spec.get("enum") not in context["enums"]:
        add(UNKNOWN_ENUM, path, f"enum={spec.get('enum')!r} 不在 enums")

    extractor = spec.get("extractor")
    fallback = (spec.get("fallback") or {}).get("extractor")
    if extractor not in context["extractors"]:
        add(EXTRACTOR_UNKNOWN, path,
            f"extractor={extractor!r} 没有实现在 extract.py（已注册："
            f"{sorted(context['extractors'])}）")
    if fallback and fallback not in context["extractors"]:
        add(EXTRACTOR_UNKNOWN, path, f"fallback.extractor={fallback!r} 没有实现")

    if extractor == "table_cell":
        if "[i]" not in path:
            add(EXTRACTOR_PATH_MISMATCH, path, "声明 table_cell 但路径没有 [i]")
        for key in ("table", "column", "array"):
            if not spec.get(key):
                add(EXTRACTOR_PATH_MISMATCH, path, f"声明 table_cell 但缺 {key}")
        head = path.split("[i]")[0]
        if spec.get("array") and spec["array"] != head:
            add(ARRAY_MISMATCH, path, f"array={spec['array']!r} 与路径前缀 {head!r} 不一致")
    elif extractor == "table_row":
        if "[i]" in path:
            add(EXTRACTOR_PATH_MISMATCH, path, "声明 table_row 但路径带 [i]")
        for key in ("table", "row_label", "column"):
            if not spec.get(key):
                add(EXTRACTOR_PATH_MISMATCH, path, f"声明 table_row 但缺 {key}")
    elif "[i]" in path:
        add(EXTRACTOR_PATH_MISMATCH, path,
            f"路径带 [i] 但 extractor={extractor!r}（数组字段必须声明 table_cell）")

    table_name = spec.get("table")
    if not table_name:
        return
    table = tables.get(table_name)
    if table is None:
        add(TABLE_UNKNOWN, path, f"table={table_name!r} 不在 doc_types.{doc_type}.tables")
        return
    headers = list(table.get("headers") or [])
    if not headers:
        add(TABLE_NO_HEADERS, path, f"表 {table_name!r} 没有 headers")
    column = spec.get("column")
    if column and column not in _column_names(table, spec):
        add(COLUMN_UNKNOWN, path, f"column={column!r} 不在表 {table_name!r} 的列名里")
    row_label = spec.get("row_label")
    row_labels = set(table.get("row_labels") or [])
    if row_label and row_labels and row_label not in row_labels:
        add(ROW_LABEL_UNKNOWN, path,
            f"row_label={row_label!r} 不在表 {table_name!r} 的 row_labels 里")


def audit_type(doc_type: str, config: Dict, context: Dict,
               sample_count: Optional[int]) -> Dict:
    block = (config.get("doc_types") or {}).get(doc_type) or {}
    specs = field_specs(doc_type, config)
    problems: List[Dict] = []

    def add(code: str, detail: str, path: str = "") -> None:
        problems.append({"code": code, "doc_type": doc_type,
                         "path": path, "detail": detail})

    keywords = block.get("keywords") or {}
    if not keywords:
        add(EMPTY_KEYWORDS, "没有检测关键词，这个类型永远判不出来")
    bad = {name: weight for name, weight in keywords.items()
           if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0}
    if bad:
        add(BAD_WEIGHT, f"权重必须是正整数：{bad}")
    ceiling = sum(weight for weight in keywords.values()
                  if isinstance(weight, int) and not isinstance(weight, bool))
    if ceiling < context["min_score"]:
        add(UNDETECTABLE, f"关键词权重和={ceiling} < parsing.doc_type.min_score="
                          f"{context['min_score']}，最高分也过不了门")

    if not block.get("label"):
        add(MISSING_LABEL, "缺 label（中文名），界面与文档类型对账会两头对不上")
    if not specs:
        add(EMPTY_FIELDS, "没有字段 schema")
    required = [spec for spec in specs if spec.get("required")]
    if len(required) < context["min_required_fields"]:
        add(THIN_REQUIRED, f"必填字段 {len(required)} 个 < min_required_fields="
                           f"{context['min_required_fields']}")
    prefixes = {str(spec.get("path") or "").split(".")[0] for spec in specs}
    if len(prefixes) > 1:
        add(PATH_PREFIX, f"字段路径前缀不唯一：{sorted(prefixes)}")

    seen: Dict[str, int] = {}
    tables = block.get("tables") or {}
    for spec in specs:
        path = str(spec.get("path") or "")
        seen[path] = seen.get(path, 0) + 1
        _check_field(doc_type, spec, tables, context, problems)
    duplicates = sorted(path for path, count in seen.items() if count > 1)
    if duplicates:
        add(DUPLICATE_PATH, f"路径重复（后写的会覆盖先写的）：{duplicates}")

    if doc_type in context["require_synth"]:
        if doc_type not in context["builders"]:
            add(NO_BUILDER, "require_synth 点名了它，但 synth.SYNTH_BUILDERS 里没有构建器")
        if sample_count is not None and sample_count < 1:
            add(NO_SAMPLES, "require_synth 点名了它，但 samples/manifest.json 里 0 份")

    return {
        "doc_type": doc_type,
        "label": block.get("label"),
        "fields": len(specs),
        "required": len(required),
        "keywords": len(keywords),
        "keyword_score": ceiling,
        "tables": sorted(tables),
        "samples": sample_count,
        "synth_builder": doc_type in context["builders"],
        "synth_required": doc_type in context["require_synth"],
        "problems": problems,
    }


def audit(config: Optional[Dict] = None, manifest: Optional[Dict] = None,
          builders: Optional[set] = None, extractors: Optional[set] = None) -> Dict:
    """逐类型对账。`manifest` 传已解析的 samples/manifest.json；None 表示拿不到。"""
    config = config or load_config()
    opts = options(config)
    context = {
        "risks": set(((config.get("thresholds") or {}).get("risk_tiers") or {}).keys()),
        "enums": set((config.get("enums") or {}).keys()),
        "min_score": int(((config.get("parsing") or {}).get("doc_type", {})
                          .get("min_score", 3))),
        "min_required_fields": opts["min_required_fields"],
        "require_synth": set(opts["require_synth"]),
        "builders": _synth_builders() if builders is None else set(builders),
        "extractors": _known_extractors() if extractors is None else set(extractors),
    }

    problems: List[Dict] = []
    counts: Dict[str, int] = {}
    if manifest is None:
        problems.append({"code": MANIFEST_MISSING, "doc_type": "", "path": "",
                         "detail": "samples/manifest.json 不存在，样本覆盖无从对账"})
        counts[MANIFEST_MISSING] = 1
        sample_counts: Optional[Dict[str, int]] = None
    else:
        sample_counts = {}
        for item in manifest.get("documents") or []:
            doc_type = item.get("doc_type")
            if doc_type:
                sample_counts[doc_type] = sample_counts.get(doc_type, 0) + 1

    entries = []
    declared = sorted(doc_types(config))
    for doc_type in declared:
        entry = audit_type(doc_type, config, context,
                           None if sample_counts is None else sample_counts.get(doc_type, 0))
        entries.append(entry)
        for problem in entry["problems"]:
            counts[problem["code"]] = counts.get(problem["code"], 0) + 1

    if sample_counts is not None:
        for doc_type, count in sorted(sample_counts.items()):
            if doc_type not in declared:
                problems.append({"code": UNKNOWN_MANIFEST_TYPE, "doc_type": doc_type,
                                 "path": "", "detail": f"清单里有 {count} 份，配置里没有这个类型"})
                counts[UNKNOWN_MANIFEST_TYPE] = counts.get(UNKNOWN_MANIFEST_TYPE, 0) + 1

    declared_paths = {str(spec.get("path")) for doc_type in declared
                      for spec in field_specs(doc_type, config)}
    cross_checked = 0
    for rule_id, rule in sorted(((config.get("cross_document") or {}).get("rules") or {}).items()):
        for path in rule.get("paths") or []:
            cross_checked += 1
            if path not in declared_paths:
                problems.append({"code": DEAD_CROSS_PATH, "doc_type": "",
                                 "path": str(path),
                                 "detail": f"{rule_id} 引用的路径没有任何类型声明"})
                counts[DEAD_CROSS_PATH] = counts.get(DEAD_CROSS_PATH, 0) + 1

    for problem in problems:
        if problem["code"] not in DECLARED_PROBLEMS:
            raise ValueError(f"未登记的问题码：{problem['code']}")

    return {
        "version": MODULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_required_fields": opts["min_required_fields"],
        "require_synth": sorted(opts["require_synth"]),
        "type_count": len(entries),
        "field_count": sum(entry["fields"] for entry in entries),
        "types": entries,
        "cross_paths": {"checked": cross_checked},
        "manifest": {"documents": sum(sample_counts.values()) if sample_counts else 0,
                     "types": sample_counts or {}},
        "problems": counts,
        "problems_total": len(problems) + sum(len(entry["problems"]) for entry in entries),
        "details": problems,
        "skipped_types": sorted(entry["doc_type"] for entry in entries
                                if not entry["synth_required"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="文档类型接入契约审计（只读配置与样本清单，不写 out/）")
    parser.add_argument("--manifest", default=str(ROOT / "samples" / "manifest.json"))
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    config = load_config()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else None
    payload = audit(config, manifest)
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / "doctype_contract.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"类型={payload['type_count']} 字段={payload['field_count']} "
          f"问题={payload['problems_total']} 明细={payload['problems']} "
          f"跨文档路径={payload['cross_paths']['checked']} "
          f"无合成样本={payload['skipped_types']}")
    print("report:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
