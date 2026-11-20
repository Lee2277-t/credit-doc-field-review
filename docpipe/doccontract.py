"""SCENARIO §8.1/§6.4 的产物契约自检：原因码三张表同源 + 单文档 JSON 形状。

三张表指的是同一件事的三个侧面，任何一张漏一条都会让人在界面上看到裸码、或者让一条
证据白算：

* `trust.REVIEW_REASON_CODES` —— 代码真会发出的原因码（「谁在发」）；
* `hard_abstain_reasons` —— 出现就必须复核的码（「谁强制拦」）；
* `validation.review_actions` —— 每个码对应的「该去核对什么」（「谁带人话」）。

§8.1 允许本实现比 §6.4 多码（符号推断、局部补读之类是本实现特有的证据），但三个集合
之间的关系是固定的：§6.4 的 13 个码必须全在强制表里；发得出来的码必须有文案、必须
被强制（或落在文档化的自弃权清单 `SELF_ABSTAIN_CODES` 里，且那种码只在决策已经落成
review 之后才写）；文案表里不许留永远发不出来的死条目。

同一份文件还顺手核对单文档 JSON 的形状（§8.1 的约束段）：金额是字符串、置信度 3 位
小数、时间戳带 +08:00、必填为空必须带 REQUIRED_EMPTY、接受/复核只能是这两个词。

用法：`python -m docpipe.doccontract --out out/dev`（只读产物，不写任何文件）。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .trust import REVIEW_REASON_CODES, SELF_ABSTAIN_CODES

# SCENARIO §6.4（line 340）的原因码表。这张表是契约，不是配置——配置可以比它多。
SCENARIO_ABSTAIN_CODES = (
    "GROUNDING_MISSING", "LOW_OCR", "FORMAT_FAIL", "CROSS_FIELD_FAIL",
    "EXTRACTOR_CONFLICT", "AMBIGUOUS_UNIT", "AMBIGUOUS_DATE", "UNKNOWN_DOC_TYPE",
    "PARSE_LOW_QUALITY", "NON_CNY_CURRENCY", "REQUIRED_EMPTY", "TEMPLATE_MISMATCH",
    "CHAIN_UNRELIABLE",
)
DECISIONS = ("accept", "review")
VALIDATOR_STATUS = ("pass", "fail", "warn", "skip")
TOP_KEYS = ("schema_version", "document_id", "doc_type", "doc_type_confidence", "source",
            "extraction_timestamp", "pipeline_version", "document_quality", "fields",
            "validation_summary", "summary")
SOURCE_KEYS = ("file_name", "sha256", "page_count")
QUALITY_KEYS = ("parse_score", "ocr_mean_confidence", "warnings")
SUMMARY_KEYS = ("field_count", "accept_count", "review_count")
FIELD_KEYS = ("value", "raw_text", "value_type", "confidence", "decision", "grounding",
              "extractor_id", "validators", "review_reasons")
GROUNDING_KEYS = ("page", "bbox", "bbox_norm", "block_id", "char_span")
TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+08:00$")
VERSION = re.compile(r"^\d+\.\d+\.\d+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validation(config: Dict) -> Dict:
    return config.get("validation") or {}


def forced_codes(config: Dict) -> List[str]:
    return [str(code) for code in config.get("hard_abstain_reasons") or []]


def guidance_codes(config: Dict) -> List[str]:
    return [str(code) for code in (_validation(config).get("review_actions") or {})]


def check_reason_tables(config: Dict) -> List[str]:
    """三张表的关系是否还成立；返回人话问题清单（空 = 成立）。"""
    emitted = set(REVIEW_REASON_CODES)
    forced = set(forced_codes(config))
    guided = set(guidance_codes(config))
    self_abstain = set(SELF_ABSTAIN_CODES)
    problems: List[str] = []
    missing_by_scenario = [c for c in SCENARIO_ABSTAIN_CODES if c not in forced]
    if missing_by_scenario:
        problems.append(f"§6.4 的码没有强制复核条目: {missing_by_scenario}")
    no_action = sorted(emitted - guided)
    if no_action:
        problems.append(f"发得出来但没文案（界面只剩裸码）: {no_action}")
    dead_action = sorted(guided - emitted)
    if dead_action:
        problems.append(f"文案表里永远发不出来的码: {dead_action}")
    unforced = sorted(emitted - forced - self_abstain)
    if unforced:
        problems.append(f"发得出来却不强制复核: {unforced}")
    forced_not_emitted = sorted(forced - emitted)
    if forced_not_emitted:
        problems.append(f"强制表里代码发不出的码: {forced_not_emitted}")
    leaked = sorted(self_abstain & forced)
    if leaked:
        problems.append(f"自弃权码被写进坏证据清单（语义混淆）: {leaked}")
    return problems


EMPTY_VALUES = (None, "")


def _check_grounding(field: Dict) -> List[str]:
    """交付出去的格子必须有坐标（§7.1）；没抽到值的格子没有东西可定位。

    §8.1 写的是空值用 `null`。本实现的抽取层历史上用空字符串表示「没抽到」，
    四个出口都按 `in (None, "")` 判空，所以契约这里收两种写法；改成纯 `null` 要
    同时动抽取与四个出口，登记为后续单独立项，不在这一轮顺手改。
    """
    grounding = field.get("grounding")
    if (grounding is None and field.get("value") in EMPTY_VALUES
            and field.get("decision") != "accept"):
        return []
    if not isinstance(grounding, dict):
        return ["grounding 不是对象"]
    if field.get("value") in EMPTY_VALUES and field.get("decision") != "accept":
        return []
    missing = [key for key in GROUNDING_KEYS if key not in grounding]
    return [f"grounding 缺键 {missing}"] if missing else []


def check_field(field_id: str, field: Dict, emitted: set) -> List[str]:
    problems: List[str] = []
    missing = [key for key in FIELD_KEYS if key not in field]
    if missing:
        problems.append(f"{field_id}: 缺字段键 {missing}")
    if field.get("decision") not in DECISIONS:
        problems.append(f"{field_id}: decision={field.get('decision')!r}")
    confidence = field.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        problems.append(f"{field_id}: confidence 不是数字 {confidence!r}")
    elif round(float(confidence), 3) != float(confidence):
        problems.append(f"{field_id}: confidence 超过 3 位小数 {confidence!r}")
    if field.get("value_type") == "decimal" and field.get("value") is not None \
            and not isinstance(field.get("value"), str):
        problems.append(f"{field_id}: decimal 值不是字符串 {field.get('value')!r}")
    reasons = field.get("review_reasons")
    if not isinstance(reasons, list):
        problems.append(f"{field_id}: review_reasons 不是列表")
        reasons = []
    unknown = [str(reason) for reason in reasons if str(reason) not in emitted]
    if unknown:
        problems.append(f"{field_id}: 未知原因码 {unknown}")
    if field.get("decision") == "accept" and reasons:
        problems.append(f"{field_id}: 已接受却挂着理由 {reasons}")
    if field.get("value") in EMPTY_VALUES:
        if field.get("required") and "REQUIRED_EMPTY" not in reasons:
            problems.append(f"{field_id}: 必填为空却没有 REQUIRED_EMPTY")
        if not field.get("required") and field.get("decision") != "review":
            problems.append(f"{field_id}: 可选为空却被接受")
    for item in field.get("validators") or []:
        missing = [key for key in ("id", "status", "detail") if key not in item]
        if missing:
            problems.append(f"{field_id}: validator 缺键 {missing}")
        elif item.get("status") not in VALIDATOR_STATUS:
            problems.append(f"{field_id}: validator 状态 {item.get('status')!r}")
    problems.extend(f"{field_id}: {text}" for text in _check_grounding(field))
    return problems


def check_document(doc: Dict) -> List[str]:
    problems: List[str] = []
    missing = [key for key in TOP_KEYS if key not in doc]
    if missing:
        return [f"顶层缺键 {missing}"]
    if doc.get("schema_version") != "1.0":
        problems.append(f"schema_version={doc.get('schema_version')!r}")
    source = doc.get("source") or {}
    missing = [key for key in SOURCE_KEYS if key not in source]
    if missing:
        problems.append(f"source 缺键 {missing}")
    if not SHA256.match(str(source.get("sha256") or "")):
        problems.append(f"source.sha256 不是 64 位小写十六进制: {source.get('sha256')!r}")
    if not isinstance(source.get("page_count"), int) or int(source.get("page_count") or 0) < 1:
        problems.append(f"source.page_count={source.get('page_count')!r}")
    if not TIMESTAMP.match(str(doc.get("extraction_timestamp") or "")):
        problems.append(f"extraction_timestamp 不带 +08:00: {doc.get('extraction_timestamp')!r}")
    if not VERSION.match(str(doc.get("pipeline_version") or "")):
        problems.append(f"pipeline_version={doc.get('pipeline_version')!r}")
    quality = doc.get("document_quality") or {}
    missing = [key for key in QUALITY_KEYS if key not in quality]
    if missing:
        problems.append(f"document_quality 缺键 {missing}")
    if not isinstance(quality.get("warnings"), list):
        problems.append("document_quality.warnings 不是列表")
    fields = doc.get("fields")
    if not isinstance(fields, dict) or not fields:
        return problems + ["fields 为空或不是对象"]
    emitted = set(REVIEW_REASON_CODES)
    for field_id, field in fields.items():
        if not isinstance(field, dict):
            problems.append(f"{field_id}: 字段不是对象")
            continue
        problems.extend(check_field(field_id, field, emitted))
    summary = doc.get("summary") or {}
    missing = [key for key in SUMMARY_KEYS if key not in summary]
    if missing:
        problems.append(f"summary 缺键 {missing}")
    accept = sum(1 for field in fields.values() if field.get("decision") == "accept")
    review = sum(1 for field in fields.values() if field.get("decision") == "review")
    if summary.get("field_count") != len(fields):
        problems.append(f"summary.field_count={summary.get('field_count')} 实际 {len(fields)}")
    if summary.get("accept_count") != accept:
        problems.append(f"summary.accept_count={summary.get('accept_count')} 实际 {accept}")
    if summary.get("review_count") != review:
        problems.append(f"summary.review_count={summary.get('review_count')} 实际 {review}")
    if accept + review != len(fields):
        problems.append(f"判决不成对：accept {accept} + review {review} != {len(fields)}")
    validations = doc.get("validation_summary") or {}
    for key in ("hard_fail_count", "warn_count"):
        if not isinstance(validations.get(key), int):
            problems.append(f"validation_summary.{key} 不是整数")
    if not isinstance(validations.get("rules"), list):
        problems.append("validation_summary.rules 不是列表")
    return problems


def document_files(out_dir: Path) -> List[Path]:
    """只认 `<目录>/<同名>.json` 这种单文档产物，跳过 summary/台账/导出。"""
    found: List[Path] = []
    for child in sorted(out_dir.iterdir()) if out_dir.exists() else []:
        if not child.is_dir():
            continue
        candidate = child / f"{child.name}.json"
        if child.name in ("final", "render", "review", "work") or not candidate.exists():
            continue
        found.append(candidate)
    return found


def check_tree(out_dir: Path, config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    problems = check_reason_tables(config)
    files = document_files(Path(out_dir))
    observed: Dict[str, int] = {}
    field_count = 0
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # 坏文件本身就是要报的事，不该让整条腿崩
            problems.append(f"{path.name}: 读不出来 {exc}")
            continue
        problems.extend(f"{path.name}: {text}" for text in check_document(doc))
        for field in (doc.get("fields") or {}).values():
            field_count += 1
            for reason in field.get("review_reasons") or []:
                observed[str(reason)] = observed.get(str(reason), 0) + 1
    unmapped = sorted(set(observed) - set(guidance_codes(config)))
    if unmapped:
        problems.append(f"产物里出现但没有文案的原因码: {unmapped}")
    return {
        "documents": len(files),
        "fields": field_count,
        "reasons": dict(sorted(observed.items())),
        "problems": problems,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SCENARIO §8.1 单文档产物契约自检")
    parser.add_argument("--out", required=True, help="数据区目录（如 out/dev）")
    parser.add_argument("--min-documents", type=int, default=1,
                        help="少于这个份数按覆盖不足报错（默认 1）")
    args = parser.parse_args(argv)
    config = load_config()
    result = check_tree(Path(args.out), config)
    print(f"文档 {result['documents']} 份 / 格 {result['fields']} 个 / "
          f"原因码 {result['reasons']} / 问题 {len(result['problems'])} 条")
    thin = result["documents"] < args.min_documents
    if thin:
        print(f"覆盖不足：只有 {result['documents']} 份，下限 {args.min_documents}")
    for item in result["problems"]:
        print("  -", item)
    if thin or result["problems"]:
        print("契约不成立")
        return 1
    print("契约成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
