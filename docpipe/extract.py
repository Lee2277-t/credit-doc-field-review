"""Schema-driven field extraction with page/bbox grounding.

Every field in config/scenario.yaml is resolved by its configured extractor:
label_value, regex, period_range, unit, table_cell, table_row.

Output per field:
  path, label, value_type, value, raw_text, confidence, extractor_id,
  grounding {page, bbox, bbox_norm, block_id, char_span}, meta
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ROOT, all_field_labels, field_specs, load_config
from .parse import (extract_header_matrix, extract_period_range, extract_regex,
                    extract_unit, locate_label_region, locate_label_value)
from .textutil import (decimal_str, normalize_code, normalize_text, parse_date,
                       parse_decimal, parse_int)

EXTRACTOR_VERSION = "0.1.0"

# 配置里 `extractor:` 能写的名字。这张表是「声明」与「实现」之间的唯一对账口径：
# docpipe.doctypeaudit 要求每个字段声明的名字都在这里，而这里的每个名字都必须在下面
# 真的有分支。过去 table_cell 只写在配置里没人读——分派靠路径里有没有 `[i]`，字段写
# `extractor: label_value` 却带 `[i]` 也照样按表格取数，声明成了装饰。
SCALAR_EXTRACTORS = frozenset({"label_value", "label_region", "period_range",
                               "regex", "unit", "header_matrix"})
TABLE_EXTRACTORS = frozenset({"table_cell", "table_row"})
KNOWN_EXTRACTORS = SCALAR_EXTRACTORS | TABLE_EXTRACTORS


def _compact(text) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


def _page_index(parsed: Dict) -> Dict[int, Dict]:
    return {page["page"]: page for page in parsed["pages"]}


def _grounding(page: Dict, bbox: Optional[List[float]], block_id: str,
               char_span: Optional[List[int]]) -> Optional[Dict]:
    if not bbox:
        return None
    width, height = page.get("width") or 0, page.get("height") or 0
    norm = [round(bbox[0] / width, 4), round(bbox[1] / height, 4),
            round(bbox[2] / width, 4), round(bbox[3] / height, 4)] if width and height else None
    return {
        "page": page["page"],
        "bbox": [round(value, 2) for value in bbox],
        "bbox_norm": norm,
        "block_id": block_id,
        "char_span": char_span,
    }


def _find_on_pages(parsed: Dict, finder) -> Optional[Tuple[Dict, Dict]]:
    for page in parsed["pages"]:
        hit = finder(page["lines"])
        if hit:
            return page, hit
    return None


def enum_matches(raw: str, enum_name: str, config: Dict) -> List[str]:
    """原文能落到哪些规范值：0 条 = 不在枚举里，2 条以上 = 映射不唯一。"""
    table = (config.get("enums") or {}).get(enum_name) or {}
    target = _compact(raw).upper()
    if not target:
        return []
    found: List[str] = []
    for canonical, aliases in table.items():
        names = [str(canonical)] + [str(alias) for alias in (aliases or [])]
        if any(_compact(name).upper() == target for name in names):
            if str(canonical) not in found:
                found.append(str(canonical))
    return found


def normalize_enum(raw: str, enum_name: str, config: Dict) -> Optional[str]:
    """SCENARIO §4.2：只接受配置里的枚举，映射不唯一时不许挑第一个。"""
    matches = enum_matches(raw, enum_name, config)
    return matches[0] if len(matches) == 1 else None


def _doc_unit(parsed: Dict, config: Dict) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    found = _find_on_pages(parsed, extract_unit)
    if not found:
        return None, None, None
    _, hit = found
    unit = normalize_enum(hit["value"], "unit", config)
    factor = {"CNY_YUAN": 1, "CNY_WAN_YUAN": 10000,
              "CNY_MILLION": 1000000}.get(unit)
    return unit, factor, str(hit.get("value") or "").strip()


def placeholders(config: Dict) -> Tuple[str, ...]:
    """SCENARIO §4.1 规则 2：整格只印了占位符时，它不是一个值。

    清单在 `validation.placeholders` 里；这里只做整格相等比较，所以词内出现
    （「生产--销售」）不会被误伤。代码里不认任何一个占位符字面量。
    """
    values = (config.get("validation") or {}).get(
        "placeholders", ["-", "--", "—", "***", "无", "不适用", "/", "空"])
    return tuple(normalize_text(item) for item in values)


def _normalize(raw: str, spec: Dict, config: Dict, context: Dict,
               row_cells: Optional[Dict] = None) -> Tuple[object, str, Dict]:
    """Return (normalized_value, status, meta)."""
    meta: Dict = {}
    if raw is None or not str(raw).strip():
        return None, "empty", meta
    text = normalize_text(raw)
    value_type = spec.get("type", "string")
    if value_type == "string" and spec.get("strip_pattern"):
        text = re.sub(spec["strip_pattern"], "", text).strip()
    if value_type == "enum":
        # §4.2 的枚举映射优先于 §4.1 的占位符折叠：`flag` 把 `-`/`无` 明确映射成
        # unknown/no，那是文档自己的词汇；占位符规则管的是没有词表可归的自由文本格。
        matches = enum_matches(text, spec.get("enum"), config) if spec.get("enum") else []
        if len(matches) == 1:
            return matches[0], "ok", meta
        if text in placeholders(config):
            meta["placeholder"] = text
            return None, "placeholder", meta
        meta["parse_error"] = "enum"
        if len(matches) > 1:
            meta["enum_ambiguous"] = sorted(matches)
        return text, "unmapped", meta
    if text in placeholders(config):
        meta["placeholder"] = text
        return None, "placeholder", meta

    if value_type == "string":
        value = normalize_code(text) if spec.get("risk") == "identity" else text
        return value, "ok", meta

    if value_type == "date":
        parsed = parse_date(text)
        if parsed is None:
            meta["parse_error"] = "date"
            return None, "parse_failed", meta
        return parsed.isoformat(), "ok", meta

    if value_type == "integer":
        parsed = parse_int(text)
        if parsed is None:
            meta["parse_error"] = "integer"
            return None, "parse_failed", meta
        return parsed, "ok", meta

    if value_type == "decimal":
        number_text = text
        for suffix in spec.get("strip_suffix") or []:
            if suffix and number_text.endswith(suffix):
                number_text = number_text[:-len(suffix)].strip()
        number = parse_decimal(number_text)
        if number is None:
            meta["parse_error"] = "decimal"
            return None, "parse_failed", meta
        if spec.get("unit_from_document"):
            meta["unit_from_document"] = True
            factor = context.get("unit_factor")
            if factor:
                number = (number * Decimal(factor)).quantize(Decimal("0.01"))
                meta["unit_factor"] = factor
                if context.get("unit_text"):
                    meta["unit_text"] = context["unit_text"]
            else:
                meta["unit_unknown"] = True
        if spec.get("signed_by_direction") and row_cells is not None:
            direction_text = row_cells.get("收支方向", {}).get("text", "")
            direction = normalize_enum(direction_text, "direction", config)
            if direction is None:
                # 借贷双列版式没有方向字：哪一列有金额就取哪个方向，
                # 但两列都有值时绝不猜（_extract_table_cell 会先拦下这种行）。
                for column, label in (spec.get("direction_columns") or {}).items():
                    if str((row_cells.get(column) or {}).get("text") or "").strip():
                        direction = normalize_enum(label, "direction", config)
                        if direction:
                            meta["direction_column"] = column
                            break
            meta["direction"] = direction
            printed_direction = str(direction_text or "").strip()
            if direction is not None and printed_direction:
                meta["direction_text"] = printed_direction
            explicit_negative = bool(re.search(r"[-−]|\(.+\)|（.+）|[负]", text)) or number < 0
            if number == 0:
                meta["zero_amount"] = True
            elif number > 0 and direction == "expense":
                number = -number
                meta["sign_source"] = "direction"
            else:
                meta["sign_source"] = "text"
                if number < 0 and direction == "income":
                    meta["sign_conflict"] = True
        return decimal_str(number), "ok", meta

    return text, "ok", meta


def _base_confidence(source_confidence: float, status: str, meta: Dict) -> float:
    confidence = float(source_confidence or 0.0)
    if status in {"parse_failed", "unmapped"}:
        confidence = min(confidence, 0.5)
    if meta.get("sign_source") == "direction":
        confidence -= 0.03
    if meta.get("sign_conflict"):
        confidence -= 0.08
    if meta.get("unit_unknown"):
        confidence -= 0.1
    return round(max(0.0, min(0.99, confidence)), 4)


def _field(spec: Dict, value, status: str, raw_text: str, confidence: float,
           extractor_id: str, grounding: Optional[Dict], meta: Dict,
           table_score: float = 1.0) -> Dict:
    rule_exact = 1.0 if status == "ok" else 0.0
    if status == "ok" and meta.get("sign_source") == "direction":
        rule_exact = 0.5
    return _field_payload(spec, value, status, raw_text, confidence, extractor_id,
                          grounding, meta, table_score, rule_exact)


def _field_payload(spec: Dict, value, status: str, raw_text: str, confidence: float,
                   extractor_id: str, grounding: Optional[Dict], meta: Dict,
                   table_score: float, rule_exact: float) -> Dict:
    # §8.1：未知/可选字段为空时 `value` 为 null。抽取层这一处把空白串折成 None，四个出口
    # 就只需要处理一种「空」——JSON 写 null、CSV 写空单元格、网页写「（未识别）」。
    if isinstance(value, str) and not value.strip():
        value = None
    payload = {
        "path": spec["path"],
        "label": spec.get("label", ""),
        "value_type": spec.get("type", "string"),
        "required": bool(spec.get("required")),
        "risk": spec.get("risk", ""),
        "value": value,
        "raw_text": normalize_text(raw_text),
        "status": status,
        "confidence": confidence,
        "extractor_id": extractor_id,
        "grounding": grounding,
        "signals": {
            "ocr_conf": round(float(confidence or 0.0), 4),
            "rule_exact": rule_exact,
            "format_valid": None,
            "table_score": round(float(table_score), 4),
            "cross_field_score": None,
            "extractor_agreement": None,
            "grounding_present": 1 if grounding else 0,
        },
    }
    if meta:
        payload["meta"] = meta
    return payload


def _extract_scalar(parsed: Dict, spec: Dict, config: Dict, context: Dict) -> Dict:
    doc_type = parsed["doc_type"]
    stop_labels = all_field_labels(doc_type, config)
    extractor = spec["extractor"]
    hit = None
    page = None
    extractor_id = ""

    if extractor == "label_value":
        found = _find_on_pages(
            parsed,
            lambda lines: locate_label_value(lines, spec["label"], spec.get("aliases"), stop_labels))
        extractor_id = f"label_value:{spec['label']}"
    elif extractor == "label_region":
        options = {key: spec[key] for key in
                   ("x_slack", "row_overlap", "center_slack", "max_value_parts",
                    "value_join_gap", "value_index", "max_label_parts", "label_join_gap",
                    "allow_below", "below_max_gap", "below_min_overlap",
                    "preserve_spaces")
                   if key in spec}
        found = _find_on_pages(
            parsed,
            lambda lines: locate_label_region(
                lines, spec["label"], spec.get("aliases"), stop_labels,
                occurrence=int(spec.get("occurrence", 1)),
                strip_prefix=spec.get("value_strip_prefix"),
                require_delimiter=bool(spec.get("require_delimiter", False)),
                **options))
        extractor_id = f"label_region:{spec['label']}"
    elif extractor == "period_range":
        part = spec.get("part", "start")
        found = _find_on_pages(
            parsed,
            lambda lines: extract_period_range(lines, spec["label"], spec.get("aliases"),
                                               part, stop_labels))
        extractor_id = f"period_range:{part}:{spec['label']}"
    elif extractor == "regex":
        found = _find_on_pages(
            parsed,
            lambda lines: extract_regex(lines, spec["pattern"], spec.get("label"),
                                        spec.get("aliases"), int(spec.get("group", 0))))
        # 配了 label 就不许退化成整页正则：同页另一个同形号码（中征码、账号）
        # 会顶替成"标签缺失也能取到"的值。模式自带锚点（如「小写」）的字段
        # 显式开 allow_unlabeled。
        if not found and spec.get("label") and spec.get("allow_unlabeled"):
            found = _find_on_pages(
                parsed, lambda lines: extract_regex(lines, spec["pattern"], group=int(spec.get("group", 0))))
        extractor_id = f"regex:{spec.get('label', spec['pattern'])}"
    elif extractor == "unit":
        found = _find_on_pages(parsed, extract_unit)
        extractor_id = "unit"
    elif extractor == "header_matrix":
        found = _find_on_pages(
            parsed,
            lambda lines: extract_header_matrix(
                lines,
                section=spec.get("section"),
                header_sequence=spec.get("header_sequence"),
                column_labels=spec.get("column_labels"),
                row_label=spec.get("row_label"),
                value_index=int(spec.get("value_index", 0)),
                max_header_lines=int(spec.get("max_header_lines", 3)),
                max_row_gap=int(spec.get("max_row_gap", 8))))
        extractor_id = f"header_matrix:{spec.get('row_label', '')}:{int(spec.get('value_index', 0))}"
    else:
        return _field(spec, None, "unsupported", "", 0.0,
                      f"unsupported:{extractor}", None, {})

    if not found:
        fallback = spec.get("fallback") or {}
        if fallback:
            merged = dict(spec)
            merged.pop("fallback", None)
            merged.update(fallback)
            result = _extract_scalar(parsed, merged, config, context)
            if result.get("status") != "missing":
                result.setdefault("meta", {})["extractor_fallback"] = str(
                    fallback.get("extractor") or "")
                return result
        return _field(spec, None, "missing", "", 0.0, extractor_id, None, {})

    page, hit = found
    raw = hit["value"]
    value, status, meta = _normalize(raw, spec, config, context)
    if extractor == "header_matrix":
        meta["matrix_fallback"] = True
    line_index = hit.get("line_index")
    lines = page.get("lines") or []
    if isinstance(line_index, int) and 0 <= line_index < len(lines) \
            and lines[line_index].get("recovered"):
        meta["local_recrop"] = True
    confidence = _base_confidence(hit.get("confidence"), status, meta)
    block_id = f"p{page['page']}-l{int(hit.get('line_index', 0)) + 1}"
    grounding = _grounding(page, hit.get("bbox"), block_id, hit.get("char_span"))
    return _field(spec, value, status, raw, confidence, extractor_id, grounding, meta)


def table_score(table: Dict, table_spec: Dict) -> float:
    expected = table_spec["headers"]
    header_cov = len([header for header in expected if header in table["headers"]]) / len(expected)
    rows = table["rows"]
    if not rows:
        return header_cov * 0.5
    # Templates legitimately leave optional columns blank (e.g. 上期金额),
    # so only score columns that are rendered on most rows.
    core = [header for header in expected
            if sum(1 for row in rows if header in row["cells"]) >= 0.5 * len(rows)]
    complete = sum(1 for row in rows if core and all(header in row["cells"] for header in core))
    row_cov = complete / len(rows)
    return header_cov * (0.5 + 0.5 * row_cov)


def _table_rows(parsed: Dict, table_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for page in parsed["pages"]:
        for table in page["tables"]:
            if table["name"] != table_name:
                continue
            for row in table["rows"]:
                rows.append({"page": page, "table": table, "row": row})
    return rows


def _table_row_anomalies(table: Dict, row_index: int) -> List[str]:
    """Anomaly reasons touching this row; entries without ``rows`` hit every row."""
    hits: List[str] = []
    for anomaly in table.get("anomalies") or []:
        rows = anomaly.get("rows") or []
        if not rows or row_index in rows:
            hits.append(anomaly["reason"])
    return hits


def _mark_row_anomalies(meta: Dict, table: Dict, row_index: int) -> None:
    """Attach the row's anomaly reasons, plus which columns a merge swallowed.

    ``table_anomaly`` stays a plain list of reason codes (trust decisions and the
    contract only look at that); ``table_anomaly_columns`` is review-screen text so a
    human sees 「项目名称 + 规格型号」 without reopening the parsed JSON.
    """
    reasons = _table_row_anomalies(table, row_index)
    if not reasons:
        return
    meta["table_anomaly"] = reasons
    groups = [anomaly["columns"] for anomaly in table.get("anomalies") or []
              if anomaly.get("columns") and anomaly["reason"] == "column_merge"
              and (not anomaly.get("rows") or row_index in anomaly["rows"])]
    if groups:
        meta["table_anomaly_columns"] = groups


def _extract_table_cell(parsed: Dict, spec: Dict, config: Dict, context: Dict) -> Dict[str, Dict]:
    rows = _table_rows(parsed, spec["table"])
    fields: Dict[str, Dict] = {}
    extractor_id = f"table_cell:{spec['table']}:{spec['column']}"
    for index, entry in enumerate(rows):
        page, table, row = entry["page"], entry["table"], entry["row"]
        path = spec["path"].replace("[i]", f"[{index}]")
        row_index = int(row.get("row_index", index))
        cell = row["cells"].get(spec["column"])
        source_column = spec["column"]
        if cell is not None and not str(cell.get("text") or "").strip():
            cell = None
        if cell is None and spec.get("fallback_columns"):
            filled = [name for name in spec["fallback_columns"]
                      if str((row["cells"].get(name) or {}).get("text") or "").strip()]
            if len(filled) > 1:
                # 借贷两列同时有数：这一行要么版式坏了，要么两行被并成一行。
                # 取哪一列都会猜错符号，直接留空交给人。
                meta = {"row_index": index, "ambiguous_columns": filled}
                _mark_row_anomalies(meta, table, row_index)
                fields[path] = _field(spec, None, "missing", "", 0.0, extractor_id, None, meta)
                fields[path]["path"] = path
                continue
            if filled:
                source_column = filled[0]
                cell = row["cells"][source_column]
        if cell is None and spec.get("column_enum"):
            filled = [name for name in spec["column_enum"]
                      if str((row["cells"].get(name) or {}).get("text") or "").strip()]
            if len(filled) != 1:
                meta = {"row_index": index}
                if len(filled) > 1:
                    meta["ambiguous_columns"] = filled
                _mark_row_anomalies(meta, table, row_index)
                fields[path] = _field(spec, None, "missing", "", 0.0, extractor_id, None, meta)
                fields[path]["path"] = path
                continue
            source_column = filled[0]
            evidence = row["cells"][source_column]
            # 方向不是印出来的字，是「哪一列有金额」这一版面事实；坐标仍指向
            # 那笔金额，溯源不凭空。
            cell = {"text": spec["column_enum"][source_column],
                    "bbox": evidence.get("bbox"),
                    "confidence": evidence.get("confidence")}
        if not cell:
            meta = {"row_index": index}
            _mark_row_anomalies(meta, table, row_index)
            fields[path] = _field(spec, None, "missing", "", 0.0, extractor_id, None, meta)
            fields[path]["path"] = path
            continue
        value, status, meta = _normalize(cell["text"], spec, config, context, row["cells"])
        if source_column != spec["column"]:
            meta["source_column"] = source_column
            meta["derived_from"] = "column_layout"
        _mark_row_anomalies(meta, table, row_index)
        confidence = _base_confidence(cell.get("confidence"), status, meta)
        grounding = _grounding(page, cell.get("bbox"),
                               f"{table['table_id']}-r{row['row_index']}",
                               [0, len(cell["text"])])
        score = table_score(table, config["doc_types"][parsed["doc_type"]]["tables"][spec["table"]])
        fields[path] = _field(spec, value, status, cell["text"], confidence,
                              extractor_id, grounding, meta, score)
        fields[path]["path"] = path
    return fields


def _statement_header_anchors(line: Dict, options: Dict) -> List[float]:
    """Right edges of the year columns in a header such as ``附注 2024 年度 2023 年度``.

    Text layers often split ``2024`` and ``年度`` into two boxes; the following
    suffix is merged back so the anchor matches the right-aligned amount below it.
    """
    pattern = re.compile(options.get("header_pattern") or r"^20\d{2}\s*(?:年度|年)?$")
    merge_tokens = {_compact(item) for item in options.get("header_merge") or []}
    gap_limit = float(options.get("header_merge_gap", 12))
    parts = sorted(line.get("parts") or [], key=lambda part: part["bbox"][0])
    anchors: List[float] = []
    for index, part in enumerate(parts):
        text = normalize_text(part.get("text") or "").strip()
        if not pattern.fullmatch(text):
            continue
        right = float(part["bbox"][2])
        if index + 1 < len(parts):
            following = parts[index + 1]
            if _compact(following.get("text")) in merge_tokens \
                    and float(following["bbox"][0]) - right <= gap_limit:
                right = float(following["bbox"][2])
        if not anchors or abs(anchors[-1] - right) >= 1:
            anchors.append(right)
    return anchors


def _statement_label(line: Dict, prefixes: List[str]) -> str:
    """Row name from the leading item box, without 一、/减：/（损失以…） decorations."""
    parts = sorted(line.get("parts") or [], key=lambda part: part["bbox"][0])
    if not parts:
        return ""
    text = normalize_text(parts[0].get("text") or "")
    if len(parts) > 1 and len(_compact(text)) <= 2:
        text = f"{text} {normalize_text(parts[1].get('text') or '')}"
    compact = _compact(text)
    for prefix in prefixes:
        token = _compact(prefix)
        if token and compact.startswith(token):
            compact = compact[len(token):]
            break
    compact = re.sub(r"^\d+[.．、]", "", compact)
    return re.split(r"[（(]", compact, 1)[0]


def _extract_statement_line(parsed: Dict, spec: Dict, config: Dict,
                            context: Dict) -> Optional[Dict]:
    """Fallback for text-layer statements without a header-anchored table.

    Annual-report statements have no 项目 header and no rules: the item column
    starts at the page margin, followed by 附注 and two right-aligned year columns.
    Only configured engines, exact row names and parts aligned to the year header
    are accepted; a line without an aligned amount stays missing instead of
    borrowing a neighbouring column.
    """
    table_spec = (((config.get("doc_types") or {}).get(parsed.get("doc_type"), {})
                   .get("tables") or {}).get(spec.get("table")) or {})
    options = table_spec.get("line_fallback") or {}
    columns = list(options.get("columns") or [])
    if not options.get("enabled") or spec.get("column") not in columns:
        return None
    engines = set(options.get("engines") or [])
    aliases = options.get("label_aliases") or {}
    labels = {_compact(spec["row_label"])}
    labels.update(_compact(item) for item in aliases.get(spec["row_label"], []))
    prefixes = list(options.get("label_prefixes") or [])
    stop_labels = [_compact(item) for item in options.get("stop_labels") or []]
    tolerance = float(options.get("column_tolerance", 20))
    structural_score = float(options.get("structural_score", 0.6))
    column_index = columns.index(spec["column"])
    extractor_id = f"table_row:{spec['table']}:{spec['row_label']}:{spec['column']}"
    number = re.compile(r"^[（(]?\s*-?[\d,]+(?:\.\d+)?\s*[）)]?$")

    for page in parsed.get("pages") or []:
        if engines and page.get("engine") not in engines:
            continue
        anchors: Optional[List[float]] = None
        header_bottom = 0.0
        for line in page.get("lines") or []:
            hits = _statement_header_anchors(line, options)
            if len(hits) >= len(columns):
                anchors = hits[:len(columns)]
                header_bottom = float(line["bbox"][3])
                break
        if anchors is None:
            continue
        for line_index, line in enumerate(page.get("lines") or []):
            if float(line["bbox"][1]) <= header_bottom:
                continue
            text = _compact(line.get("text"))
            if stop_labels and any(text.startswith(label) for label in stop_labels):
                break
            if _statement_label(line, prefixes) not in labels:
                continue
            target = anchors[column_index]
            for part in sorted(line.get("parts") or [], key=lambda item: item["bbox"][0]):
                raw = normalize_text(part.get("text") or "").strip()
                if not number.fullmatch(raw):
                    continue
                if abs(float(part["bbox"][2]) - target) > tolerance:
                    continue
                value, status, meta = _normalize(raw, spec, config, context)
                meta["line_fallback"] = True
                confidence = _base_confidence(part.get("confidence"), status, meta)
                grounding = _grounding(page, part["bbox"],
                                       f"p{page['page']}-line-{line_index}",
                                       [0, len(raw)])
                return _field(spec, value, status, raw, confidence, extractor_id,
                              grounding, meta, structural_score)
            return _field(spec, None, "missing", "", 0.0, extractor_id, None, {},
                          structural_score)
    return None


def _extract_table_row(parsed: Dict, spec: Dict, config: Dict, context: Dict) -> Dict:
    rows = _table_rows(parsed, spec["table"])
    row_label = _compact(spec["row_label"])
    for entry in rows:
        page, table, row = entry["page"], entry["table"], entry["row"]
        label_cell = row["cells"].get("项目")
        if not label_cell or _compact(label_cell["text"]) != row_label:
            continue
        cell = row["cells"].get(spec["column"])
        extractor_id = f"table_row:{spec['table']}:{spec['row_label']}:{spec['column']}"
        if not cell:
            return _field(spec, None, "missing", "", 0.0, extractor_id, None, {})
        value, status, meta = _normalize(cell["text"], spec, config, context)
        _mark_row_anomalies(meta, table, int(row.get("row_index", 0)))
        confidence = _base_confidence(cell.get("confidence"), status, meta)
        grounding = _grounding(page, cell.get("bbox"),
                               f"{table['table_id']}-r{row['row_index']}",
                               [0, len(cell["text"])])
        score = table_score(table, config["doc_types"][parsed["doc_type"]]["tables"][spec["table"]])
        return _field(spec, value, status, cell["text"], confidence,
                      extractor_id, grounding, meta, score)
    fallback = _extract_statement_line(parsed, spec, config, context)
    if fallback is not None:
        return fallback
    return _field(spec, None, "missing", "",
                  0.0, f"table_row:{spec['table']}:{spec['row_label']}", None, {})


def _qr_records(parsed: Dict, doc_type: str, config: Dict) -> List[Dict]:
    """Decoded QR payloads mapped through ``doc_types.<TYPE>.qr.fields``."""
    qr_cfg = (config.get("parsing") or {}).get("qr") or {}
    if not qr_cfg.get("enabled", True):
        return []
    doc_qr = ((config.get("doc_types") or {}).get(doc_type) or {}).get("qr") or {}
    fields = doc_qr.get("fields") or {}
    if not fields:
        return []
    delimiter = str(doc_qr.get("delimiter") or qr_cfg.get("delimiter", ","))
    records: List[Dict] = []
    for page in parsed.get("pages", []):
        for code in page.get("codes") or []:
            pieces = str(code.get("text") or "").split(delimiter)
            values: Dict[str, str] = {}
            for name, item in fields.items():
                index = int((item or {}).get("index", -1))
                if 0 <= index < len(pieces):
                    raw = pieces[index].strip()
                    if raw:
                        values[name] = raw
            if values:
                records.append({"page": page, "code": code, "values": values})
    return records


def _qr_crosscheck(fields: Dict[str, Dict], specs: List[Dict], parsed: Dict,
                   doc_type: str, config: Dict, context: Dict) -> None:
    """Fill empty QR-backed fields and flag disagreements, never overwrite OCR.

    The QR payload is printed by the issuer, so it is an independent source for
    the invoice number, date and total. A missing OCR value can be filled from
    it; an existing value that disagrees stays as extracted and is marked for
    review, because two originals disagreeing is exactly what a human must see.
    """
    records = _qr_records(parsed, doc_type, config)
    if not records:
        return
    for spec in specs:
        name = spec.get("qr_field")
        if not name or "[i]" in spec["path"]:
            continue
        target = fields.get(spec["path"])
        record = next((item for item in records if name in item["values"]), None)
        if target is None or record is None:
            continue
        raw = record["values"][name]
        value, status, meta = _normalize(raw, spec, config, context)
        if value in (None, "") or status != "ok":
            continue
        meta = dict(meta)
        meta["qr_field"] = name
        existing = target.get("value")
        if existing in (None, ""):
            page = record["page"]
            meta["source"] = "qr"
            meta["qr_payload"] = record["code"].get("text", "")
            target.update({
                "value": value,
                "raw_text": normalize_text(raw),
                "status": "ok",
                "confidence": 0.99,
                "extractor_id": f"{target.get('extractor_id', '')}:qr",
                "grounding": _grounding(page, record["code"].get("bbox"),
                                        f"p{page['page']}-qr", [0, len(raw)]),
                "meta": meta,
            })
            target.setdefault("signals", {}).update({
                "ocr_conf": 0.99,
                "rule_exact": 1.0,
                "format_valid": 1.0,
                "grounding_present": 1,
            })
            continue
        same = str(existing) == str(value)
        meta["qr_agreement"] = same
        if not same:
            meta["qr_conflict"] = {
                "ocr": existing,
                "qr": value,
                "payload": record["code"].get("text", ""),
            }
        target.setdefault("meta", {}).update(meta)
        target.setdefault("signals", {})["extractor_agreement"] = 1.0 if same else 0.0


def _crosscheck_values(primary, alternate, spec: Dict, options: Dict) -> Optional[Dict]:
    """两个通路取到的值是否对不上；数值按配置的绝对/相对容差判。"""
    if primary in (None, "") or alternate in (None, ""):
        return None
    if str(spec.get("type")) in ("decimal", "integer", "money"):
        try:
            left, right = Decimal(str(primary)), Decimal(str(alternate))
        except ArithmeticError:
            return None
        absolute = Decimal(str(options.get("tolerance_absolute", "0.01")))
        relative = Decimal(str(options.get("tolerance_relative", "0.0001")))
        if abs(left - right) <= max(absolute, abs(left) * relative):
            return None
        return {"alt_value": decimal_str(right), "difference": decimal_str(left - right)}
    if _compact(str(primary)) == _compact(str(alternate)):
        return None
    return {"alt_value": str(alternate)}


def _extract_line_cell(parsed: Dict, spec: Dict, config: Dict,
                       context: Dict) -> Optional[Dict]:
    """第二条定位通路：不看表格引擎的框格，只按正文行的数字顺序取列。

    表格引擎（pdfplumber 线框 / 版面推断）用二维框格决定某个片段属于哪一列；这里只看
    「行名片段之后按 x 排好的数字片段」，用列在表头里的序号选数。两套分配规则互不相干，
    所以串行、漏格、把邻列借过去的行会在这里对不上。两种情况一律拒绝作证，不猜：
    数字片段比数字列多（可能是附注列混进来，序号不可信），或序号超出片段数量
    （这行少印了一列，借隔壁的数就是伪造）。
    """
    label = _compact(spec.get("row_label"))
    column_name = _compact(spec.get("column"))
    if not label or not column_name:
        return None
    number = re.compile(r"^[（(]?-?[\d][\d, ]*(?:\.\d+)?[）)]?$")
    findings: List[Dict] = []
    for page in parsed.get("pages") or []:
        for table in page.get("tables") or []:
            if table.get("name") != spec.get("table"):
                continue
            table_bbox = table.get("bbox") or []
            headers = [_compact(item) for item in table.get("headers") or []]
            # 表头第 0 列是行名列，其余按顺序就是数字列；序号超出片段数量即不作证。
            if (column_name not in headers or len(table_bbox) < 4
                    or headers.index(column_name) < 1):
                continue
            rank = headers.index(column_name) - 1
            numeric_count = len(headers) - 1
            for line_index, line in enumerate(page.get("lines") or []):
                bbox = line.get("bbox") or []
                if len(bbox) < 4:
                    continue
                middle = (float(bbox[1]) + float(bbox[3])) / 2.0
                if not table_bbox[1] <= middle <= table_bbox[3]:
                    continue
                parts = sorted(line.get("parts") or [], key=lambda item: item["bbox"][0])
                hits = [index for index, part in enumerate(parts)
                        if _compact(part.get("text")) == label]
                if len(hits) != 1:
                    continue
                numbers = [part for part in parts[hits[0] + 1:]
                           if number.fullmatch(normalize_text(part.get("text") or "").strip())]
                if len(numbers) > numeric_count or rank >= len(numbers):
                    continue
                part = numbers[rank]
                findings.append({"page": page, "bbox": part["bbox"],
                                 "block_id": f"p{page['page']}-line-{line_index}",
                                 "raw": re.sub(r"(?<=\d)[\s\u00a0]+(?=\d)", "",
                                               normalize_text(part.get("text") or "").strip()),
                                 "confidence": float(part.get("confidence") or 0.0)})
    if not findings or len({_compact(item["raw"]) for item in findings}) != 1:
        return None
    found = findings[0]
    value, status, meta = _normalize(found["raw"], spec, config, context)
    meta["line_cell"] = True
    return _field(spec, value, status, found["raw"],
                  _base_confidence(found["confidence"], status, meta),
                  f"line_cell:{spec['table']}:{spec['row_label']}:{spec['column']}",
                  _grounding(found["page"], found["bbox"], found["block_id"],
                             [0, len(found["raw"])]), meta)


def _extractor_crosscheck(fields: Dict[str, Dict], specs: List[Dict], parsed: Dict,
                          config: Dict, context: Dict) -> None:
    """同一字段有两条互不依赖的通路时，两条都跑一遍再比对，对不上就交复核。

    SCENARIO 把 EXTRACTOR_CONFLICT 列为硬性原因，所以它必须真的能触发。可比通路：
    「表格引擎取单元格」↔「列 x 窗口在正文行里取数」（line_cell）/「年度列头右边缘取行」
    （statement_line），以及「标签取值」↔ 字段自己声明的 fallback。只记证据、不改值——
    两条谁对由人判断，程序没有资格在同一轮里既改值又宣布自己修好了。
    """
    options = (config.get("validation") or {}).get("extractor_crosscheck") or {}
    if not options.get("enabled"):
        return
    for spec in specs:
        if "[i]" in spec["path"]:
            continue
        field = fields.get(spec["path"])
        if not field or field.get("value") in (None, ""):
            continue
        meta = field.setdefault("meta", {})
        verdicts: Dict[str, str] = {}
        for pair in options.get("alternates") or []:
            if pair.get("primary") != spec.get("extractor"):
                continue
            alternate = str(pair.get("alternate") or "")
            if alternate == "line_cell":
                other = _extract_line_cell(parsed, spec, config, context)
            elif alternate == "statement_line":
                if meta.get("line_fallback"):
                    continue          # 值本来就是这条通路取的，自己不能给自己作证
                other = _extract_statement_line(parsed, spec, config, context)
            elif alternate == "declared_fallback":
                if meta.get("extractor_fallback") or meta.get("matrix_fallback"):
                    continue
                fallback = spec.get("fallback") or {}
                if not fallback:
                    continue
                merged = dict(spec)
                merged.pop("fallback", None)
                merged.update(fallback)
                other = _extract_scalar(parsed, merged, config, context)
            else:
                continue
            if not other or other.get("value") in (None, ""):
                continue
            evidence = _crosscheck_values(field.get("value"), other.get("value"),
                                          spec, options)
            verdicts[alternate] = "conflict" if evidence else "agreed"
            if evidence and not meta.get("extractor_conflict"):
                evidence["path"] = f"{spec.get('extractor')} vs {alternate}"
                # 证据要带备选通路读到的原文：数值 228.00 单独看像出错，
                # 配上片段「228,」才看得出是数字被截断了一组。
                evidence["alt_raw"] = other.get("raw_text")
                meta["extractor_conflict"] = evidence
        if not verdicts:
            continue
        field.setdefault("meta", {})["extractor_alternates"] = verdicts
        # signals.extractor_agreement 是 SCENARIO.md 里既有的「两通路是否一致」字段，
        # 二维码那条路已经会写它；取最小值，任何一条通路对不上都算对不上。
        signal = field.setdefault("signals", {})
        agreement = 0.0 if "conflict" in verdicts.values() else 1.0
        previous = signal.get("extractor_agreement")
        signal["extractor_agreement"] = agreement if previous is None else min(previous, agreement)


def _page_alphabet(page: Dict) -> str:
    """一页里解析器真正见过的所有字符，拼成一串（空白已去掉）。

    行文本、行内片段、单元格、二维码原文都要收进来：单元格由 pdfplumber 自己切，
    切法与 OCR 行不同，只比行文本会把正常读到的值误判成编造。
    """
    pieces: List[str] = []
    for line in page.get("lines") or []:
        pieces.append(str(line.get("text") or ""))
        for part in line.get("parts") or []:
            pieces.append(str(part.get("text") or ""))
    for table in page.get("tables") or []:
        pieces.extend(str(item) for item in table.get("headers") or [])
        for row in table.get("rows") or []:
            for cell in (row.get("cells") or {}).values():
                pieces.append(str(cell.get("text") or ""))
    for code in page.get("codes") or []:
        pieces.append(str(code.get("text") or ""))
    return "".join(_compact(item) for item in pieces)


def _raw_text_audit(fields: Dict[str, Dict], parsed: Dict, config: Dict) -> None:
    """回查字段声称读到的原文，是否真在这页出现过——编造率的唯一客观口径。

    标注集只能告诉你「抽错了几个」，要回答「有没有凭空造出内容」，不需要标注：
    凡是没有逐字出现在解析层已见过的字符里的原文，就不是读来的。判定故意宽松
    （去空白后跨片段连续即可、允许命中同文档其他页），宁可漏报也不误报，
    否则这个指标会变成复核台上的噪音。值一律不改，只记证据。
    """
    options = (config.get("validation") or {}).get("raw_text_audit") or {}
    if not options.get("enabled"):
        return
    pages = {int(page.get("page") or 0): _page_alphabet(page)
             for page in parsed.get("pages") or []}
    whole = "".join(pages[number] for number in sorted(pages))
    minimum = int(options.get("min_chars", 2))
    for field in fields.values():
        raw = str(field.get("raw_text") or "")
        key = _compact(raw)
        if len(key) < minimum:
            continue
        page = int((field.get("grounding") or {}).get("page") or 0)
        signal = field.setdefault("signals", {})
        meta = field.setdefault("meta", {})
        if key in pages.get(page, ""):
            meta["raw_grounded"] = "page"
            signal["raw_grounded"] = 1.0
        elif key in whole:
            # 跨页表格与表头续列会读到相邻页的字符：算有据，但记下来便于归因。
            meta["raw_grounded"] = "other_page"
            signal["raw_grounded"] = 1.0
        else:
            meta["raw_unverified"] = {"raw_text": raw, "page": page,
                                      "pages_checked": len(pages)}
            signal["raw_grounded"] = 0.0


def extract_document(parsed: Dict, config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    doc_type = parsed.get("doc_type", "UNKNOWN")
    specs = [dict(spec) for spec in field_specs(doc_type, config)]
    for spec in specs:
        if doc_type == "INCOME_STATEMENT" and spec.get("extractor") == "table_row":
            spec["unit_from_document"] = True
        if doc_type == "BANK_STATEMENT" and spec.get("path", "").endswith(".amount"):
            spec["signed_by_direction"] = True

    needs_unit = any(spec.get("unit_from_document")
                     or (spec.get("fallback") or {}).get("unit_from_document")
                     for spec in specs)
    unit, unit_factor, unit_text = _doc_unit(parsed, config) if needs_unit else (None, None, None)
    context = {"unit": unit, "unit_factor": unit_factor, "unit_text": unit_text}

    fields: Dict[str, Dict] = {}
    for spec in specs:
        extractor = spec.get("extractor")
        # 声明的名字优先：写了 table_cell 才走向量表，写了别的名字就按那个名字走
        # （认不出的名字在 _extract_scalar 里产出 unsupported、进复核）。只有完全
        # 没写 extractor 的数组路径才按形状兜底。声明与形状对不上由 doctypeaudit
        # 报 extractor_path_mismatch，运行时不悄悄改写配置的意思。
        if extractor == "table_cell" or (not extractor and "[i]" in spec["path"]):
            fields.update(_extract_table_cell(parsed, spec, config, context))
        elif extractor == "table_row":
            fields[spec["path"]] = _extract_table_row(parsed, spec, config, context)
        else:
            fields[spec["path"]] = _extract_scalar(parsed, spec, config, context)

    _qr_crosscheck(fields, specs, parsed, doc_type, config, context)
    _extractor_crosscheck(fields, specs, parsed, config, context)
    _raw_text_audit(fields, parsed, config)

    return {
        "document_id": parsed.get("document_id"),
        "source": parsed.get("source"),
        "doc_type": doc_type,
        "doc_type_confidence": parsed.get("doc_type_confidence"),
        "engine": parsed.get("engine"),
        "extractor_version": EXTRACTOR_VERSION,
        "currency_unit": unit,
        "fields": fields,
    }


def extract_folder(input_dir: Path, out_dir: Path, config: Optional[Dict] = None) -> List[Dict]:
    config = config or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(input_dir.glob("*.json")):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        result = extract_document(parsed, config)
        target = out_dir / f"{path.stem}.json"
        target.write_text(json.dumps(result, ensure_ascii=False,
                                     indent=int(config.get("output", {}).get("json_indent", 2))),
                          encoding="utf-8")
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract configured fields from parsed documents")
    parser.add_argument("--input", default=str(ROOT / "work" / "parsed"))
    parser.add_argument("--out", default=str(ROOT / "work" / "extracted"))
    args = parser.parse_args(argv)
    results = extract_folder(Path(args.input), Path(args.out))
    for result in results:
        fields = result["fields"]
        filled = sum(1 for item in fields.values() if item["value"] not in (None, ""))
        grounded = sum(1 for item in fields.values() if item["grounding"])
        print(f"{result['document_id']:<5} {result['doc_type']:<24} "
              f"fields={len(fields):<4} filled={filled:<4} grounded={grounded}")


if __name__ == "__main__":
    main()
