"""fields.csv 契约审计：SCENARIO §8.4 说「列固定为」11 列，写出来是 14 列。

§8.4 把交付表格写死了：`document_id, doc_type, field_path, value, value_type, confidence,
decision, page, bbox, extractor_id, review_reasons`，每字段一行、数组字段按行展开、
UTF-8 with BOM（Excel 直接打开）。实现侧后来又追加了三列（`review_actions` 记复核动作、
`value_rewrite` 记值改写、`chain_evidence` 记流水链证据）——这三列是真实的事后补录，
但不是契约列。此前没有任何机械装置盯着这件事：

- 列顺序被谁调一下、某个列名拼错，下游按位置读的脚本会静默读错；
- BOM 掉了，`utf-8-sig` 读得动，Excel 里中文变乱码，验收也不红；
- 漏掉一行或重复一行，`fields.csv` 的格数与单文档 JSON 就对不上。

这里把「契约 11 列」与「已登记的三列扩展」分开写在配置里（`output.csv_columns` /
`output.csv_appendix`），生产侧照配置写，审计侧核对四件事：配置前 11 列是否逐字等于 §8.4、
实际表头是否等于配置拼接、编码是否带 BOM、行是否逐字段覆盖且取值合法。本模块只读产物，
只诊断，不写任何文件。
"""
from __future__ import annotations

import argparse
import codecs
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ROOT, load_config
from .textutil import csv_cell

SPEC_LINE = re.compile(r"列固定为：\s*\n+\s*`([^`\n]+)`")
DECISIONS = ("accept", "review")
SKIP_DIRS = {"review", "final", "exports", "report", "log"}


def section84_columns(text: str) -> List[str]:
    """从 SCENARIO §8.4 的 fenced 行里读出契约列；读不出来返回空表，不猜。"""
    match = SPEC_LINE.search(text)
    if not match:
        return []
    return [column.strip() for column in match.group(1).split(",") if column.strip()]


def expected_columns(config: Dict) -> List[str]:
    output = config.get("output") or {}
    return list(output.get("csv_columns") or []) + list(output.get("csv_appendix") or [])


def is_bom_encoding(config: Dict) -> bool:
    encoding = str((config.get("output") or {}).get("csv_encoding") or "").lower()
    return encoding.replace("_", "-") == "utf-8-sig"


def header_problems(header: List[str], columns: List[str]) -> List[str]:
    """表头与配置拼接的差异，逐种分别命名：少列、多列、顺序/改名都落到 drift。"""
    if header == columns:
        return []
    problems = []
    missing = [column for column in columns if column not in header]
    extra = [column for column in header if column not in columns]
    if missing:
        problems.append("header_missing")
    if extra:
        problems.append("header_extra")
    if not missing and not extra:
        problems.append("header_order")
    return problems


def field_documents(out_dir: Path) -> Dict[str, List[str]]:
    """产物里每份文档的字段路径表；只认 `<out>/<doc>/<doc>.json` 这种目录形状。"""
    documents: Dict[str, List[str]] = {}
    if not out_dir.exists():
        return documents
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        path = entry / f"{entry.name}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        documents[entry.name] = list((payload.get("fields") or {}).keys())
    return documents


def row_problems(row: Dict[str, str]) -> List[Tuple[str, str]]:
    """单行列值：decision 只认两个值，value_type 不许空，bbox 给了就得是四个有限数。"""
    problems = []
    where = f"{row.get('document_id')}:{row.get('field_path')}"
    if row.get("decision") not in DECISIONS:
        problems.append(("decision_invalid", f"{where} decision={row.get('decision')!r}"))
    if not (row.get("value_type") or "").strip():
        problems.append(("value_type_missing", f"{where} 没有 value_type"))
    bbox = (row.get("bbox") or "").strip()
    if bbox:
        try:
            values = json.loads(bbox)
        except ValueError:
            values = None
        if (not isinstance(values, list) or len(values) != 4
                or not all(isinstance(value, (int, float)) for value in values)):
            problems.append(("bbox_shape", f"{where} bbox={bbox}"))
    return problems


def audit(out_dir: Path, config: Optional[Dict] = None,
          scenario_text: Optional[str] = None,
          csv_bytes: Optional[bytes] = None) -> Dict:
    """逐项核对 `fields.csv` 与 §8.4；问题带理由码与位置。

    `csv_bytes` 供验收在内存里喂一份篡改过的表头/行，不必落临时目录。
    """
    config = config or load_config()
    text = scenario_text if scenario_text is not None else \
        (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    spec = section84_columns(text)
    columns = expected_columns(config)
    problems: List[Dict] = []

    def note(code: str, detail: str) -> None:
        problems.append({"code": code, "detail": detail})

    if not spec:
        note("scenario_columns_unread", "SCENARIO §8.4 的固定列读不出来")
    if not columns:
        note("config_columns_missing", "output.csv_columns / csv_appendix 都没配")
    elif spec and columns[:len(spec)] != spec:
        note("config_columns_drift", f"配置前 {len(spec)} 列 {columns[:len(spec)]} != {spec}")

    path = out_dir / "fields.csv"
    if csv_bytes is None and not path.exists():
        note("csv_missing", f"{path} 不存在")
        return {"columns": columns, "rows": 0, "documents": 0, "problems": problems}

    raw = csv_bytes if csv_bytes is not None else path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8) != is_bom_encoding(config):
        note("encoding_mismatch",
             f"BOM={raw.startswith(codecs.BOM_UTF8)} 配置要求 csv_encoding="
             f"{(config.get('output') or {}).get('csv_encoding')!r}")
    body = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(body))
    header = list(reader.fieldnames or [])
    for code in header_problems(header, columns):
        note(code, f"表头={header} 期望={columns}")
    rows = list(reader)

    documents = {csv_cell(doc_id): fields for doc_id, fields in field_documents(out_dir).items()}
    expected = {(doc_id, field) for doc_id, fields in documents.items() for field in fields}
    seen: Dict[Tuple[str, str], int] = {}
    for row in rows:
        key = (row.get("document_id") or "", row.get("field_path") or "")
        if key[0] not in documents:
            note("row_unknown_document", f"{key[0]} 不在产物目录里")
        elif key[1] not in documents[key[0]]:
            note("row_unknown_field", f"{key[0]}:{key[1]} 不在单文档 JSON 里")
        if key in seen:
            seen[key] += 1
            note("row_duplicate", f"{key[0]}:{key[1]} 出现 {seen[key]} 次")
        else:
            seen[key] = 1
        for code, detail in row_problems(row):
            note(code, detail)
    missing = sorted(expected - set(seen))
    for doc_id, field in missing[:20]:
        note("row_missing", f"{doc_id}:{field} 没有行")
    if len(missing) > 20:
        note("row_missing", f"另有 {len(missing) - 20} 格没有行")

    return {"columns": header, "rows": len(rows), "documents": len(documents),
            "fields": len(expected), "problems": problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SCENARIO §8.4 fields.csv 契约审计")
    parser.add_argument("--out", required=True, help="产物目录（含 fields.csv 与单文档 JSON）")
    args = parser.parse_args(argv)
    config = load_config()
    result = audit(Path(args.out), config)
    print(f"表头 {len(result['columns'])} 列 · 行 {result['rows']} 条"
          f" · 文档 {result['documents']} 份 / 字段 {result.get('fields', 0)} 个")
    for problem in result["problems"]:
        print(f"       问题 {problem['code']}: {problem['detail']}")
    print(f"CSV 契约问题: {len(result['problems'])}")
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
