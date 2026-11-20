"""标注契约审计：SCENARIO §9.2 的标注格式必须先被评估器自己读得动。

§9.2 规定每份标注是 `{document_id, doc_type, fields{path: {value, page, bbox}}}`，
且 `bbox` 只框住值、不框整行。评估器把它们当真：`evaluate.grounding_ok` 直接取
`gt["bbox"]`，`page` 对不上就判未溯源。所以标注少一个 `bbox` 是当场 KeyError，
少一个 `page` 是**静默**把所有字段记成未溯源——指标照常算完，只是分母被悄悄污染。

`corpusaudit` 只数 `bs.transactions[i].*` 的行号，不看这些键；`samplefreeze` 只核对
样本文与标注文件的存在与 sha256。缺的是对标注**内部**形状的核对，判据全读出厂配置：

- `label_missing` / `label_unreadable`：标注文件读不出来；
- `document_id_mismatch` / `doc_type_missing` / `doc_type_unknown` / `doc_type_mismatch`：
  标注自称的身份与清单对不上；
- `fields_missing` / `field_value_missing` / `field_path_unknown`：字段表本身的问题；
- `page_count_missing` / `field_page_missing` / `field_page_out_of_range`：页码无从校验或越界；
- `field_bbox_missing` / `field_bbox_shape` / `field_bbox_degenerate` / `field_bbox_outside`：
  直接复用 `parse.grounding_problem` 的四个理由码，容差读 `parsing.grounding.tolerance_pt`。

越界核对要页宽高，取自 `work/parsed/<document_id>.json`；解析缓存缺失时跳过越界项并记
`bounds_checked`，不装作核对过。本模块只读清单、标注与解析缓存，只诊断，不写任何文件。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import load_config, resolve_path
from .parse import grounding_problem, grounding_tolerance

REASONS = (
    "label_missing", "label_unreadable",
    "document_id_mismatch", "doc_type_missing", "doc_type_unknown", "doc_type_mismatch",
    "fields_missing", "field_value_missing", "field_path_unknown",
    "page_count_missing", "field_page_missing", "field_page_out_of_range",
    "field_bbox_missing", "field_bbox_shape", "field_bbox_degenerate", "field_bbox_outside",
)

DYNAMIC_INDEX = re.compile(r"\[\d+\]")
BBOX_REASONS = ("bbox_missing", "bbox_shape", "bbox_degenerate", "bbox_outside")


def schema_paths(doc_type: str, config: Dict) -> Set[str]:
    """该文档类型在 §8 schema 里登记过的全部路径，含表格列（`bs.transactions[i].*`）。"""
    spec = (config.get("doc_types") or {}).get(doc_type) or {}
    paths = {str(field["path"]) for field in spec.get("fields") or []}
    for table in (spec.get("tables") or {}).values():
        for column in table.get("columns") or []:
            paths.add(str(column["path"] if isinstance(column, dict) else column))
    return paths


def normalize_path(path: str) -> str:
    return DYNAMIC_INDEX.sub("[i]", str(path))


def label_of(doc: Dict, config: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """标注文件本体与读不出来的理由；清单里的 `label_path` 是唯一入口。"""
    value = doc.get("label_path")
    if not value:
        return None, "label_missing"
    path = resolve_path(str(value), config)
    if not path.exists():
        return None, "label_missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError):
        return None, "label_unreadable"


def page_sizes(document_id: Optional[str], config: Dict) -> Dict[int, Tuple[float, float]]:
    """解析缓存里的页宽高；缓存缺失返回空表，越界核对随之跳过。"""
    if not document_id:
        return {}
    path = resolve_path(config["paths"]["work"], config) / "parsed" / f"{document_id}.json"
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sizes: Dict[int, Tuple[float, float]] = {}
    for page in parsed.get("pages") or []:
        number = page.get("page")
        width = page.get("width")
        height = page.get("height")
        if (isinstance(number, int) and isinstance(width, (int, float))
                and isinstance(height, (int, float))):
            sizes[number] = (float(width), float(height))
    return sizes


def audit_label(doc: Dict, label: Optional[Dict], config: Dict,
                sizes: Optional[Dict[int, Tuple[float, float]]] = None,
                error: Optional[str] = None) -> Dict:
    """按 §9.2 核对一份标注；问题逐条带理由码、样本号与字段路径。"""
    document_id = doc.get("document_id")
    problems: List[Dict] = []

    def note(code: str, detail: str) -> None:
        assert code in REASONS, code
        problems.append({"code": code, "document_id": document_id, "detail": detail})

    shape = {"document_id": document_id, "doc_type": doc.get("doc_type"),
             "fields": 0, "bounds_checked": 0, "problems": problems}
    if error or label is None:
        note(error or "label_missing", f"标注读不出来：{doc.get('label_path')}")
        return shape

    if label.get("document_id") != document_id:
        note("document_id_mismatch", f"{label.get('document_id')} != {document_id}")
    declared = label.get("doc_type")
    known_types = config.get("doc_types") or {}
    if not declared:
        note("doc_type_missing", "标注没有 doc_type")
    elif str(declared) not in known_types:
        note("doc_type_unknown", f"{declared} 不在 §8 schema 里")
    elif declared != doc.get("doc_type"):
        note("doc_type_mismatch", f"{declared} != {doc.get('doc_type')}")

    pages = doc.get("page_count")
    if not isinstance(pages, int) or isinstance(pages, bool) or pages < 1:
        note("page_count_missing", f"清单里的 page_count 是 {pages!r}")
        pages = None

    fields = label.get("fields")
    if not isinstance(fields, dict) or not fields:
        note("fields_missing", "标注没有 fields，或字段表为空")
        return shape
    shape["fields"] = len(fields)

    known = schema_paths(str(declared), config) if str(declared) in known_types else set()
    tolerance = grounding_tolerance(config)
    sizes = sizes or {}
    for path, item in fields.items():
        item = item if isinstance(item, dict) else {}
        where = f"{path}"
        if "value" not in item or item.get("value") is None:
            note("field_value_missing", f"{where} 没有 value")
        if known and normalize_path(path) not in known:
            note("field_path_unknown", f"{where} 不在 {declared} 的 schema 里")
        page = item.get("page")
        if not isinstance(page, int) or isinstance(page, bool):
            note("field_page_missing", f"{where} 的 page 是 {page!r}")
            width, height = 0.0, 0.0
        else:
            if page < 1 or (pages is not None and page > pages):
                note("field_page_out_of_range", f"{where} 的 page={page} 不在 [1, {pages}]")
            width, height = sizes.get(page, (0.0, 0.0))
        reason = grounding_problem(width, height, item.get("bbox"), tolerance)
        if reason in BBOX_REASONS:
            note(f"field_{reason}", f"{where} 的 bbox={item.get('bbox')!r}")
        elif width > 0 and height > 0:
            shape["bounds_checked"] += 1
    return shape


def split_documents(split: str, config: Dict,
                    documents: Optional[List[Dict]] = None) -> List[Dict]:
    if documents is None:
        from .corpusaudit import manifest_documents

        documents = manifest_documents(config)
    return [doc for doc in documents if doc.get("split") == split]


def audit_split(split: str, config: Optional[Dict] = None,
                documents: Optional[List[Dict]] = None) -> Dict:
    """逐份核对一个 split 的全部标注，返回字段总数与边界核对数。"""
    config = config or load_config()
    docs = split_documents(split, config, documents)
    shapes = []
    for doc in docs:
        label, error = label_of(doc, config)
        shape = audit_label(doc, label, config, page_sizes(doc.get("document_id"), config), error)
        shapes.append(shape)
    return {
        "split": split,
        "documents": len(docs),
        "fields": sum(shape["fields"] for shape in shapes),
        "bounds_checked": sum(shape["bounds_checked"] for shape in shapes),
        "labels": shapes,
        "problems": [problem for shape in shapes for problem in shape["problems"]],
    }


def audit(config: Optional[Dict] = None, splits: Optional[List[str]] = None,
          documents: Optional[List[Dict]] = None) -> Dict:
    """逐 split 审计；`splits` 缺省取清单里出现过的分片（不靠目录名猜）。"""
    from .corpusaudit import manifest_documents

    config = config or load_config()
    documents = manifest_documents(config) if documents is None else documents
    if splits is None:
        splits = sorted({str(doc.get("split")) for doc in documents if doc.get("split")})
    shapes = {split: audit_split(split, config, documents) for split in splits}
    total = sum(len(shape["problems"]) for shape in shapes.values())
    return {"splits": shapes, "problems": total}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SCENARIO §9.2 标注契约审计")
    parser.add_argument("--split", action="append",
                        help="只审计这些 split（可重复；默认清单里出现过的全部）")
    args = parser.parse_args(argv)
    config = load_config()
    result = audit(config, splits=args.split)
    for name, shape in result["splits"].items():
        print(f"[{name}] 文档 {shape['documents']} 份 · 字段 {shape['fields']} 个"
              f" · 越界核对 {shape['bounds_checked']} 个")
        for problem in shape["problems"]:
            who = f" {problem['document_id']}" if problem["document_id"] else ""
            print(f"       问题 {problem['code']}{who}: {problem['detail']}")
    print(f"标注契约问题: {result['problems']}")
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
