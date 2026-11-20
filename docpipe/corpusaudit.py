"""语料形状审计：SCENARIO §9.1 声明的合成语料形状必须等于磁盘上的语料。

§9.1 把语料写死了：每个 split 10 份、四类按 3/3/2/2 分布、每个 split 5 份文本层 + 5 份
扫描件、每类文档在每个 split 至少 1 份扫描件、每份流水 45～60 笔交易且 3～4 页、每个 split
至少 120 笔交易。这份形状是**所有指标的共同分母**：少两份、整个 split 全用文本层、流水砍到
20 笔，指标会在更小的分母上照常算完——验收里只有「交付了多少格」的下限
（`validation.delivery.min_delivered_fields`），管不到语料设计本身；`gate.small_group_not_claimed`
管的是「别宣称达标」，也不管分母是不是本来就是薄的那十来份。

这里拿 `samples/manifest.json`（每份样本的 `split` / `doc_type` / `format` / `page_count`）
与标注里的 `bs.transactions[i].*` 行号重新数一遍，判据全读 `evaluation.corpus`：

- `document_count` / `doc_type_count` / `undeclared_doc_type`：语料清单的形状；
- `format_count` / `scan_coverage`：文本层与扫描件的混合比例；
- `statement_rows` / `statement_pages` / `transaction_floor`：金额字段的样本量；
- `label_missing`：标注读不出来，行数无从数起——不装作 0 笔。

标注在这里只数行号、不读值；本模块只读清单与标注、只诊断，不参与判决，也不写任何文件。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config, resolve_path

DEFAULTS = {
    "documents_per_split": 10,
    "per_doc_type": {"BUSINESS_LICENSE": 3, "BANK_STATEMENT": 3,
                     "INCOME_STATEMENT": 2, "CREDIT_REPORT_ENTERPRISE": 2},
    "formats": {"text": 5, "scan": 5},
    "min_scans_per_doc_type": 1,
    "statement_rows": {"min": 45, "max": 60},
    "statement_pages": {"min": 3, "max": 4},
    "min_transactions_per_split": 120,
}

TXN_PATH = re.compile(r"^bs\.transactions\[(\d+)\]\.\w")


def options(config: Dict) -> Dict:
    """`evaluation.corpus` 整段；缺键时退回与出厂配置逐字相同的默认值。"""
    section = ((config.get("evaluation") or {}).get("corpus")) or {}
    merged = dict(DEFAULTS)
    merged.update({key: value for key, value in section.items() if value is not None})
    return merged


def manifest_documents(config: Dict) -> List[Dict]:
    """顶层 `samples/manifest.json` 的全部样本条目（与 `docpipe.synth` 写的是同一份）。"""
    from .samplefreeze import load_manifest

    path = resolve_path(config["paths"]["samples"], config) / "manifest.json"
    if not path.exists():
        return []
    return load_manifest(path).get("documents") or []


def split_documents(split: str, config: Dict,
                    documents: Optional[List[Dict]] = None) -> List[Dict]:
    if documents is None:
        from .samplefreeze import load_manifest, manifest_path

        documents = load_manifest(manifest_path(split, config)).get("documents") or []
    return [doc for doc in documents if doc.get("split") == split]


def label_transactions(doc: Dict) -> Optional[int]:
    """标注里出现过的 `bs.transactions[i].*` 行号个数；标注读不出来返回 None。"""
    from .config import ROOT

    value = doc.get("label_path")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    indexes = set()
    for key in (labels.get("fields") or {}):
        match = TXN_PATH.match(key)
        if match:
            indexes.add(int(match.group(1)))
    return len(indexes)


def audit_split(split: str, config: Optional[Dict] = None,
                documents: Optional[List[Dict]] = None) -> Dict:
    """按 `evaluation.corpus` 逐条核对一个 split 的语料形状，问题逐条带 code 与样本号。"""
    config = config or load_config()
    opts = options(config)
    docs = split_documents(split, config, documents)
    problems: List[Dict] = []

    def note(code: str, detail: str, document_id: Optional[str] = None) -> None:
        problems.append({"code": code, "document_id": document_id, "detail": detail})

    if len(docs) != int(opts["documents_per_split"]):
        note("document_count", f"{len(docs)} != {opts['documents_per_split']}")

    types = Counter(str(doc.get("doc_type")) for doc in docs)
    declared = {str(name): int(want) for name, want in (opts["per_doc_type"] or {}).items()}
    for name, want in declared.items():
        if types.get(name, 0) != want:
            note("doc_type_count", f"{name}: {types.get(name, 0)} != {want}")
    for name in sorted(types):
        if name not in declared:
            note("undeclared_doc_type", f"{name}: {types[name]}")

    formats = Counter(str(doc.get("format")) for doc in docs)
    for name, want in (opts["formats"] or {}).items():
        if formats.get(name, 0) != int(want):
            note("format_count", f"{name}: {formats.get(name, 0)} != {want}")

    scans = Counter(str(doc.get("doc_type")) for doc in docs if doc.get("format") == "scan")
    floor_scans = int(opts["min_scans_per_doc_type"])
    for name in declared:
        if scans.get(name, 0) < floor_scans:
            note("scan_coverage", f"{name}: {scans.get(name, 0)} < {floor_scans}")

    row_bounds = opts["statement_rows"] or {}
    page_bounds = opts["statement_pages"] or {}
    statements = []
    transactions = 0
    for doc in docs:
        if doc.get("doc_type") != "BANK_STATEMENT":
            continue
        document_id = doc.get("document_id")
        rows = label_transactions(doc)
        if rows is None:
            note("label_missing", "标注读不出来，流水笔数无从数起", document_id)
        else:
            transactions += rows
            statements.append({"document_id": document_id, "rows": rows,
                               "pages": doc.get("page_count")})
            low, high = int(row_bounds.get("min")), int(row_bounds.get("max"))
            if not low <= rows <= high:
                note("statement_rows", f"{rows} 笔不在 [{low}, {high}]", document_id)
        pages = doc.get("page_count")
        low, high = int(page_bounds.get("min")), int(page_bounds.get("max"))
        if not isinstance(pages, int) or not low <= pages <= high:
            note("statement_pages", f"{pages} 页不在 [{low}, {high}]", document_id)

    floor = int(opts["min_transactions_per_split"])
    if transactions < floor:
        note("transaction_floor", f"{transactions} < {floor}")

    return {
        "split": split,
        "documents": len(docs),
        "by_doc_type": dict(sorted(types.items())),
        "by_format": dict(sorted(formats.items())),
        "scans_by_type": dict(sorted(scans.items())),
        "statements": statements,
        "transactions": transactions,
        "problems": problems,
    }


def audit(config: Optional[Dict] = None, splits: Optional[List[str]] = None,
          documents: Optional[List[Dict]] = None) -> Dict:
    """逐 split 审计；`splits` 缺省取清单里出现过的分片（不靠目录名猜，`hard/` 不算语料）。"""
    config = config or load_config()
    documents = manifest_documents(config) if documents is None else documents
    if splits is None:
        splits = sorted({str(doc.get("split")) for doc in documents if doc.get("split")})
    shapes = {split: audit_split(split, config, documents) for split in splits}
    total = sum(len(shape["problems"]) for shape in shapes.values())
    return {"splits": shapes, "problems": total}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SCENARIO §9.1 语料形状审计")
    parser.add_argument("--split", action="append",
                        help="只审计这些 split（可重复；默认清单里出现过的全部）")
    args = parser.parse_args(argv)
    config = load_config()
    result = audit(config, splits=args.split)
    for name, shape in result["splits"].items():
        rows = " · ".join(f"{item['document_id']} {item['rows']} 笔/{item['pages']} 页"
                          for item in shape["statements"])
        print(f"[{name}] 文档 {shape['documents']} 份 {shape['by_doc_type']}")
        print(f"       格式 {shape['by_format']} 扫描覆盖 {shape['scans_by_type']}")
        print(f"       流水 {rows or '无'} · 合计 {shape['transactions']} 笔")
        for problem in shape["problems"]:
            who = f" {problem['document_id']}" if problem["document_id"] else ""
            print(f"       问题 {problem['code']}{who}: {problem['detail']}")
    print(f"语料形状问题: {result['problems']}")
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
