"""第二通路表格对账（SCENARIO §12.2）。

§12.2 要求文本层 PDF 上「pdfplumber vs PyMuPDF 表格」两个通路都跑一遍、把差异记下来，再定默认
组合。此前只有 pdfplumber 一条通路在跑：它把某张表的行或列切错了，下游只会看到一张「结构正常」
的错表——金额串列、掉一整行都能一路走到自动接受。这里把 PyMuPDF 的 `Page.find_tables()` 拉进来
做第二遍，只核对三件不需要解释的事：表框位置、总行数（含表头）、列数。

口径边界（都写进产物，也进文档）：
- 只比形状，不比格子文本。PyMuPDF 按字符位置拆 span，数字列取出来的串与 pdfplumber 不同
  （实测 D004 第 1 页首行是 `189 588 50\\n, .`，pdfplumber 是 `189,588.50`），拿它做文本对账
  只会产出噪声；「值是不是原文」由 `raw_audit` 那条腿对着文字层核。
- 只跑文本层页。扫描页的第二通路是 OCR，不是表格解析，这里如实记一条跳过理由，不冒充跑过。
- 行数按「pdfplumber 数据行 + 表头行」对「PyMuPDF row_count」：两家都把表头算成一行。
- 只读 `work/parsed` 的解析缓存与 `samples/` 的原件，不重跑解析、不写 `out/`，所以这份数字
  与交付物字节无关，也不可能靠重解析一次变好看。
- 合成集上两条通路形状完全一致（dev 8 张、holdout 7 张表），这只证明管线对得上，不证明真实件
  也对得上——真实件到位前不据此改默认组合。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, load_config

SECOND_ENGINE = "pymupdf"
MODULE_VERSION = "0.1.0"
DEFAULTS = {"bbox_min_iou": 0.98}

SKIP_NO_CACHE = "无解析缓存"
SKIP_PAGE_NOT_TEXT_LAYER = "该页不是文字层（第二通路不适用）"
SKIP_SECOND_PATH_ERROR = "第二通路在该页报错"
DECLARED_SKIPS = frozenset({SKIP_NO_CACHE, SKIP_PAGE_NOT_TEXT_LAYER, SKIP_SECOND_PATH_ERROR})

PROBLEM_ROW_COUNT = "row_count_diff"
PROBLEM_COL_COUNT = "col_count_diff"
PROBLEM_BBOX = "bbox_off"
PROBLEM_NOT_FOUND = "not_found"
PROBLEM_SECOND_PATH = "second_path_error"
DECLARED_PROBLEMS = frozenset({PROBLEM_ROW_COUNT, PROBLEM_COL_COUNT, PROBLEM_BBOX,
                               PROBLEM_NOT_FOUND, PROBLEM_SECOND_PATH})


def bbox_iou(first: Optional[List[float]], second: Optional[List[float]]) -> float:
    if not first or not second or len(first) < 4 or len(second) < 4:
        return 0.0
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = (float(first[2]) - float(first[0])) * (float(first[3]) - float(first[1]))
    other = (float(second[2]) - float(second[0])) * (float(second[3]) - float(second[1]))
    union = area + other - inter
    return inter / union if union > 0 else 0.0


def expected_rows(table: Dict) -> int:
    """pdfplumber 的 `rows` 不含表头，PyMuPDF 的 `row_count` 含；表头计入总数再比。"""
    rows = len(table.get("rows") or [])
    return rows + (1 if table.get("headers") else 0)


def first_columns(table: Dict) -> int:
    columns = table.get("columns") or []
    if columns:
        return len(columns)
    rows = table.get("rows") or []
    if not rows:
        return 0
    return max((len(row.get("cells") or {}) for row in rows), default=0)


def compare_table(table: Dict, second: Optional[Dict],
                  min_iou: float = DEFAULTS["bbox_min_iou"]) -> Dict:
    """把一张 pdfplumber 表与它在同一页上最近的一张 PyMuPDF 表对账。"""
    detail = {
        "table_id": table.get("table_id"),
        "name": table.get("name"),
        "page": table.get("page"),
        "rows_first": expected_rows(table),
        "cols_first": first_columns(table),
        "rows_second": (second or {}).get("rows"),
        "cols_second": (second or {}).get("cols"),
        "bbox_iou": round(bbox_iou(table.get("bbox"), (second or {}).get("bbox")), 4),
    }
    problems: List[str] = []
    if second is None:
        problems.append(PROBLEM_NOT_FOUND)
    else:
        if detail["rows_first"] != detail["rows_second"]:
            problems.append(PROBLEM_ROW_COUNT)
        if detail["cols_first"] != detail["cols_second"]:
            problems.append(PROBLEM_COL_COUNT)
        if detail["bbox_iou"] < min_iou:
            problems.append(PROBLEM_BBOX)
    detail["problems"] = problems
    detail["verdict"] = "disagree" if problems else "agree"
    detail["min_iou"] = min_iou
    return detail


def _second_tables(page) -> List[Dict]:
    found = page.find_tables()
    return [{"bbox": [round(float(value), 2) for value in item.bbox],
             "rows": int(item.row_count),
             "cols": int(item.col_count)}
            for item in getattr(found, "tables", []) or []]


def measure_document(pdf_path: Path, parsed: Dict,
                     min_iou: float = DEFAULTS["bbox_min_iou"], opener=None) -> Dict:
    """两份通路对一份文档的每张表逐一取账；不开 pdf 也能测（注入 `opener`）。"""
    if opener is None:
        import pymupdf as fitz
        opener = fitz.open

    entry = {
        "document_id": parsed.get("document_id"),
        "engine": parsed.get("engine"),
        "pages_scanned": 0,
        "tables": [],
        "skipped": {},
    }
    skipped: Dict[str, int] = entry["skipped"]
    try:
        document = opener(str(pdf_path))
    except Exception:
        skipped[SKIP_SECOND_PATH_ERROR] = skipped.get(SKIP_SECOND_PATH_ERROR, 0) + 1
        return entry

    try:
        for page in parsed.get("pages") or []:
            if page.get("engine") != "pymupdf_text":
                skipped[SKIP_PAGE_NOT_TEXT_LAYER] = skipped.get(SKIP_PAGE_NOT_TEXT_LAYER, 0) + 1
                continue
            tables = page.get("tables") or []
            if not tables:
                continue
            try:
                seconds = _second_tables(document[int(page.get("page") or 1) - 1])
            except Exception:
                for table in tables:
                    entry["tables"].append({
                        "table_id": table.get("table_id"), "name": table.get("name"),
                        "page": page.get("page"), "problems": [PROBLEM_SECOND_PATH],
                        "verdict": "disagree", "min_iou": min_iou})
                continue
            entry["pages_scanned"] += 1
            for table in tables:
                best = max(seconds, key=lambda item: bbox_iou(table.get("bbox"), item["bbox"]),
                           default=None)
                entry["tables"].append(compare_table(table, best, min_iou))
    finally:
        try:
            document.close()
        except Exception:
            pass
    return entry


def measure_split(samples_dir: Path, parsed_dir: Path, manifest_path: Path,
                  split: str, min_iou: float = DEFAULTS["bbox_min_iou"],
                  opener=None) -> Dict:
    """按 manifest 挑该分片的文字层文档，逐份对账；扫描件不进分母。"""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    documents = [item for item in manifest.get("documents") or []
                 if item.get("split") == split and item.get("format") == "text"]

    payload = {
        "version": MODULE_VERSION,
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "second_engine": SECOND_ENGINE,
        "bbox_min_iou": min_iou,
        "documents": [],
        "coverage": {"text_layer_docs": len(documents), "documents_scanned": 0,
                     "documents_with_tables": 0},
        "tables": 0,
        "verdicts": {"agree": 0, "disagree": 0},
        "problems": {},
        "skipped": {},
    }
    skipped = payload["skipped"]
    for item in documents:
        document_id = item.get("document_id")
        cache = Path(parsed_dir) / f"{document_id}.json"
        if not cache.exists():
            skipped[SKIP_NO_CACHE] = skipped.get(SKIP_NO_CACHE, 0) + 1
            continue
        parsed = json.loads(cache.read_text(encoding="utf-8"))
        if parsed.get("engine") != "pymupdf_text":
            skipped[SKIP_PAGE_NOT_TEXT_LAYER] = skipped.get(SKIP_PAGE_NOT_TEXT_LAYER, 0) + 1
            continue
        source = Path(samples_dir) / Path(str(parsed.get("source") or "")).name
        entry = measure_document(source, parsed, min_iou, opener=opener)
        for reason, count in (entry.get("skipped") or {}).items():
            skipped[reason] = skipped.get(reason, 0) + count
        payload["documents"].append(entry)
        payload["coverage"]["documents_scanned"] += 1
        if entry["tables"]:
            payload["coverage"]["documents_with_tables"] += 1
        for table in entry["tables"]:
            payload["tables"] += 1
            payload["verdicts"][table["verdict"]] += 1
            for problem in table.get("problems") or []:
                payload["problems"][problem] = payload["problems"].get(problem, 0) + 1
    payload["undeclared_skips"] = sorted(set(skipped) - DECLARED_SKIPS)
    payload["undeclared_problems"] = sorted(set(payload["problems"]) - DECLARED_PROBLEMS)
    return payload


def threshold_from_config(config: Dict) -> float:
    section = (config.get("parsing") or {}).get("table_crosspath") or {}
    return float(section.get("bbox_min_iou", 0.98))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="第二通路表格对账（诊断用，不写 out/）")
    parser.add_argument("--samples", default=None, help="默认取 samples/<split>")
    parser.add_argument("--parsed", default=str(ROOT / "work" / "parsed"))
    parser.add_argument("--manifest", default=str(ROOT / "samples" / "manifest.json"))
    parser.add_argument("--split", default=None, help="分片名，默认取 --samples 目录名")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    split = args.split or (Path(args.samples).name if args.samples else "dev")
    samples_dir = Path(args.samples) if args.samples else ROOT / "samples" / split
    config = load_config()
    payload = measure_split(samples_dir, Path(args.parsed), Path(args.manifest),
                            split, threshold_from_config(config))
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / f"table_crosspath_{split}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"分片={split} 文字层文档={payload['coverage']['text_layer_docs']} "
          f"已扫={payload['coverage']['documents_scanned']} 表={payload['tables']} "
          f"一致={payload['verdicts']['agree']} 不一致={payload['verdicts']['disagree']} "
          f"跳过={payload['skipped']} 问题={payload['problems']}")
    print("report:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
