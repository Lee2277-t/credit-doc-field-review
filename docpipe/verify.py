"""Phase 1 acceptance: compare parsed cells/lines with ground-truth bboxes.

Usage:
  python -m docpipe.verify --split dev
  python -m docpipe.verify --split dev --render D004 D005 D008
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz

from .config import ROOT, field_specs
from .textutil import normalize_text, parse_date, parse_decimal

TABLE_COLUMNS = {
    "BANK_STATEMENT": {"transactions": ["txn_date", "summary", "counterparty_name",
                                        "counterparty_account", "direction", "amount",
                                        "balance", "txn_id"]},
    "INCOME_STATEMENT": {"income": ["revenue", "cost_of_sales", "taxes_surcharges",
                                    "selling_expenses", "admin_expenses", "rd_expenses",
                                    "finance_expenses", "other_income", "investment_income",
                                    "operating_profit", "non_operating_income",
                                    "non_operating_expense", "total_profit", "income_tax",
                                    "net_profit"]},
}


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def center_inside(inner: List[float], outer: List[float]) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _norm_value(text: str) -> str:
    return re.sub(r"[,，\s]+", "", normalize_text(text))


DIRECTION_LABELS = {"income": "收入", "expense": "支出", "credit": "收入", "debit": "支出"}
FLAG_LABELS = {"yes": "是", "no": "否", "unknown": "未知"}


def _same_value(kind: str, gt: str, parsed: str, factor: int = 1) -> bool:
    if kind in {"amount", "balance", "monetary"}:
        left, right = parse_decimal(gt), parse_decimal(parsed)
        if right is not None:
            right *= factor
        if left is None or right is None:
            return False
        # 万元 display rounds to 0.01, i.e. 50 yuan
        return abs(left - right) <= (factor * 0.005 + 0.01)
    if kind in {"txn_date", "date"}:
        left, right = parse_date(gt), parse_date(parsed)
        return left is not None and left == right
    if kind == "direction":
        return DIRECTION_LABELS.get(_norm_value(gt).lower(), _norm_value(gt)) == _norm_value(parsed)
    if kind == "flag":
        return FLAG_LABELS.get(_norm_value(gt).lower(), _norm_value(gt)) == _norm_value(parsed)
    if kind == "currency" and _norm_value(parsed) in {"人民币", "CNY", "人民币元"}:
        return _norm_value(gt) in {"CNY", "RMB", "人民币"}
    return _norm_value(gt) == _norm_value(parsed)


def _gt_rows(labels: Dict) -> List[Dict]:
    fields = labels["fields"]
    rows: Dict[int, Dict] = {}
    for path, item in fields.items():
        match = re.match(r"\w+\.(\w+)\[(\d+)\]\.(\w+)$", path)
        if not match:
            continue
        rows.setdefault(int(match.group(2)), {})[match.group(3)] = item
    return [rows[i] for i in sorted(rows)]


def _parsed_rows(parsed: Dict, table_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for page in parsed["pages"]:
        for table in page["tables"]:
            if table["name"] != table_name:
                continue
            for row in table["rows"]:
                rows.append({"page": page["page"], "cells": row["cells"]})
    return rows


def verify_document(doc_id: str, split: str) -> Dict:
    labels = json.loads((ROOT / "labels" / split / f"{doc_id}.json").read_text(encoding="utf-8"))
    parsed = json.loads((ROOT / "work" / "parsed" / f"{doc_id}.json").read_text(encoding="utf-8"))
    doc_type = labels["doc_type"]
    result = {"document_id": doc_id, "doc_type": doc_type, "engine": parsed["engine"]}

    table_name = "transactions" if doc_type == "BANK_STATEMENT" else (
        "income" if doc_type == "INCOME_STATEMENT" else None)
    if table_name:
        parsed_rows = _parsed_rows(parsed, table_name)
        pairs: List[Tuple[Dict, Dict, str, int, int]] = []
        factor = 1
        if doc_type == "BANK_STATEMENT":
            columns = {
                "交易日期": "txn_date", "摘要": "summary", "对方户名": "counterparty_name",
                "对方账号": "counterparty_account", "收支方向": "direction",
                "交易金额": "amount", "余额": "balance", "交易流水号": "txn_id",
            }
            gt_rows = _gt_rows(labels)
            for index, gt_row in enumerate(gt_rows[:len(parsed_rows)]):
                cells = parsed_rows[index]["cells"]
                page_no = parsed_rows[index]["page"]
                for header, key in columns.items():
                    if gt_row.get(key) and cells.get(header):
                        pairs.append((gt_row[key], cells[header], key, factor, page_no))
            gt_count = len(gt_rows)
        else:
            unit_value = (labels["fields"].get("is.currency_unit") or {}).get("value", "")
            factor = 10000 if ("万" in unit_value or "WAN" in unit_value) else 1
            gt_rows = []
            row_labels = set()
            for spec in field_specs("INCOME_STATEMENT"):
                gt_item = labels["fields"].get(spec.get("path"))
                if spec.get("extractor") != "table_row" or not gt_item:
                    continue
                gt_rows.append(gt_item)
                row_labels.add(spec["row_label"])
                row = next((item for item in parsed_rows
                            if _norm_value(item["cells"].get("项目", {}).get("text", ""))
                            == _norm_value(spec["row_label"])), None)
                cell = row["cells"].get(spec["column"]) if row else None
                if cell:
                    pairs.append((gt_item, cell, "monetary", factor, row["page"]))
            gt_count = len(row_labels)
        cell_stats = []
        value_hits = 0
        for gt_item, cell, kind, pair_factor, page_no in pairs:
            same_page = page_no == gt_item["page"]
            if _same_value(kind, gt_item["value"], cell["text"], factor=pair_factor):
                value_hits += 1
            cell_stats.append({
                "iou": iou(gt_item["bbox"], cell["bbox"]) if same_page else 0.0,
                "center": center_inside(gt_item["bbox"], cell["bbox"]) if same_page else False,
                "parsed_center": center_inside(cell["bbox"], gt_item["bbox"]) if same_page else False,
                "overlap": iou(gt_item["bbox"], cell["bbox"]) > 0 if same_page else False,
            })
        ious = sorted(item["iou"] for item in cell_stats)
        result["table"] = {
            "gt_rows": gt_count,
            "parsed_rows": len(parsed_rows),
            "row_match": gt_count == len(parsed_rows),
            "cells": len(cell_stats),
            "value_match_rate": round(value_hits / len(pairs), 4) if pairs else None,
            "bbox_iou_median": round(ious[len(ious) // 2], 3) if ious else None,
            "bbox_iou_ge_0_5": round(sum(1 for v in ious if v >= 0.5) / len(ious), 4) if ious else None,
            "gt_center_in_cell": round(sum(1 for item in cell_stats if item["center"]) / len(cell_stats), 4)
            if cell_stats else None,
            "parsed_center_in_gt": round(sum(1 for item in cell_stats if item["parsed_center"]) / len(cell_stats), 4)
            if cell_stats else None,
            "overlap_rate": round(sum(1 for item in cell_stats if item["overlap"]) / len(cell_stats), 4)
            if cell_stats else None,
        }

    # label fields: value text must appear in a parsed line whose bbox covers the GT center
    line_stats = []
    for path, item in labels["fields"].items():
        if re.search(r"\[\d+\]", path):
            continue
        page_no = item["page"]
        page = parsed["pages"][page_no - 1]
        matched = None
        for line in page["lines"]:
            if _norm_value(item["value"]) and _norm_value(item["value"]) in _norm_value(line["text"]):
                if matched is None or iou(item["bbox"], line["bbox"]) > matched["iou"]:
                    matched = {"iou": iou(item["bbox"], line["bbox"]),
                               "center": center_inside(item["bbox"], line["bbox"])}
        if matched is None:
            covering = next((line for line in page["lines"]
                             if center_inside(item["bbox"], line["bbox"])), None)
            if covering:
                matched = {"iou": iou(item["bbox"], covering["bbox"]), "center": True}
        line_stats.append(matched or {"iou": 0.0, "center": False})
    result["fields"] = {
        "total": len(line_stats),
        "found_in_line": sum(1 for item in line_stats if item["iou"] > 0),
        "gt_center_in_line": round(sum(1 for item in line_stats if item["center"]) / len(line_stats), 4)
        if line_stats else None,
        "bbox_iou_median": round(sorted(item["iou"] for item in line_stats)[len(line_stats) // 2], 3)
        if line_stats else None,
    }
    return result


def render_debug(doc_id: str, split: str, out_dir: Path, dpi: int = 130) -> List[Path]:
    source = ROOT / "samples" / split / f"{doc_id}.scan.pdf"
    if not source.exists():
        source = ROOT / "samples" / split / f"{doc_id}.pdf"
    parsed = json.loads((ROOT / "work" / "parsed" / f"{doc_id}.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    doc = fitz.open(source)
    for page_data in parsed["pages"]:
        page = doc[page_data["page"] - 1]
        for line in page_data["lines"]:
            page.draw_rect(fitz.Rect(line["bbox"]), color=(0, 0.65, 0), width=0.6)
        for table in page_data["tables"]:
            for row in table["rows"]:
                for cell in row["cells"].values():
                    page.draw_rect(fitz.Rect(cell["bbox"]), color=(0.1, 0.35, 0.95), width=0.5)
        target = out_dir / f"{doc_id}_p{page_data['page']}.png"
        page.get_pixmap(dpi=dpi).save(target)
        outputs.append(target)
    doc.close()
    return outputs


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify parsed coordinates against labels")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--render", nargs="*", default=[])
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args(argv)
    label_dir = ROOT / "labels" / args.split
    results = []
    for label_path in sorted(label_dir.glob("*.json")):
        result = verify_document(label_path.stem, args.split)
        results.append(result)
        table = result.get("table")
        extra = ""
        if table:
            extra = (f" rows={table['parsed_rows']}/{table['gt_rows']}"
                     f" values={fmt(table['value_match_rate'])}"
                     f" iou_med={fmt(table['bbox_iou_median'])}"
                     f" iou>=.5={fmt(table['bbox_iou_ge_0_5'])}"
                     f" gt_center={fmt(table['gt_center_in_cell'])}"
                     f" p_center={fmt(table['parsed_center_in_gt'])}"
                     f" overlap={fmt(table['overlap_rate'])}")
        fields = result["fields"]
        print(f"{result['document_id']:<5} {result['doc_type']:<24} {result['engine']:<12}"
              f" labels={fields['found_in_line']}/{fields['total']}"
              f" center={fmt(fields['gt_center_in_line'])}"
              f" iou_med={fields['bbox_iou_median']}{extra}")
    out_path = ROOT / "work" / "verify" / f"{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for doc_id in args.render:
        outputs = render_debug(doc_id, args.split, ROOT / "work" / "debug", args.dpi)
        print("rendered:", ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
