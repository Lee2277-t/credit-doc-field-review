"""表格结构独立度量（G4 / P1-6）。

要量的是交付物里没人单独看过的一件事：**格子取对没有**。值读对不代表格子取对——取到相邻列的
金额一样能过格式校验，而余额串到交易金额位上正是复核台上最常见的一类错。
对标做法来自 docTR 的 `TableCellMetric`（先配对、再要求逻辑坐标 (行, 列) 精确相等才算结构对）
和 PaddleOCR PP-Structure 的 `structure_score`（结构分与识别分分开报）。

一处有实测依据的改写：docTR 用 IoU 阈值配对，本项目改成「嵌套度」配对（交集占任一格面积的
最大值）。原因是各家引擎交的框语义不一样——同一格，pdfplumber 交整列带、segment_table 交贴着
字的框，IoU 只有 0.2 上下。按 IoU 配会在 dev 上少配 455/1344 格，把「找到了、只是框更紧」
误判成漏检；嵌套度对框的松紧不敏感，对「跑到别的行/列」仍然敏感。IoU 没有丢，它按引擎单独报，
当作框的贴合度看，不进结构判定。

口径边界（都写进产物，不藏在代码里）：
- 真值的行号只来自标注路径里的数组下标（`bs.transactions[3].amount` → 第 3 行）。按行标签
  取数的字段（利润表 `is.*`）真值里没有行号，只计数、不参与结构分。
- 检出侧只保留「标注里出现过的列」，否则 `项目` 这种行标签列会整列算成多检。
- 配对限制在同一张表、同一页内：跨页配上的高重叠是坐标巧合，不是命中。
- `structure_exact_rate` 的分母是「找到的格」（docTR 同口径）：掉一整行落在 `missed_cell`
  与 `cell_found_rate` 上，结构分照样可以是 1.0。看网格掉没掉要一起看这两个数。
- 只读 `work/parsed` 的解析缓存与 `labels/`，不重跑解析、不写 `out/`，因此这份数字
  不可能被「重新解析一次」改变，也和交付物的字节无关。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ROOT, field_spec, load_config
from .evaluate import _iou

ARRAY_INDEX_RE = re.compile(r"\[(\d+)\]")
SCHEMA_ARRAY_RE = re.compile(r"\[\d+\]")
METRIC_VERSION = "0.1.0"

# 每条标注字段落到结构度的哪个桶，缺哪个就说明清楚，不做静默丢弃。
SKIP_NO_TABLE = "非表格字段"
SKIP_TABLE_NOT_IN_SCHEMA = "表未在 schema"
SKIP_COLUMN_NOT_IN_HEADERS = "声明列不在该表表头"
SKIP_NO_ROW_INDEX = "行标签锚定（真值没有行号）"
SKIP_NO_BBOX = "标注缺坐标"

# 检出侧的丢弃理由同样必须有名有姓，验收会拿这张清单去查有没有静默丢弃。
SKIP_DET_NO_ROW_INDEX = "检出行缺 row_index"
SKIP_DET_HEADER_UNKNOWN = "格子的表头不在列清单"
SKIP_DET_HEADER_NOT_IN_SCHEMA = "表头不在 schema 列清单（版式变体）"
SKIP_DET_COLUMN_UNANNOTATED = "该列无带行号的标注真值（不计入多检）"
SKIP_DET_NO_BBOX = "检出格缺坐标"

DECLARED_SKIP_REASONS = frozenset({
    SKIP_NO_TABLE, SKIP_TABLE_NOT_IN_SCHEMA, SKIP_COLUMN_NOT_IN_HEADERS,
    SKIP_NO_ROW_INDEX, SKIP_NO_BBOX,
    SKIP_DET_NO_ROW_INDEX, SKIP_DET_HEADER_UNKNOWN, SKIP_DET_HEADER_NOT_IN_SCHEMA,
    SKIP_DET_COLUMN_UNANNOTATED, SKIP_DET_NO_BBOX,
})


def _params(config: Dict) -> Dict:
    params = ((config.get("evaluation") or {}).get("table_structure") or {})
    return {
        "cover_threshold": float(params.get("cover_threshold", 0.5)),
        "strict_iou": float(params.get("strict_iou", 0.9)),
        "max_examples": int(params.get("max_examples", 20)),
    }


def canonical_column_order(doc_type: str, config: Dict) -> Dict[str, List[str]]:
    """表名的列顺序以 schema 为准——两侧都用它，才量得出「解析把表头读串了」。"""
    tables = (config["doc_types"].get(doc_type) or {}).get("tables") or {}
    return {name: list(spec.get("headers") or []) for name, spec in tables.items()}


def ground_truth_cells(labels: Dict, doc_type: str, config: Dict) -> Tuple[List[Dict], Counter]:
    """标注侧单元格：逻辑坐标 (行, 列) 来自数组下标与 schema 表头顺序。"""
    cells: List[Dict] = []
    skipped: Counter = Counter()
    tables = (config["doc_types"].get(doc_type) or {}).get("tables") or {}
    for path, item in (labels.get("fields") or {}).items():
        spec = field_spec(doc_type, SCHEMA_ARRAY_RE.sub("[i]", path), config)
        if not spec or not spec.get("table") or not spec.get("column"):
            skipped[SKIP_NO_TABLE] += 1
            continue
        tspec = tables.get(spec["table"])
        if not tspec:
            skipped[SKIP_TABLE_NOT_IN_SCHEMA] += 1
            continue
        headers = list(tspec.get("headers") or [])
        if spec["column"] not in headers:
            skipped[SKIP_COLUMN_NOT_IN_HEADERS] += 1
            continue
        match = ARRAY_INDEX_RE.search(path)
        bbox = item.get("bbox")
        if not match:
            skipped[SKIP_NO_ROW_INDEX] += 1
            continue
        if not bbox:
            skipped[SKIP_NO_BBOX] += 1
            continue
        cells.append({
            "table": spec["table"],
            "row": int(match.group(1)),
            "col": headers.index(spec["column"]),
            "page": item.get("page"),
            "bbox": [float(value) for value in bbox],
            "path": path,
        })
    return cells, skipped


def detected_cells(parsed: Dict, annotated_columns: Optional[Dict[str, set]],
                   canonical_headers: Optional[Dict[str, List[str]]] = None
                   ) -> Tuple[List[Dict], Dict[str, List[Dict]], Counter]:
    """检出侧单元格：同一张表的续表按 (页, table_id) 排序后累加行号，得到全文档行号。

    annotated_columns 里没有列可保留的表（如利润表 is.* 这种按行标签取数的表）整表不参与配对，
    只把网格实例记下来；这样一份没有行号真值的文档不会冒出一堆「多检」。

    列号一律按 **schema 的表头顺序**（canonical_headers）数，不用解析记录里那一份 headers：
    用后者等于让解析器自己给自己编列号，表头行读串了也照样「对得上」，而那一类错位正是
    余额串进交易金额位的原因。检出表头里不在 schema 清单上的（如借贷双列版式的 借方发生额）
    记进 skipped，不冒充多检。
    """
    cells: List[Dict] = []
    grids: Dict[str, List[Dict]] = defaultdict(list)
    skipped: Counter = Counter()
    annotated_columns = annotated_columns or {}
    canonical_headers = canonical_headers or {}
    instances: Dict[str, List[Tuple[int, str, Dict]]] = defaultdict(list)
    for page in parsed.get("pages") or []:
        for table in page.get("tables") or []:
            instances[table.get("name", "")].append(
                (page.get("page", 0), table.get("table_id", ""), table))
    for name, items in instances.items():
        keep = set(annotated_columns.get(name) or ())
        canonical = list(canonical_headers.get(name) or [])
        offset = 0
        for page_no, table_id, table in sorted(items, key=lambda row: (row[0], row[1])):
            headers = list(table.get("headers") or [])
            rows = table.get("rows") or []
            row_indices = [row.get("row_index") for row in rows
                           if isinstance(row.get("row_index"), int)]
            counted = 0
            for row in sorted(rows, key=lambda r: (r.get("row_index") if isinstance(r.get("row_index"), int) else 0)):
                index = row.get("row_index")
                if not isinstance(index, int):
                    skipped[SKIP_DET_NO_ROW_INDEX] += 1
                    continue
                for header, cell in (row.get("cells") or {}).items():
                    if header not in headers:
                        skipped[SKIP_DET_HEADER_UNKNOWN] += 1
                        continue
                    if header not in canonical:
                        skipped[SKIP_DET_HEADER_NOT_IN_SCHEMA] += 1
                        continue
                    col = canonical.index(header)
                    if col not in keep:
                        skipped[SKIP_DET_COLUMN_UNANNOTATED] += 1
                        continue
                    bbox = (cell or {}).get("bbox")
                    if not bbox:
                        skipped[SKIP_DET_NO_BBOX] += 1
                        continue
                    cells.append({
                        "table": name,
                        "row": offset + index,
                        "col": col,
                        "page": page_no,
                        "table_id": table_id,
                        "local_row": index,
                        "engine": table.get("engine"),
                        "bbox": [float(value) for value in bbox],
                    })
                counted += 1
            grids[name].append({
                "table_id": table_id,
                "page": page_no,
                "engine": table.get("engine"),
                "rows_detected": counted,
                "max_row_index": max(row_indices, default=None),
                "row_offset": offset,
                "columns": len(headers),
                "header_order_matches_schema": headers == canonical,
                "continued_from": table.get("continued_from"),
            })
            offset += counted
    return cells, grids, skipped


def _nesting(a: List[float], b: List[float]) -> float:
    """一个框被另一个框「装进去」的程度：max(交集/A 面积, 交集/B 面积)。

    不用 IoU 判结构对不对，是因为各家引擎交的框语义不一样（实测）：pdfplumber 交整列带、
    segment_table 交贴着字的框。用 IoU 会把「格子找到了、只是框更紧」误判成漏检——
    dev 上这么算会少配 455/1344 格。嵌套度对框的松紧不敏感，对「跑到别的列」仍然敏感。
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return max(inter / area_a, inter / area_b)


def score_document(doc_id: str, doc_type: str, labels: Dict, parsed: Dict,
                   config: Dict, params: Dict,
                   degradation: Optional[str] = None) -> Dict:
    gt_cells, skipped = ground_truth_cells(labels, doc_type, config)
    annotated_columns: Dict[str, set] = defaultdict(set)
    for cell in gt_cells:
        annotated_columns[cell["table"]].add(cell["col"])
    det_cells, grids, det_skipped = detected_cells(parsed, dict(annotated_columns),
                                                   canonical_column_order(doc_type, config))
    tables: Dict[str, Dict] = {}

    def bucket(name: str) -> Dict:
        if name not in tables:
            tables[name] = {
                "table": name,
                "gt_cells": 0, "det_cells": 0,
                "aligned": 0, "misassigned": 0, "missed_cell": 0, "shared_det": 0,
                "extra_det": 0,
                "align_row_off": 0, "align_col_off": 0, "align_both_off": 0,
                "iou_sum": 0.0, "strict_iou_hits": 0, "iou_pairs": 0,
                "gt_rows": 0, "det_rows": 0,
                "by_engine": {},
                "examples": [],
            }
        return tables[name]

    det_index: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    logical_index: Dict[Tuple[str, int, int, int], Dict] = {}
    for cell in det_cells:
        bucket(cell["table"])["det_cells"] += 1
        det_index[(cell["table"], cell["page"])].append(cell)
        logical_index[(cell["table"], cell["page"], cell["row"], cell["col"])] = cell

    claimed: Dict[Tuple[str, int, int], int] = Counter()
    cover_threshold = params["cover_threshold"]
    for gt_cell in gt_cells:
        name, page = gt_cell["table"], gt_cell["page"]
        rec = bucket(name)
        rec["gt_cells"] += 1
        candidates = [(_nesting(gt_cell["bbox"], cell["bbox"]), cell)
                      for cell in det_index.get((name, page), [])]
        candidates = [row for row in candidates if row[0] >= cover_threshold]
        partner = None
        if candidates:
            partner = max(candidates, key=lambda row: (
                row[0], (row[1]["row"], row[1]["col"]) == (gt_cell["row"], gt_cell["col"])))[1]
        row_off = col_off = False
        if partner is None:
            rec["missed_cell"] += 1
        else:
            row_off = partner["row"] != gt_cell["row"]
            col_off = partner["col"] != gt_cell["col"]
            key = (partner["table_id"], partner["row"], partner["col"])
            claimed[key] += 1
            if claimed[key] > 1:
                rec["shared_det"] += 1
            if row_off and col_off:
                rec["align_both_off"] += 1
                rec["misassigned"] += 1
            elif row_off:
                rec["align_row_off"] += 1
                rec["misassigned"] += 1
            elif col_off:
                rec["align_col_off"] += 1
                rec["misassigned"] += 1
            else:
                rec["aligned"] += 1
            overlap = _iou(gt_cell["bbox"], partner["bbox"])
            rec["iou_sum"] += overlap
            rec["iou_pairs"] += 1
            rec["strict_iou_hits"] += int(overlap > params["strict_iou"])
            engine_stats = rec["by_engine"].setdefault(
                partner["engine"] or "unknown",
                {"cells": 0, "aligned": 0, "misassigned": 0, "iou_sum": 0.0, "strict_iou_hits": 0})
            engine_stats["cells"] += 1
            engine_stats["aligned"] += int(not row_off and not col_off)
            engine_stats["misassigned"] += int(row_off or col_off)
            engine_stats["iou_sum"] += overlap
            engine_stats["strict_iou_hits"] += int(overlap > params["strict_iou"])
            if (row_off or col_off) and len(rec["examples"]) < params["max_examples"]:
                rec["examples"].append({
                    "page": page,
                    "path": gt_cell["path"],
                    "gt_row": gt_cell["row"], "gt_col": gt_cell["col"],
                    "det_row": partner["row"], "det_col": partner["col"],
                    "det_local_row": partner["local_row"],
                    "det_table_id": partner["table_id"],
                    "engine": partner["engine"],
                    "nesting": round(_nesting(gt_cell["bbox"], partner["bbox"]), 4),
                    "iou": round(overlap, 4),
                    "gt_bbox": gt_cell["bbox"], "det_bbox": partner["bbox"],
                })
    for cell in det_cells:
        if not claimed.get((cell["table_id"], cell["row"], cell["col"])):
            bucket(cell["table"])["extra_det"] += 1

    for name, rec in tables.items():
        found = rec["aligned"] + rec["misassigned"]
        rec["gt_rows"] = len({cell["row"] for cell in gt_cells if cell["table"] == name})
        rec["det_rows"] = len({(cell["table_id"], cell["row"])
                               for cell in det_cells if cell["table"] == name})
        rec["row_count_delta"] = rec["det_rows"] - rec["gt_rows"]
        rec["cell_found_rate"] = _rate(found, rec["gt_cells"])
        rec["structure_exact_rate"] = _rate(rec["aligned"], found)
        rec["misassignment_rate"] = _rate(rec["misassigned"], found)
        rec["missed_cell_rate"] = _rate(rec["missed_cell"], rec["gt_cells"])
        rec["phantom_cell_rate"] = _rate(rec.get("extra_det", 0), rec["det_cells"])
        rec["mean_iou"] = round(rec["iou_sum"] / rec["iou_pairs"], 4) if rec["iou_pairs"] else None
        rec["strict_iou_rate"] = _rate(rec["strict_iou_hits"], rec["iou_pairs"])
        for engine_stats in rec["by_engine"].values():
            engine_stats["mean_iou"] = (round(engine_stats["iou_sum"] / engine_stats["cells"], 4)
                                        if engine_stats["cells"] else None)
            engine_stats["strict_iou_rate"] = _rate(engine_stats["strict_iou_hits"],
                                                    engine_stats["cells"])
            engine_stats["structure_exact_rate"] = _rate(engine_stats["aligned"],
                                                         engine_stats["cells"])
            del engine_stats["iou_sum"]
            del engine_stats["strict_iou_hits"]
        rec["grid_instances"] = grids.get(name, [])

    return {
        "document_id": doc_id,
        "doc_type": doc_type,
        "degradation": degradation,
        "tables": tables,
        "skipped": dict(skipped),
        "detected_skipped": dict(det_skipped),
        "grid_instances": grids,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def _totals(rows: List[Dict]) -> Dict:
    totals = {"gt_cells": 0, "det_cells": 0, "aligned": 0, "misassigned": 0,
              "missed_cell": 0, "shared_det": 0, "extra_det": 0,
              "strict_iou_hits": 0, "iou_pairs": 0, "iou_sum": 0.0}
    for row in rows:
        for key in totals:
            totals[key] += row.get(key, 0)
    found = totals["aligned"] + totals["misassigned"]
    return {
        **{key: value for key, value in totals.items() if key != "iou_sum"},
        "iou_sum": round(totals["iou_sum"], 4),
        "cell_found_rate": _rate(found, totals["gt_cells"]),
        "structure_exact_rate": _rate(totals["aligned"], found),
        "misassignment_rate": _rate(totals["misassigned"], found),
        "missed_cell_rate": _rate(totals["missed_cell"], totals["gt_cells"]),
        "phantom_cell_rate": _rate(totals["extra_det"], totals["det_cells"]),
        "mean_iou": (round(totals["iou_sum"] / totals["iou_pairs"], 4)
                     if totals["iou_pairs"] else None),
        "strict_iou_rate": _rate(totals["strict_iou_hits"], totals["iou_pairs"]),
    }


def _grouped(documents: List[Dict], key: str) -> List[Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for document in documents:
        label = document.get(key)
        if label is None:
            continue
        for rec in document["tables"].values():
            buckets[label].append(rec)
    rows = []
    for label, recs in sorted(buckets.items()):
        doc_ids = {document["document_id"] for document in documents if document.get(key) == label}
        rows.append({"name": label, "documents": len(doc_ids), **_totals(recs)})
    return rows


def load_degradation_map(manifest_path: Optional[Path]) -> Dict[str, str]:
    if not manifest_path or not Path(manifest_path).exists():
        return {}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {item["document_id"]: item.get("degradation")
            for item in manifest.get("documents") or []}


def score_split(labels_dir: Path, parsed_dir: Path, config: Dict,
                manifest_path: Optional[Path] = None) -> Dict:
    params = _params(config)
    degradations = load_degradation_map(manifest_path)
    documents: List[Dict] = []
    missing_cache: List[str] = []
    for label_path in sorted(Path(labels_dir).glob("*.json")):
        doc_id = label_path.stem
        parsed_path = Path(parsed_dir) / f"{doc_id}.json"
        if not parsed_path.exists():
            missing_cache.append(doc_id)
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        documents.append(score_document(doc_id, labels.get("doc_type", ""), labels, parsed,
                                       config, params, degradations.get(doc_id)))
    all_recs = [rec for doc in documents for rec in doc["tables"].values()]
    return {
        "metric_version": METRIC_VERSION,
        "params": params,
        "labels_dir": str(labels_dir),
        "parsed_dir": str(parsed_dir),
        "note": ("仅诊断：不进自动接受、不参与信任决策，改这里的阈值不会放行或拦下任何字段。"
                 "结构分按「嵌套度」配对后看逻辑坐标是否相等；真值行号来自标注路径的数组下标，"
                 "按行标签取数的字段不计结构分、只计数。"),
        "documents_scored": len(documents),
        "cache_missing": missing_cache,
        "overall": _totals(all_recs),
        "by_table": _grouped_by_table(documents),
        "by_doc_type": _grouped(documents, "doc_type"),
        "by_degradation": _grouped(documents, "degradation"),
        "documents": documents,
    }


def _grouped_by_table(documents: List[Dict]) -> List[Dict]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for document in documents:
        for name, rec in document["tables"].items():
            buckets[name].append(rec)
    return [{"name": name,
             "documents": len({d["document_id"] for d in documents if name in d["tables"]}),
             **_totals(recs)}
            for name, recs in sorted(buckets.items())]


def _print(payload: Dict) -> None:
    def number(value) -> str:
        return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))

    print(f"{'table':<16}{'docs':>5}{'gt':>7}{'det':>7}{'found':>7}{'align':>7}{'misass':>7}"
          f"{'missed':>7}{'struct':>8}{'misrate':>9}{'meanIoU':>9}{'strict':>8}")
    for row in payload["by_table"]:
        print(f"{row['name']:<16}{row['documents']:>5}{row['gt_cells']:>7}{row['det_cells']:>7}"
              f"{row['aligned'] + row['misassigned']:>7}{row['aligned']:>7}"
              f"{row['misassigned']:>7}{row['missed_cell']:>7}"
              f"{number(row['structure_exact_rate']):>8}{number(row['misassignment_rate']):>9}"
              f"{number(row['mean_iou']):>9}{number(row['strict_iou_rate']):>8}")
    overall = payload["overall"]
    print(f"{'overall':<16}{payload['documents_scored']:>5}{overall['gt_cells']:>7}"
          f"{overall['det_cells']:>7}{overall['aligned'] + overall['misassigned']:>7}"
          f"{overall['aligned']:>7}{overall['misassigned']:>7}{overall['missed_cell']:>7}"
          f"{number(overall['structure_exact_rate']):>8}{number(overall['misassignment_rate']):>9}"
          f"{number(overall['mean_iou']):>9}{number(overall['strict_iou_rate']):>8}")
    for row in payload["by_degradation"]:
        print(f"  退化 {row['name']:<14} gt={row['gt_cells']} align={row['aligned']} "
              f"misassign={row['misassigned']} missed={row['missed_cell']} "
              f"struct={number(row['structure_exact_rate'])} "
              f"合并框={row['shared_det']} 多检格={row['extra_det']}")
    if payload["cache_missing"]:
        print(f"解析缓存缺失: {len(payload['cache_missing'])} 份 {payload['cache_missing'][:5]}")
    print(payload["note"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="表格结构独立度量（诊断用）")
    parser.add_argument("--labels", default=str(ROOT / "labels" / "dev"))
    parser.add_argument("--parsed", default=str(ROOT / "work" / "parsed"))
    parser.add_argument("--manifest", default=None, help="难例集 manifest，用来按退化归因")
    parser.add_argument("--split", default=None, help="产物文件名用的分片名，默认取 labels 目录名")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    config = load_config()
    labels_dir = Path(args.labels)
    if not labels_dir.is_absolute():
        labels_dir = ROOT / labels_dir
    parsed_dir = Path(args.parsed)
    if not parsed_dir.is_absolute():
        parsed_dir = ROOT / parsed_dir
    manifest = Path(args.manifest) if args.manifest else None
    payload = score_split(labels_dir, parsed_dir, config, manifest)
    _print(payload)
    split = args.split or labels_dir.name
    out_path = Path(args.json_path) if args.json_path else ROOT / "work" / "eval" / f"table_structure_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report:", out_path)


if __name__ == "__main__":
    main()
