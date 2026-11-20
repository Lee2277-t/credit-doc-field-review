"""SCENARIO §12.1「统一解析输出」的契约审计。

§12.1 用一段 JSON 画出了解析层的对外接口。它从来没被机械核对过：没人比对「缓存里
真有这些键吗」，也没人比对「文档与实现差在哪、差的地方有没有人认领」。2026-10-29
实测三处偏离：

    pages[].blocks[]（block_id / type / lines）  → pages[].lines[]（行级，无块分组）
    tables[].cells[]（cell_id / row / col）      → tables[].rows[].cells{表头: {…}}
    engine {name, version, text_layer, ocr}      → 文档级与页面级各一个引擎名字符串

三件事各自能单独失败：

1. 契约键：§12.1 里出现的每个叶子路径，要么在缓存里真的存在，要么在 `DEVIATIONS`
   里被认领。偏差清单**双向可证伪**——`deviation_stale` 是「实现追上了文档，这条
   偏差该删」，`deviation_broken` 是「偏差声称的替代形状一次都没出现，清单在说谎」。
2. 语义不变量：文档给的是形状，这里查形状背后的自洽性（页码连续、bbox 在页内、
   行号连续、格键属于声明列……）。每个问题一个独立码，互不连坐。
3. 覆盖下限：`min_documents` / `min_pages` / `min_lines` 防「0 违规其实是 0 文件」。

口径：只读解析缓存与 `SCENARIO.md`，不读 `out/`、不写交付物、不参与判决
（验收 `parse_contract.is_diagnostic_only` 盯着这条）。缓存里旧 `parser_version`
的记录如实分组打印，不当作违规：旧记录本就按旧版形状写，拿今天的规定追溯昨天的
产物只会把版本偏斜说成缺陷。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, load_config

MODULE_VERSION = "0.1.0"
DEFAULTS = {
    "min_documents": 5,
    "min_pages": 20,
    "min_lines": 100,
    "bbox_tolerance": 1.0,
}

# 契约叶子路径 → 实现里对应的形状。`implemented` 为空表示「这一层根本不存在」，
# 不是「换了个名字」——空值不做 broken 检查，因为删掉的层没有替代物可查。
# `doc_type` 是这个叶子在文档里的 JS 类型：缓存里同名键长成这个类型，说明文档形状
# 已经出现（`deviation_stale`）。不写 `doc_type` 的条目只查「键是否整层消失」。
DEVIATIONS: Dict[str, Dict] = {
    "pages[].blocks": {
        "implemented": "pages[].lines",
        "doc_type": list,
        "reason": "解析层以行为单位（行内保留 parts 与合并证据），不做版面块分组；"
                  "块的语义由字段级溯源承担，引入一层没有消费者的分组只会多一份要对账的账",
    },
    "pages[].blocks[].block_id": {
        "implemented": "",
        "reason": "行记录没有 id：没有块分组，也就不需要块内序号",
    },
    "pages[].blocks[].type": {
        "implemented": "",
        "reason": "行记录不带类型；表格与文本靠 tables / lines 两个容器区分，不再在行上标注",
    },
    "pages[].blocks[].lines": {
        "implemented": "",
        "reason": "blocks 那层不存在，行本身就是最小单位（它的 parts 承担了块内分行证据）",
    },
    "pages[].tables[].rows": {
        "implemented": "pages[].tables[].rows[]",
        "doc_type": int,
        "reason": "文档写行数整数，实现写行对象列表（每行带 row_index / page / cells）；"
                  "行数等于 len(rows)，不需要再存一份会跟列表对不上的计数",
    },
    "pages[].tables[].cols": {
        "implemented": "pages[].tables[].columns[]",
        "doc_type": int,
        "reason": "文档写列数整数，实现写列锚点列表（header / center / radius）；"
                  "列数是 len(columns)，锚点才是拆分单元格真正用到的信息",
    },
    "pages[].tables[].cells": {
        "implemented": "pages[].tables[].rows[].cells{}",
        "doc_type": list,
        "reason": "文档是扁平单元格数组，实现是按行分组的字典（键=表头）；"
                  "逐格字段（text / bbox / confidence）一致，只是位置挂在行下面",
    },
    "pages[].tables[].cells[].cell_id": {
        "implemented": "",
        "reason": "单元格没有独立 id：身份由 (table_id, row_index, 表头) 三元组给出，"
                  "再发一个 id 就是第二份会漂的账",
    },
    "pages[].tables[].cells[].row": {
        "implemented": "pages[].tables[].rows[].row_index",
        "reason": "行号挂在行对象上而不是每个单元格上：同一行的格子不可能属于不同行",
    },
    "pages[].tables[].cells[].col": {
        "implemented": "pages[].tables[].columns[]",
        "reason": "列身份用表头名（字典键）而不是列下标：列序换过之后下标会指向别的列，"
                  "表头名不会",
    },
    "engine": {
        "implemented": "engine",
        "doc_type": dict,
        "reason": "文档写 {name, version, text_layer, ocr} 对象，实现写引擎名字符串；"
                  "text_layer / ocr 由引擎注册表（docpipe.ocr 的 is_ocr）回答，"
                  "版本记在 parser_version 上",
    },
    "engine.name": {
        "implemented": "engine",
        "reason": "见 engine 那一条：名字就是字符串本身",
    },
    "engine.version": {
        "implemented": "parser_version",
        "reason": "版本号记在解析记录的顶层 parser_version，不塞进引擎字段",
    },
    "engine.text_layer": {
        "implemented": "",
        "reason": "这一页有没有文本层由 lines[].engine 逐行回答，比一个页级布尔更细",
    },
    "engine.ocr": {
        "implemented": "",
        "reason": "同上：逐行 engine 已经能算出这页有没有 OCR 证据，不再另存一份会过期的布尔",
    },
}

MISSING_CONTRACT_KEY = "missing_contract_key"
DEVIATION_STALE = "deviation_stale"
DEVIATION_BROKEN = "deviation_broken"
DEVIATION_UNLISTED = "deviation_unlisted"
SPEC_MISSING = "spec_missing"

BAD_DOCUMENT_ID = "bad_document_id"
BAD_PAGE_COUNT = "bad_page_count"
BAD_PAGE_SEQ = "bad_page_seq"
BAD_PAGE_SIZE = "bad_page_size"
BAD_ENGINE = "bad_engine"
BAD_LINE = "bad_line"
LINE_BBOX_OUT = "line_bbox_out"
BAD_LINE_CONFIDENCE = "bad_line_confidence"
BAD_TABLE = "bad_table"
EMPTY_COLUMNS = "empty_columns"
BAD_COLUMN_SHAPE = "bad_column_shape"
TABLE_BBOX_OUT = "table_bbox_out"
BAD_ROW_INDEX = "bad_row_index"
BAD_ROW_PAGE = "bad_row_page"
BAD_CELLS_TYPE = "bad_cells_type"
BAD_CELL = "bad_cell"
CELL_NOT_IN_COLUMNS = "cell_not_in_columns"
CELL_BBOX_OUT = "cell_bbox_out"
THIN_CORPUS = "thin_corpus"

DECLARED_PROBLEMS = frozenset({
    MISSING_CONTRACT_KEY, DEVIATION_STALE, DEVIATION_BROKEN, DEVIATION_UNLISTED,
    SPEC_MISSING, BAD_DOCUMENT_ID, BAD_PAGE_COUNT, BAD_PAGE_SEQ, BAD_PAGE_SIZE,
    BAD_ENGINE, BAD_LINE, LINE_BBOX_OUT, BAD_LINE_CONFIDENCE, BAD_TABLE,
    EMPTY_COLUMNS, BAD_COLUMN_SHAPE, TABLE_BBOX_OUT, BAD_ROW_INDEX, BAD_ROW_PAGE,
    BAD_CELLS_TYPE, BAD_CELL, CELL_NOT_IN_COLUMNS, CELL_BBOX_OUT, THIN_CORPUS,
})


def options(config: Dict) -> Dict:
    section = (config.get("evaluation") or {}).get("parse_contract") or {}
    return {
        "min_documents": int(section.get("min_documents", 5)),
        "min_pages": int(section.get("min_pages", 20)),
        "min_lines": int(section.get("min_lines", 100)),
        "bbox_tolerance": float(section.get("bbox_tolerance", 1.0)),
    }


def section121_spec(text: str) -> Dict:
    """从 SCENARIO.md 取出 §12.1 的 json 代码块并解析。"""
    marker = "\n### 12.1 "
    start = text.find(marker)
    if start < 0:
        raise ValueError("SCENARIO.md 里找不到 §12.1")
    body = text[start:]
    fence = body.find("```json")
    if fence < 0:
        raise ValueError("SCENARIO.md §12.1 里找不到 json 代码块")
    end = body.find("```", fence + len("```json"))
    if end < 0:
        raise ValueError("SCENARIO.md §12.1 的 json 代码块没有结束标记")
    spec = json.loads(body[fence + len("```json"):end])
    if not isinstance(spec, dict):
        raise ValueError("SCENARIO.md §12.1 的 json 代码块不是映射")
    return spec


def contract_paths(node, prefix: str = "") -> List[str]:
    """把 §12.1 的样例对象摊成叶子路径。

    标量数组（`bbox: [100.0, 80.0, 300.0, 104.0]`）整体算一个叶子：它是坐标元组，
    不是「一组对象」，逐元素摊开会得到四个同名字段。对象数组与空数组才往下走，
    空数组本身是叶子（文档没告诉我们元素长什么样）。
    """
    out: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(contract_paths(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        if not node or all(not isinstance(item, (dict, list)) for item in node):
            out.append(prefix)
        else:
            # 对象数组本身也是一个契约键（`pages`、`pages[].tables`）：
            # 漏了它，缺键只会以「所有子键都缺」的形式出现，定位成本更高。
            out.append(prefix)
            for item in node:
                out.extend(contract_paths(item, f"{prefix}[]"))
    else:
        out.append(prefix)
    return out


def ancestors(path: str) -> List[str]:
    """`a.b[].c[]` 的所有前缀，`[]`/`{}` 两种写法都给（偏差清单里不带尾标）。"""
    out: List[str] = []
    parts = path.split(".")
    for index in range(1, len(parts)):
        prefix = ".".join(parts[:index])
        out.append(prefix)
        if prefix.endswith("[]") or prefix.endswith("{}"):
            out.append(prefix[:-2])
    return out


def all_paths(node, prefix: str = "") -> List[str]:
    """每个节点（不只是叶子）的路径，用来查偏差清单有没有打错字。

    偏差清单常点名的是一整层容器（`pages[].blocks`、`pages[].tables[].cells`），
    它们不是叶子；拿叶子清单去校验会把合法条目误判成 typo。
    """
    out: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.append(path)
            out.extend(all_paths(value, path))
    elif isinstance(node, list) and node and any(
            isinstance(item, (dict, list)) for item in node):
        path = f"{prefix}[]"
        out.append(path)
        out.extend(all_paths(node[0], path))
    return out


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[\]|\{\})?")


def values_at(nodes: List[object], path: str) -> List[object]:
    """按 `a.b[].c{}` 形状在若干根对象里找值；走不通返回空列表。

    中间层的列表会被摊平（`pages[].tables[].table_id` 里 `tables` 值本身是列表，
    不摊平下一段就取不到）；**末段原样返回**——`tables[].cells` 这种偏差要靠
    「末段值是不是文档里那种类型」来判断，摊平会把形状证据抹掉。
    """
    tokens = _TOKEN.findall(path)
    current = list(nodes)
    for index, part in enumerate(tokens):
        name, kind = part, ""
        if part.endswith("[]") or part.endswith("{}"):
            name, kind = part[:-2], part[-2:]
        last = index == len(tokens) - 1
        found: List[object] = []
        for node in current:
            if not (isinstance(node, dict) and name in node):
                continue
            value = node[name]
            if kind == "[]":
                if isinstance(value, list):
                    found.extend(value)
                else:
                    found.append(value)
            elif not last and isinstance(value, list):
                found.extend(value)
            elif not kind and isinstance(value, list) and not value:
                # 末段是空列表：键在，但里面什么都没有。「有没有这一层」不能靠
                # 「键的字面量出现过」回答——空的 lines 与没有 lines 对下游是同一件事。
                continue
            else:
                found.append(value)
        if not found:
            return []
        current = found
    return current


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _four(value) -> bool:
    return (isinstance(value, list) and len(value) == 4
            and all(_number(v) for v in value))


def _inside(bbox, width, height, tol: float) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 >= -tol and y0 >= -tol and x1 <= width + tol and y1 <= height + tol
            and x1 > x0 and y1 > y0)


def _applicable(records: List[Dict], path: str) -> bool:
    """这条偏差在这份语料上有没有可比的对象。

    偏差说的是「文档这一层换成了实现的那一层」。语料里压根没有表格时
    `pages[].tables[].rows` 的替代形状当然找不到——那是语料没覆盖到，不是清单
    在说谎。判据：看它的直接父层（父层本身是偏差就换成认领的实现路径）在语料里
    至少出现一次。
    """
    parents = ancestors(path)
    if not parents:
        return True
    container = parents[-1]
    entry = DEVIATIONS.get(container)
    if entry is not None:
        container = str(entry.get("implemented") or "")
    return not container or bool(values_at(records, container))


def audit_contract(records: List[Dict], spec: Dict) -> Dict:
    """契约键这一问：每个叶子要么在缓存里，要么被偏差清单（自己或祖先）认领。

    偏差清单单独整张过一遍：`engine` 这种「文档里是对象、实现里是字符串」的条目
    本身不是叶子，跟着叶子循环走会漏掉它的双向检查。
    """
    details: List[Dict] = []
    counted = present_keys = inherited = 0
    paths = contract_paths(spec)
    for path in paths:
        counted += 1
        if path in DEVIATIONS:
            continue
        parent = next((name for name in ancestors(path) if name in DEVIATIONS), None)
        if parent is not None:
            # 父层整层不存在，子键不可能单独存在：由父级那条偏差一次认领，
            # 不要求为每个子键再抄一遍理由。
            inherited += 1
            continue
        if values_at(records, path):
            present_keys += 1
        else:
            details.append({"code": MISSING_CONTRACT_KEY, "path": path,
                            "detail": "§12.1 写了这个键，缓存里没有，也没在 DEVIATIONS 认领"})

    known = set(all_paths(spec))
    for path in sorted(DEVIATIONS):
        entry = DEVIATIONS[path]
        if path not in known:
            details.append({"code": DEVIATION_UNLISTED, "path": path,
                            "detail": "偏差清单点了一个 §12.1 里不存在的路径（打错字等于假豁免）"})
            continue
        doc_type = entry.get("doc_type")
        if doc_type is not None and any(isinstance(value, doc_type)
                                        for value in values_at(records, path)):
            details.append({"code": DEVIATION_STALE, "path": path,
                            "detail": f"缓存里已经出现文档形状（{doc_type.__name__}），"
                                      f"这条偏差该删"})
        implemented = str(entry.get("implemented") or "")
        if implemented and _applicable(records, path) and not values_at(records, implemented):
            details.append({"code": DEVIATION_BROKEN, "path": path,
                            "detail": f"认领的替代形状 {implemented} 在缓存里一次都没出现"})
    return {"paths": counted, "present": present_keys, "deviations": len(DEVIATIONS),
            "inherited": inherited, "details": details}


def audit_invariants(records: List[Dict], tolerance: float) -> Dict:
    """语义不变量：形状对不对之外，这些数彼此自洽吗。"""
    details: List[Dict] = []
    totals = {"documents": 0, "pages": 0, "lines": 0, "tables": 0,
              "rows": 0, "cells": 0}

    def add(code: str, path: str, detail: str) -> None:
        details.append({"code": code, "path": path, "detail": detail})

    for record in records:
        doc_id = record.get("document_id")
        name = str(doc_id) if isinstance(doc_id, str) else "<无 document_id>"
        totals["documents"] += 1
        if not isinstance(doc_id, str) or not doc_id:
            add(BAD_DOCUMENT_ID, "", f"document_id={doc_id!r}")
        if not isinstance(record.get("engine"), str):
            add(BAD_ENGINE, "", "文档级 engine 不是字符串")
        pages = record.get("pages") or []
        if record.get("page_count") != len(pages):
            add(BAD_PAGE_COUNT, "", f"page_count={record.get('page_count')!r} "
                                    f"实际页={len(pages)}")
        for index, page in enumerate(pages, start=1):
            totals["pages"] += 1
            where = f"p{index}"
            if page.get("page") != index:
                add(BAD_PAGE_SEQ, where, f"页序号={page.get('page')!r}")
            if not isinstance(page.get("engine"), str):
                add(BAD_ENGINE, where, "页级 engine 不是字符串")
            width, height = page.get("width"), page.get("height")
            if not (_number(width) and _number(height) and width > 0 and height > 0):
                add(BAD_PAGE_SIZE, where, f"{width!r}x{height!r}")
                continue
            for line in page.get("lines") or []:
                totals["lines"] += 1
                bbox = line.get("bbox")
                if not (isinstance(line.get("text"), str) and _four(bbox)):
                    add(BAD_LINE, where, "行缺 text 或 bbox")
                    continue
                if not _inside(bbox, width, height, tolerance):
                    add(LINE_BBOX_OUT, where, f"bbox={bbox}")
                conf = line.get("confidence")
                if not (_number(conf) and 0.0 <= float(conf) <= 1.0):
                    add(BAD_LINE_CONFIDENCE, where, f"confidence={conf!r}")
            for table in page.get("tables") or []:
                totals["tables"] += 1
                table_id = table.get("table_id")
                if not (isinstance(table_id, str) and isinstance(table.get("rows"), list)
                        and isinstance(table.get("columns"), list)):
                    add(BAD_TABLE, where, f"table_id={table_id!r}")
                    continue
                raw = table.get("columns") or []
                if not raw:
                    add(EMPTY_COLUMNS, table_id, "没有列锚点")
                headers = set()
                for item in raw:
                    if not (isinstance(item, dict) and isinstance(item.get("header"), str)):
                        add(BAD_COLUMN_SHAPE, table_id, f"列锚点={item!r}")
                        continue
                    headers.add(item["header"])
                tbox = table.get("bbox")
                if _four(tbox) and not _inside(tbox, width, height, tolerance):
                    add(TABLE_BBOX_OUT, table_id, f"bbox={tbox}")
                for position, row in enumerate(table["rows"]):
                    totals["rows"] += 1
                    if row.get("row_index") != position:
                        add(BAD_ROW_INDEX, table_id,
                            f"位置 {position} 的行写的 row_index={row.get('row_index')!r}")
                    if row.get("page") != index:
                        add(BAD_ROW_PAGE, table_id,
                            f"行 page={row.get('page')!r} 在 p{index}")
                    cells = row.get("cells")
                    if not isinstance(cells, dict):
                        add(BAD_CELLS_TYPE, table_id, f"cells 是 {type(cells).__name__}")
                        continue
                    for header, cell in cells.items():
                        totals["cells"] += 1
                        if header not in headers:
                            add(CELL_NOT_IN_COLUMNS, table_id, f"格键={header!r}")
                        if not (isinstance(cell, dict) and isinstance(cell.get("text"), str)
                                and _four(cell.get("bbox"))
                                and _number(cell.get("confidence"))):
                            add(BAD_CELL, table_id, f"格 {header!r} 三键不全")
                            continue
                        if _four(tbox) and not _inside(cell["bbox"], tbox[2], tbox[3],
                                                       tolerance):
                            add(CELL_BBOX_OUT, table_id, f"格 {header!r} bbox={cell['bbox']}")
                        elif _four(tbox) and (cell["bbox"][0] < tbox[0] - tolerance
                                              or cell["bbox"][1] < tbox[1] - tolerance):
                            add(CELL_BBOX_OUT, table_id,
                                f"格 {header!r} bbox={cell['bbox']} 越出表左上角")
    return {"totals": totals, "details": details}


def audit(records: List[Dict], spec: Dict, config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    opts = options(config)
    contract = audit_contract(records, spec)
    invariants = audit_invariants(records, opts["bbox_tolerance"])

    versions: Dict[str, int] = {}
    for record in records:
        version = str(record.get("parser_version"))
        versions[version] = versions.get(version, 0) + 1
    from . import parse as _parse
    current = str(_parse.PARSER_VERSION)
    stale = sorted(version for version in versions if version != current)

    details = list(contract["details"]) + list(invariants["details"])
    totals = invariants["totals"]
    thin: List[str] = []
    for key, floor in (("documents", opts["min_documents"]),
                       ("pages", opts["min_pages"]),
                       ("lines", opts["min_lines"])):
        if totals[key] < floor:
            thin.append(f"{key}={totals[key]} < {floor}")
            details.append({"code": THIN_CORPUS, "path": key,
                            "detail": f"{totals[key]} < 下限 {floor}：覆盖不足时"
                                      f"「0 违规」不成立"})

    for item in details:
        if item["code"] not in DECLARED_PROBLEMS:
            raise ValueError(f"未登记的问题码：{item['code']}")
    counts: Dict[str, int] = {}
    for item in details:
        counts[item["code"]] = counts.get(item["code"], 0) + 1

    return {
        "version": MODULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "options": opts,
        "contract": {"paths": contract["paths"], "present": contract["present"],
                     "deviations": contract["deviations"],
                     "inherited": contract["inherited"]},
        "totals": totals,
        "versions": versions,
        "current_version": current,
        "stale_versions": stale,
        "problems": counts,
        "problems_total": len(details),
        "details": details,
    }


def load_records(directory: Path) -> List[Dict]:
    records: List[Dict] = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            records.append(payload)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SCENARIO §12.1 统一解析输出的契约审计（只读解析缓存，不写 out/）")
    parser.add_argument("--cache", default=None, help="解析缓存目录，默认 paths.work/parsed")
    parser.add_argument("--scenario", default=str(ROOT / "SCENARIO.md"))
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    config = load_config()
    cache = Path(args.cache) if args.cache else \
        Path(config["paths"]["work"]) / "parsed"
    try:
        spec = section121_spec(Path(args.scenario).read_text(encoding="utf-8"))
    except ValueError as error:
        print(f"规格取不出来：{error}")
        return 1
    records = load_records(cache)
    payload = audit(records, spec, config)
    if not records:
        payload["details"].append({"code": THIN_CORPUS, "path": "documents",
                                   "detail": f"{cache} 里没有解析记录"})
        payload["problems"][THIN_CORPUS] = payload["problems"].get(THIN_CORPUS, 0) + 1
        payload["problems_total"] += 1
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / "parse_contract.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    totals = payload["totals"]
    print(f"缓存={cache} 文档={totals['documents']} 页={totals['pages']} "
          f"行={totals['lines']} 表={totals['tables']} 行记录={totals['rows']} "
          f"格={totals['cells']}")
    print(f"契约键={payload['contract']['paths']} 直接命中={payload['contract']['present']} "
          f"已认领偏差={payload['contract']['deviations']} "
          f"随父级认领={payload['contract']['inherited']} "
          f"版本={payload['versions']} 现行={payload['current_version']}")
    print(f"问题={payload['problems_total']} 明细={payload['problems']}")
    for item in payload["details"][:20]:
        print(f"  {item['code']}: {item['path']} {item['detail']}")
    print("report:", json_path)
    return 0 if payload["problems_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
