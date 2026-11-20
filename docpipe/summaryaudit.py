"""summary.json 契约审计：SCENARIO §8.3 说「至少包含」那棵键树，§8.2 定了目录形状。

这两条一直是靠人眼看：`summary.json` 里少一个键，下游按 `summary["overall"]["review_count"]`
读的脚本当场 KeyError；键都在但数对不上（`accept_count + review_count != field_count`、
分了组的 `by_doc_type` 加起来跟 `overall` 不是一回事），谁也发现不了——验收只读它想读的那几个键。
同样没人核 §8.2：`out/<id>/<id>.json` 是不是每份文档都有、`summary.documents` 里列的是不是
就是产物目录里那几个、`path` 指的文件还在不在。

所以这里做三件事，全部只读产物：

1. 从 SCENARIO 现读 §8.3 的键树（叶子路径），逐条核 `summary.json` 里有没有；
2. 核数：总数字段互相对得上、比率等于计数除总数、分组加总等于总数、每份文档的字段数加总等于总数；
3. 核 §8.2 的形状：文档目录 ↔ `summary.documents` ↔ `fields.csv` 行数三方对齐，`path` 必须落盘。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ROOT, load_config

SECTION = re.compile(r"###\s*8\.3\s*summary\.json.*?```json\s*\n(.*?)```", re.S)
SPEC_GROUPS = ("run", "overall", "by_doc_type", "by_field_path", "documents")
SKIP_DIRS = {"review", "final", "exports", "report", "log"}


def section83_spec(text: str) -> Dict:
    """从 SCENARIO §8.3 读「至少包含」的样例；读不出来返回空表，不猜。"""
    match = SECTION.search(text)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _leaves(node, prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    if isinstance(node, dict):
        found: List[Tuple[str, ...]] = []
        for key, value in node.items():
            found.extend(_leaves(value, prefix + (str(key),)))
        return found
    return [prefix] if prefix else []


def spec_paths(spec: Dict) -> List[Tuple[str, ...]]:
    """§8.3 的键树只核对象层；`by_doc_type` / `by_field_path` / `documents` 是数组，只核键名。"""
    paths: List[Tuple[str, ...]] = []
    for key in SPEC_GROUPS:
        if key not in spec:
            continue
        node = spec[key]
        if isinstance(node, dict):
            paths.extend(_leaves(node, (key,)))
        else:
            if key not in paths:
                paths.append((key,))
    return paths


def _dig(payload, path: Tuple[str, ...]):
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def key_problems(summary: Dict, paths: List[Tuple[str, ...]]) -> List[Dict]:
    problems = []
    for path in paths:
        present, _ = _dig(summary, path)
        if not present:
            problems.append({"code": "key_missing", "detail": ".".join(path)})
    for key, value in (("by_doc_type", []), ("by_field_path", []), ("documents", [])):
        node = summary.get(key)
        if not isinstance(node, list):
            problems.append({"code": "collection_shape", "detail": f"{key} 不是列表"})
    return problems


def _rounded(value, digits: int = 4) -> Optional[float]:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _row_label(row: Dict) -> str:
    return str(row.get("doc_type") or row.get("field_path") or row.get("extractor")
               or row.get("document_id") or "?")


def _required_row_problems(rows: List[Dict], required_total: int,
                           required_accept: int, label: str) -> List[Dict]:
    """分组行/文档行也要说清「必填 accept / 必填」——少一列，总数对得上也看不出口径。"""
    problems: List[Dict] = []
    present = [row for row in rows if isinstance(row, dict)]
    if not present:
        return problems
    incomplete = [row for row in present
                  if not isinstance(row.get("required_field_count"), int)
                  or not isinstance(row.get("required_accept_count"), int)]
    if incomplete:
        problems.append({"code": "count_mismatch",
                         "detail": f"{label} 有 {len(incomplete)} 行缺必填分母"
                                   f"（{_row_label(incomplete[0])}）"})
        return problems
    found = sum(row["required_field_count"] for row in present)
    found_accept = sum(row["required_accept_count"] for row in present)
    if found != required_total or found_accept != required_accept:
        problems.append({"code": "count_mismatch",
                         "detail": f"{label} 必填加总 {found_accept}/{found} != overall "
                                   f"{required_accept}/{required_total}"})
    for row in present:
        counts = (row.get("field_count"), row.get("accept_count"))
        if all(isinstance(item, int) for item in counts):
            expected_all = round(counts[1] / counts[0], 4) if counts[0] else 0.0
            if _rounded(row.get("field_accept_rate")) != expected_all:
                problems.append({"code": "rate_mismatch",
                                 "detail": f"{label}:{_row_label(row)} field_accept_rate="
                                           f"{_rounded(row.get('field_accept_rate'))}"
                                           f" 应={expected_all}"})
        required = row["required_field_count"]
        expected = round(row["required_accept_count"] / required, 4) if required else 0.0
        if _rounded(row.get("auto_accept_rate")) != expected:
            problems.append({"code": "rate_mismatch",
                             "detail": f"{label}:{_row_label(row)} auto_accept_rate="
                                       f"{_rounded(row.get('auto_accept_rate'))} 应={expected}"
                                       f"（必填 {row['required_accept_count']}/{required}）"})
    return problems


def count_problems(summary: Dict) -> List[Dict]:
    """总数、比率、分组加总、每份文档加总四层互相对账。"""
    problems: List[Dict] = []
    overall = summary.get("overall") or {}
    if not isinstance(overall, dict):
        return [{"code": "count_mismatch", "detail": "overall 不是对象"}]
    total = overall.get("field_count")
    accept = overall.get("accept_count")
    review = overall.get("review_count")
    if all(isinstance(item, int) for item in (total, accept, review)):
        if accept + review != total:
            problems.append({"code": "count_mismatch",
                             "detail": f"accept {accept} + review {review} != field_count {total}"})
    else:
        problems.append({"code": "count_mismatch", "detail": "field_count/accept_count/review_count 不是整数"})
        return problems
    required_total = overall.get("required_field_count")
    required_accept = overall.get("required_accept_count")
    if not all(isinstance(item, int) for item in (required_total, required_accept)):
        problems.append({"code": "count_mismatch",
                         "detail": "required_field_count/required_accept_count 不是整数"})
        required_total = required_accept = None
    else:
        if required_accept > required_total or required_total > total or required_accept > accept:
            problems.append({"code": "count_mismatch",
                             "detail": f"必填 {required_accept}/{required_total} 超出全部 "
                                       f"{accept}/{total}"})
        claimed = _rounded(overall.get("auto_accept_rate"))
        expected = round(required_accept / required_total, 4) if required_total else 0.0
        if claimed != expected:
            problems.append({"code": "rate_mismatch",
                             "detail": f"auto_accept_rate={claimed} 但必填 "
                                       f"{required_accept}/{required_total}={expected}"})
    # 老口径改名后仍要自己自洽：`field_accept_rate` 的分母是全部字段。
    for name, part in (("field_accept_rate", accept), ("review_rate", review)):
        claimed = _rounded(overall.get(name))
        expected = round(part / total, 4) if total else 0.0
        if claimed != expected:
            problems.append({"code": "rate_mismatch",
                             "detail": f"{name}={claimed} 但 {part}/{total}={expected}"})
    for group in ("by_doc_type", "by_field_path"):
        rows = summary.get(group) or []
        if not isinstance(rows, list):
            continue
        fields = sum(int(row.get("field_count") or 0) for row in rows if isinstance(row, dict))
        accepts = sum(int(row.get("accept_count") or 0) for row in rows if isinstance(row, dict))
        if rows and (fields != total or accepts != accept):
            problems.append({"code": "count_mismatch",
                             "detail": f"{group} 加总 {fields}/{accepts} != overall {total}/{accept}"})
        if required_total is not None:
            problems.extend(_required_row_problems(rows, required_total, required_accept, group))
    documents = summary.get("documents") or []
    if isinstance(documents, list) and documents:
        per_doc = sum(int(row.get("field_count") or 0) for row in documents if isinstance(row, dict))
        if per_doc != total:
            problems.append({"code": "count_mismatch",
                             "detail": f"documents 加总 {per_doc} != field_count {total}"})
        if required_total is not None:
            problems.extend(_required_row_problems(documents, required_total,
                                                   required_accept, "documents"))
    counted = summary.get("run", {}).get("document_count") if isinstance(summary.get("run"), dict) else None
    if isinstance(counted, int) and isinstance(documents, list) and counted != len(documents):
        problems.append({"code": "count_mismatch",
                         "detail": f"run.document_count {counted} != documents {len(documents)}"})
    return problems


def document_dirs(out_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not out_dir.exists():
        return found
    for entry in sorted(out_dir.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        path = entry / f"{entry.name}.json"
        if path.exists():
            found[entry.name] = path
    return found


def csv_rows(out_dir: Path, config: Dict) -> Optional[int]:
    path = out_dir / "fields.csv"
    if not path.exists():
        return None
    encoding = (config.get("output") or {}).get("csv_encoding", "utf-8-sig")
    body = path.read_bytes().decode(encoding, errors="replace")
    return max(0, len(list(csv.reader(io.StringIO(body)))) - 1)


def layout_problems(out_dir: Path, summary: Dict,
                    csv_count: Optional[int] = None) -> List[Dict]:
    """§8.2：目录里有几份文档、summary 里列了几份、CSV 里有几行，必须是同一批。"""
    problems: List[Dict] = []
    present = document_dirs(out_dir)
    documents = summary.get("documents") or []
    listed = {str(row.get("document_id")): row for row in documents if isinstance(row, dict)}
    for doc_id in sorted(set(present) - set(listed)):
        problems.append({"code": "document_unlisted", "detail": f"{doc_id} 在产物目录里但 summary 没列"})
    for doc_id in sorted(set(listed) - set(present)):
        problems.append({"code": "document_missing", "detail": f"{doc_id} 在 summary 里但没有 {doc_id}/{doc_id}.json"})
    for doc_id, row in sorted(listed.items()):
        target = row.get("path")
        if target:
            path = Path(str(target))
            if not path.is_absolute():
                path = out_dir / path
            if not path.exists():
                problems.append({"code": "document_path_missing",
                                 "detail": f"{doc_id} path={target} 不落盘"})
    total = (summary.get("overall") or {}).get("field_count")
    if csv_count is not None and total is not None and csv_count != total:
        problems.append({"code": "csv_rows_mismatch",
                         "detail": f"fields.csv {csv_count} 行 != summary field_count {total}"})
    return problems


def audit(out_dir: Path, config: Optional[Dict] = None,
          scenario_text: Optional[str] = None,
          summary: Optional[Dict] = None,
          csv_count: Optional[int] = None) -> Dict:
    config = config or load_config()
    text = scenario_text if scenario_text is not None else \
        (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    spec = section83_spec(text)
    paths = spec_paths(spec)
    problems: List[Dict] = []

    def note(code: str, detail: str) -> None:
        problems.append({"code": code, "detail": detail})

    if not spec:
        note("spec_unread", "SCENARIO §8.3 的样例 JSON 读不出来")
    if summary is None:
        path = out_dir / "summary.json"
        if not path.exists():
            note("summary_missing", f"{path} 不存在")
            return {"keys": len(paths), "documents": 0, "problems": problems}
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            note("summary_unreadable", f"{path} 不是合法 JSON")
            return {"keys": len(paths), "documents": 0, "problems": problems}

    problems.extend(key_problems(summary, paths))
    problems.extend(count_problems(summary))
    if csv_count is None:
        csv_count = csv_rows(out_dir, config)
    problems.extend(layout_problems(out_dir, summary, csv_count))
    return {"keys": len(paths), "documents": len(summary.get("documents") or []),
            "rows": csv_count, "problems": problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SCENARIO §8.3 summary.json 契约审计")
    parser.add_argument("--out", required=True, help="产物目录（含 summary.json 与单文档 JSON）")
    args = parser.parse_args(argv)
    result = audit(Path(args.out), load_config())
    print(f"键 {result['keys']} 条 · 文档 {result['documents']} 份 · 行 {result.get('rows')}")
    for problem in result["problems"]:
        print(f"       问题 {problem['code']}: {problem['detail']}")
    print(f"summary 契约问题: {len(result['problems'])}")
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
