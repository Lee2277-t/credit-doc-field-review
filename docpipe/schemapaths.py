"""已发布字段路径的「只许追加」门槛（SCENARIO 第 102 行）。

`doc_types.*.fields[*].path` 是发布出去的名字：复核台账按它记、校准池按它折、交付
审计按它查。改名生产侧一个字都不会报——台账里那条修正只是对不上任何格子，于是
「人判过的这一格」在导出与折池两处各静默消失一次，直到有人去翻审计才发现。

机制是把发布过的路径冻结在 `config/schema_paths.json` 里：

- `missing`：基线有、出厂配置没有 → 改名或删除，必须先迁走旧路径的台账；
- `added`：出厂配置有、基线没有 → 追加合法，但必须显式登记（`--write`），
  否则下一个人分不清「新字段」与「刚改的名」。

比较前把下标归一：配置写 `bs.transactions[i].amount`，产物里是
`bs.transactions[0].amount`，两者是同一个已发布名字。

只做诊断与守门，不参与任何流水线判决。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import ROOT, load_config

BASELINE = ROOT / "config" / "schema_paths.json"
INDEX = re.compile(r"\[\d+\]")

NOTE = ("已发布字段路径快照。SCENARIO 第 102 行：路径一经发布不得随实现变化，"
        "新增字段只能追加。加完字段跑 `python -m docpipe.schemapaths --write` 登记；"
        "比对里出现 missing 就是改名或删除。")


def canonical(path: str) -> str:
    return INDEX.sub("[i]", str(path))


def published_paths(config: Dict) -> List[str]:
    paths = set()
    for spec in (config.get("doc_types") or {}).values():
        for field in spec.get("fields") or []:
            path = field.get("path") if isinstance(field, dict) else None
            if path:
                paths.add(canonical(path))
    return sorted(paths)


def load_baseline(path: Path = BASELINE) -> Optional[List[str]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [canonical(item) for item in data.get("paths") or []]


def write_baseline(paths: Iterable[str], path: Path = BASELINE) -> None:
    payload = {"note": NOTE, "paths": sorted({canonical(item) for item in paths})}
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                     .encode("utf-8"))


def compare(shipped: Iterable[str], baseline: Iterable[str]) -> Dict[str, List[str]]:
    have = {canonical(item) for item in shipped}
    frozen = {canonical(item) for item in baseline}
    return {"missing": sorted(frozen - have), "added": sorted(have - frozen)}


def audit(config: Dict, path: Path = BASELINE) -> Dict:
    shipped = published_paths(config)
    baseline = load_baseline(path)
    if baseline is None:
        return {"shipped": shipped, "baseline": None, "missing": [], "added": [],
                "problem": f"基线不存在：{path}"}
    result = compare(shipped, baseline)
    result.update({"shipped": shipped, "baseline": baseline, "problem": ""})
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="已发布字段路径的只增不改检查")
    parser.add_argument("--config", help="配置路径，默认出厂 config/scenario.yaml")
    parser.add_argument("--baseline", help=f"基线路径，默认 {BASELINE}")
    parser.add_argument("--write", action="store_true", help="按当前配置重写基线")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    baseline = Path(args.baseline) if args.baseline else BASELINE
    if args.write:
        write_baseline(published_paths(config), baseline)
        print(f"baseline written: {baseline} paths={len(published_paths(config))}")
        return 0

    result = audit(config, baseline)
    if result["problem"]:
        print(result["problem"])
        return 2
    print(f"paths={len(result['shipped'])} baseline={len(result['baseline'])} "
          f"missing={len(result['missing'])} added={len(result['added'])}")
    for name in result["missing"]:
        print(f"  - 改名或删除：{name}")
    for name in result["added"]:
        print(f"  + 未登记的新路径（跑 --write）：{name}")
    return 1 if result["missing"] or result["added"] else 0


if __name__ == "__main__":
    sys.exit(main())
