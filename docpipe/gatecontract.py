"""SCENARIO §1.3 成功定义与 §10.3 阶段门槛的配置侧对账。

§1.3 把「什么算达标」写成五个数：自动接受字段错误率 ≤ 2%、金额类 ≤ 1%、所有自动
接受字段 100% 有可用坐标、整体溯源命中率 ≥ 98%、阶段 4 自动接受率 ≥ 60%。§10.3 又把
阶段 3 / 阶段 4 的门槛抄了一遍。验收与评估只读 `acceptance.*` 这一份配置——把
`min_auto_accept_rate` 从 0.60 改成 0.50，`docpipe.acceptance` 照样 PASS、评估报告
照样宣称达标，没有任何东西回头问文档同不同意。

这一条与 `riskcontract`（§6.3）同形：文档是那一头，配置是这一头，逐个数对账。
口径：
- 两节都必须读得出来；§10.3 至少 `min_stages` 行，阶段 3 与阶段 4 必须都在。
- 同一指标在两节里都写了时必须**相等**（`doc_conflict`），否则先改文档。
- 配置的五个门槛必须**逐值相等**（`gate_mismatch`），不是「更严就算对」：
  0.61 与 0.60 都会改变复核量与指标，偏航就是偏航。
- 每个门槛必须真有人读（`gate_unread`）：静态扫 `READERS` 里的模块，键一次都不出现
  就说明它只是躺在 YAML 里。
- 只读 SCENARIO 与配置：不解析 PDF、不读 `out/`、不写交付物。
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
DEFAULTS = {"min_stages": 5}

SUCCESS_UNREAD = "success_section_unread"
STAGE_UNREAD = "stage_section_unread"
STAGE_THIN = "stage_section_thin"
STAGE_MISSING = "stage_missing"
NUMBER_MISSING = "number_missing"
DOC_CONFLICT = "doc_conflict"
GATE_MISSING = "gate_missing"
GATE_MISMATCH = "gate_mismatch"
GATE_UNREAD = "gate_unread"
DECLARED_PROBLEMS = frozenset({
    SUCCESS_UNREAD, STAGE_UNREAD, STAGE_THIN, STAGE_MISSING, NUMBER_MISSING,
    DOC_CONFLICT, GATE_MISSING, GATE_MISMATCH, GATE_UNREAD,
})

# 配置键 → 文档里的中文说法；方向只用于报告，不参与比较（契约写的就是等号）。
GATES = {
    "min_auto_accept_rate": "阶段 4 自动接受率",
    "max_accepted_error_rate": "已接受字段错误率",
    "max_monetary_accepted_error_rate": "金额类已接受错误率",
    "max_accepted_grounding_missing": "已接受字段溯源缺失",
    "min_grounding_hit_rate": "整体溯源命中率",
}

# 门槛必须由这些模块读走；只写进 YAML 没人读的键等于没有门槛。
READERS = ("evaluate.py", "acceptance.py")


def options(config: Dict) -> Dict:
    section = (config.get("evaluation") or {}).get("gate_contract") or {}
    return {"min_stages": int(section.get("min_stages", 5))}


def _lines(path: Optional[Path]) -> List[str]:
    source = Path(path) if path else ROOT / "SCENARIO.md"
    try:
        return source.read_text(encoding="utf-8").split("\n")
    except OSError:
        return []


def _slice(lines: List[str], start: str, end: str) -> List[str]:
    begin = next((i for i, line in enumerate(lines) if line.startswith(start)), None)
    stop = next((i for i, line in enumerate(lines) if line.startswith(end)), None)
    if begin is None or stop is None or stop <= begin:
        return []
    return lines[begin:stop]


def _pct(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    return round(float(match.group(1)) / 100.0, 6) if match else None


def success_spec(path: Optional[Path] = None) -> Dict:
    """从 §1.3 现读五个数；读不出的键不出现在 numbers 里，由 audit 报红。"""
    lines = _lines(path)
    rows = _slice(lines, "### 1.3", "## 2.")
    text = "\n".join(rows)
    numbers: Dict[str, float] = {}
    errors = _pct(r"自动接受字段的错误率\s*[≤<]=?\s*([0-9.]+)\s*%", text)
    if errors is not None:
        numbers["max_accepted_error_rate"] = errors
    monetary = _pct(r"金额类字段\s*[≤<]=?\s*([0-9.]+)\s*%", text)
    if monetary is not None:
        numbers["max_monetary_accepted_error_rate"] = monetary
    full = _pct(r"所有自动接受字段\s*([0-9.]+)\s*%\s*有可用溯源坐标", text)
    if full is not None:
        # 「100% 有可用坐标」翻译成配置侧那条「缺失数 = 0」；不是 100 就不翻译，由人看。
        if abs(full - 1.0) < 1e-9:
            numbers["max_accepted_grounding_missing"] = 0
    hit = _pct(r"整体溯源命中率\s*[≥>]=?\s*([0-9.]+)\s*%", text)
    if hit is not None:
        numbers["min_grounding_hit_rate"] = hit
    auto = _pct(r"阶段\s*4\s*自动接受率\s*[≥>]=?\s*([0-9.]+)\s*%", text)
    if auto is not None:
        numbers["min_auto_accept_rate"] = auto
    return {"numbers": numbers, "source": str(Path(path) if path else ROOT / "SCENARIO.md")}


def _stage_numbers(text: str) -> Dict[str, float]:
    numbers: Dict[str, float] = {}
    errors = _pct(r"已接受(?:字段)?错误率[^，；]{0,8}?[≤<]=?\s*([0-9.]+)\s*%", text)
    if errors is not None:
        numbers["max_accepted_error_rate"] = errors
    monetary = _pct(r"金额[^，；]{0,4}?[≤<]=?\s*([0-9.]+)\s*%", text)
    if monetary is not None:
        numbers["max_monetary_accepted_error_rate"] = monetary
    missing = re.search(r"溯源缺失\s*=?\s*([0-9.]+)", text)
    if missing:
        numbers["max_accepted_grounding_missing"] = int(float(missing.group(1)))
    auto = _pct(r"自动接受率[^，；]{0,4}?[≥>]=?\s*([0-9.]+)\s*%", text)
    if auto is not None:
        numbers["min_auto_accept_rate"] = auto
    return numbers


def stage_spec(path: Optional[Path] = None) -> Dict:
    """从 §10.3 的表格现读每阶段门槛；只认 `| 阶段 N | … |` 两列行。"""
    lines = _lines(path)
    rows = _slice(lines, "### 10.3", "## 11.")
    stages: Dict[int, Dict[str, object]] = {}
    for line in rows:
        row = re.match(r"^\|\s*阶段\s*(\d+)\s*\|\s*(.+?)\s*\|\s*$", line)
        if not row:
            continue
        stages[int(row.group(1))] = {
            "numbers": _stage_numbers(row.group(2)),
            "text": row.group(2).strip(),
        }
    return {"stages": stages,
            "source": str(Path(path) if path else ROOT / "SCENARIO.md")}


def _same(left, right) -> bool:
    return abs(float(left) - float(right)) < 1e-9


def readers() -> Dict[str, List[str]]:
    """每个门槛键被哪些模块读到（静态扫源码文本，不 import、不执行）。"""
    found: Dict[str, List[str]] = {key: [] for key in GATES}
    for name in READERS:
        path = ROOT / "docpipe" / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for key in GATES:
            if key in text:
                found[key].append(name)
    return found


def audit(config: Optional[Dict] = None, success: Optional[Dict] = None,
          stages: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    success = success if success is not None else success_spec()
    stages = stages if stages is not None else stage_spec()
    opts = options(config)
    gates = (config.get("acceptance") or {})
    success_numbers = success.get("numbers") or {}
    stage_rows = stages.get("stages") or {}

    details: List[Dict] = []

    def add(code: str, where: str, detail: str) -> None:
        details.append({"code": code, "where": where, "detail": detail})

    if not success_numbers:
        add(SUCCESS_UNREAD, "SCENARIO.md §1.3", "读不出成功定义里的门槛数")
    for key in GATES:
        if key not in success_numbers:
            add(NUMBER_MISSING, "SCENARIO.md §1.3", f"{GATES[key]}（{key}）没有读出数")

    if not stage_rows:
        add(STAGE_UNREAD, "SCENARIO.md §10.3", "读不出阶段门槛表")
    if len(stage_rows) < opts["min_stages"]:
        add(STAGE_THIN, "SCENARIO.md §10.3",
            f"只读出 {len(stage_rows)} 行 < min_stages={opts['min_stages']}")
    for stage in (3, 4):
        if stage not in stage_rows:
            add(STAGE_MISSING, f"SCENARIO.md §10.3 阶段 {stage}", "阶段门槛表里没有这一行")

    for stage in sorted(stage_rows):
        numbers = stage_rows[stage].get("numbers") or {}
        for key, value in numbers.items():
            expected = success_numbers.get(key)
            if expected is not None and not _same(expected, value):
                add(DOC_CONFLICT, f"§10.3 阶段 {stage} vs §1.3",
                    f"{GATES.get(key, key)}：两节分别写 {value} / {expected}")

    for key, label in GATES.items():
        if key not in gates:
            add(GATE_MISSING, f"acceptance.{key}", f"§1.3 要求「{label}」，配置里没有")
            continue
        expected = success_numbers.get(key)
        if expected is not None and not _same(expected, gates[key]):
            add(GATE_MISMATCH, f"acceptance.{key}",
                f"§1.3 写 {expected} / 配置 {gates[key]}")

    read_by = readers()
    for key in GATES:
        if not read_by[key]:
            add(GATE_UNREAD, f"acceptance.{key}", f"没有任何模块读它（应读：{list(READERS)}）")

    for problem in details:
        if problem["code"] not in DECLARED_PROBLEMS:
            raise ValueError(f"未登记的问题码：{problem['code']}")
    counts: Dict[str, int] = {}
    for problem in details:
        counts[problem["code"]] = counts.get(problem["code"], 0) + 1

    return {
        "version": MODULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"success": success.get("source"), "stages": stages.get("source")},
        "metrics": len(GATES),
        "numbers_read": len(success_numbers),
        "stages": len(stage_rows),
        "min_stages": opts["min_stages"],
        "gates_compared": sum(1 for key in GATES if key in gates),
        "readers": read_by,
        "problems": counts,
        "problems_total": len(details),
        "details": details,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SCENARIO §1.3 / §10.3 门槛与 acceptance 配置对账（只读文档与配置，不写 out/）")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    payload = audit()
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / "gate_contract.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"门槛={payload['metrics']} §1.3 读出={payload['numbers_read']} "
          f"阶段={payload['stages']} 对账={payload['gates_compared']} "
          f"问题={payload['problems_total']} 明细={payload['problems']}")
    print("report:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
