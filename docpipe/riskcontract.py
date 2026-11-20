"""SCENARIO §6.3 风险档阈值契约的产物侧对账。

§6.3 是一张表：八个风险档、每档一个接受阈值、`包含字段` 一列点名哪类字段归这一档，
外加一句全局下限（任何字段 < 0.90 一律复核）。这张表是**判决的输入**：`trust.decide_field`
只读 `thresholds.risk_tiers`，所以把 `monetary` 从 0.97 改成 0.90、把某个字段从
`party_name` 挪到 `free_text`，产物里只留下一个「符合门槛」的 accept——第十二问
（`deliveryaudit.gates`）拿的还是同一份配置，照样解释得通；`configaudit` 只比
「代码兜底 == 出厂 YAML」，不比「出厂 YAML == 契约」。

这里补的就是这一环：文档是那一头，配置与字段 schema 是这一头，逐条对账。
口径：
- 阈值必须**逐字相等**（不是「不高于」）：契约写 0.97，配置写 0.98 同样是偏航，
  因为复核量与指标都跟着这个数走。
- `包含字段` 里的名字必须能在 schema 里找到（`member_unresolved`），找到的字段
  所在档的阈值**不许低于**该行阈值（`member_weaker`）——更高的档是更严，允许；
  更低就是「契约要求 0.96 的字段被当 0.95 放行」。
- `包含字段` 写「所有日期字段」时改判类型：每个 `type=date` 的字段都必须落在 date 档
  （`date_tier_not_universal`）。
- 契约点名、实现里确实没有对应字段的两处（渠道 / 上期数）登记在 `UNRESOLVED_OK`，
  每条带理由；哪天它们真出现了，记 `member_stale_exception` 提醒删登记。
- 只读 `SCENARIO.md` 与配置：不解析 PDF、不读 `out/`、不写交付物。
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, field_specs, doc_types, load_config

MODULE_VERSION = "0.1.0"
DEFAULTS = {"min_tiers": 8}

SECTION_UNREAD = "section63_unread"
SECTION_THIN = "section63_thin"
TIER_MISSING = "tier_missing"
TIER_UNLISTED = "tier_unlisted"
THRESHOLD_MISMATCH = "threshold_mismatch"
FLOOR_MISMATCH = "floor_mismatch"
MEMBER_UNRESOLVED = "member_unresolved"
MEMBER_WEAKER = "member_weaker"
MEMBER_STALE_EXCEPTION = "member_stale_exception"
DATE_TIER_NOT_UNIVERSAL = "date_tier_not_universal"
UNKNOWN_RISK_MEMBER = "unknown_risk_member"
DECLARED_PROBLEMS = frozenset({
    SECTION_UNREAD, SECTION_THIN, TIER_MISSING, TIER_UNLISTED,
    THRESHOLD_MISMATCH, FLOOR_MISMATCH, MEMBER_UNRESOLVED, MEMBER_WEAKER,
    MEMBER_STALE_EXCEPTION, DATE_TIER_NOT_UNIVERSAL, UNKNOWN_RISK_MEMBER,
})

UNIVERSAL_DATE = "所有日期字段"

# 契约里的写法与 schema 的 label 不是逐字相同的三处：只做名字解析，不改任何判据。
NAME_ALIASES = {
    "账号": ["账号", "帐号"],
    "支行名": ["开户支行"],
    "逾期次数": ["近24个月逾期账户数", "最长逾期月数"],
    "笔数": ["未结清贷款笔数", "公共记录条数", "数量"],
}

# 契约点名、实现里确实没有独立字段路径的两处。每条都要给理由；
# 哪天 schema 里真出现了同名/同义字段，audit 会报 `member_stale_exception`。
UNRESOLVED_OK = {
    "optional_low_risk": {
        "渠道": "没有独立字段：渠道信息在被截断的摘要里，§4.4 也不要求单列",
        "上期数": "利润表上期列按列下标承载，不是独立字段路径",
    },
}


def options(config: Dict) -> Dict:
    section = (config.get("evaluation") or {}).get("threshold_contract") or {}
    return {"min_tiers": int(section.get("min_tiers", 8))}


def section63_spec(path: Optional[Path] = None) -> Dict:
    """从 SCENARIO.md 现读 §6.3：八行表 + 全局下限。读不出就返回空表，由调用方报红。"""
    source = Path(path) if path else ROOT / "SCENARIO.md"
    try:
        lines = source.read_text(encoding="utf-8").split("\n")
    except OSError:
        return {"tiers": {}, "floor": None, "source": str(source)}
    start = next((i for i, line in enumerate(lines) if line.startswith("### 6.3")), None)
    end = next((i for i, line in enumerate(lines) if line.startswith("### 6.4")), None)
    if start is None or end is None or end <= start:
        return {"tiers": {}, "floor": None, "source": str(source)}
    tiers: Dict[str, Dict] = {}
    floor = None
    for line in lines[start:end]:
        row = re.match(r"^\|\s*`([a-z_]+)`\s*\|\s*(.+?)\s*\|\s*([0-9.]+)\s*\|", line)
        if row:
            tiers[row.group(1)] = {"threshold": float(row.group(3)),
                                   "members": row.group(2).strip()}
            continue
        match = re.search(r"全局下限.{0,40}?<\s*([0-9.]+)", line)
        if match and floor is None:
            floor = float(match.group(1))
    return {"tiers": tiers, "floor": floor, "source": str(source)}


def field_index(config: Dict) -> List[Dict]:
    items: List[Dict] = []
    for doc_type in sorted(doc_types(config)):
        for spec in field_specs(doc_type, config):
            items.append({
                "doc_type": doc_type,
                "path": str(spec.get("path") or ""),
                "label": spec.get("label"),
                "type": spec.get("type"),
                "risk": spec.get("risk"),
            })
    return items


def resolve(name: str, index: List[Dict]) -> List[Dict]:
    """契约里的一格名字 → schema 里的字段。先别名、再精确 label、最后互相包含。"""
    labels = NAME_ALIASES.get(name, [name])
    exact = [item for item in index if item["label"] in labels]
    if exact:
        return exact
    if name in NAME_ALIASES:
        return []
    return [item for item in index
            if item["label"] and (name in item["label"] or item["label"] in name)]


def _split_members(members: str) -> List[str]:
    return [part.strip() for part in re.split(r"[、/]", members) if part.strip()]


def audit(config: Optional[Dict] = None, spec: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    spec = spec if spec is not None else section63_spec()
    opts = options(config)
    tiers = (config.get("thresholds") or {}).get("risk_tiers") or {}
    floor = (config.get("thresholds") or {}).get("global_min_confidence")
    index = field_index(config)

    details: List[Dict] = []

    def add(code: str, where: str, detail: str) -> None:
        details.append({"code": code, "where": where, "detail": detail})

    doc_tiers = spec.get("tiers") or {}
    doc_floor = spec.get("floor")
    if not doc_tiers or doc_floor is None:
        add(SECTION_UNREAD, "SCENARIO.md §6.3", "读不出风险档表或全局下限")
    if len(doc_tiers) < opts["min_tiers"]:
        add(SECTION_THIN, "SCENARIO.md §6.3",
            f"只读出 {len(doc_tiers)} 档 < min_tiers={opts['min_tiers']}")

    if doc_floor is not None and floor is not None and float(floor) != float(doc_floor):
        add(FLOOR_MISMATCH, "thresholds.global_min_confidence",
            f"契约 {doc_floor} / 配置 {floor}")

    for tier, row in doc_tiers.items():
        if tier not in tiers:
            add(TIER_MISSING, f"thresholds.risk_tiers.{tier}", "契约里有这一档，配置里没有")
            continue
        if float(tiers[tier]) != float(row["threshold"]):
            add(THRESHOLD_MISMATCH, f"thresholds.risk_tiers.{tier}",
                f"契约 {row['threshold']} / 配置 {tiers[tier]}")
    for tier in tiers:
        if tier not in doc_tiers:
            add(TIER_UNLISTED, f"thresholds.risk_tiers.{tier}", "配置里有这一档，契约里没有")

    declared = 0
    resolved = 0
    checked_fields = 0
    for tier, row in doc_tiers.items():
        contract_threshold = float(row["threshold"])
        if row["members"].strip() == UNIVERSAL_DATE:
            for item in index:
                if item["type"] != "date":
                    continue
                checked_fields += 1
                if item["risk"] != tier:
                    add(DATE_TIER_NOT_UNIVERSAL, f"{item['doc_type']}.{item['path']}",
                        f"日期字段落在 {item['risk']!r}，契约要求全部在 {tier!r}")
            continue
        for name in _split_members(row["members"]):
            declared += 1
            hits = resolve(name, index)
            if not hits:
                reason = UNRESOLVED_OK.get(tier, {}).get(name)
                if reason is None:
                    add(MEMBER_UNRESOLVED, f"{tier}:{name}", "契约点名的字段在 schema 里找不到")
                continue
            resolved += 1
            for item in hits:
                checked_fields += 1
                risk = item["risk"]
                if risk not in tiers:
                    add(UNKNOWN_RISK_MEMBER, f"{item['doc_type']}.{item['path']}",
                        f"risk={risk!r} 不在 thresholds.risk_tiers")
                    continue
                if float(tiers[risk]) < contract_threshold:
                    add(MEMBER_WEAKER, f"{item['doc_type']}.{item['path']}",
                        f"契约按 {tier} 要求 {contract_threshold}，配置按 {risk!r} 只有 "
                        f"{tiers[risk]}")

    for tier, names in UNRESOLVED_OK.items():
        for name in names:
            if resolve(name, index):
                add(MEMBER_STALE_EXCEPTION, f"{tier}:{name}",
                    "登记为「没有对应字段」，但现在 schema 里找得到，删掉这条登记")

    for problem in details:
        if problem["code"] not in DECLARED_PROBLEMS:
            raise ValueError(f"未登记的问题码：{problem['code']}")
    counts: Dict[str, int] = {}
    for problem in details:
        counts[problem["code"]] = counts.get(problem["code"], 0) + 1

    return {
        "version": MODULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": spec.get("source"),
        "tiers": len(doc_tiers),
        "min_tiers": opts["min_tiers"],
        "floor": doc_floor,
        "config_floor": floor,
        "members": {"declared": declared, "resolved": resolved,
                    "exceptions": sum(len(names) for names in UNRESOLVED_OK.values())},
        "checked_fields": checked_fields,
        "problems": counts,
        "problems_total": len(details),
        "details": details,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="SCENARIO §6.3 风险档阈值契约审计（只读文档与配置，不写 out/）")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)
    payload = audit()
    json_path = Path(args.json_path) if args.json_path else \
        ROOT / "work" / "eval" / "threshold_contract.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    print(f"风险档={payload['tiers']} 全局下限={payload['floor']} "
          f"契约字段={payload['members']['declared']} 解析到={payload['members']['resolved']} "
          f"受检字段={payload['checked_fields']} 问题={payload['problems_total']} "
          f"明细={payload['problems']}")
    print("report:", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
