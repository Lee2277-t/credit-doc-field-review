"""Config-driven rejection contract: reason code -> what the reviewer should check."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List


def review_action_table(config: Dict) -> Dict[str, str]:
    validation = config.get("validation") or {}
    return {str(code): str(text)
            for code, text in (validation.get("review_actions") or {}).items()}


def field_actions(field: Dict, table: Dict[str, str]) -> List[str]:
    return [table[reason]
            for reason in dict.fromkeys(field.get("review_reasons") or [])
            if table.get(reason)]


def chain_note(stats: Dict) -> str:
    """流水链被判不可靠时，说清是「多少行里错了几行」与「被哪条证据判坏」。

    只在这件事真的发生时出话：链是健康的就不许在界面上挂一条「链检查过」，
    否则每一格都带提示，这句提示当天就废掉（和 value_rewrite 里 factor==1 闭嘴同一条纪律）。
    """
    if not stats or not stats.get("chain_unreliable"):
        return ""
    clauses: List[str] = []
    triggers = stats.get("triggers") or []
    if "ratio" in triggers:
        clauses.append(f"{int(stats.get('rows') or 0)} 行里 {int(stats.get('chain_failures') or 0)} 行"
                       f"余额接不上（{(float(stats.get('failure_ratio') or 0.0) * 100):.1f}%）")
    if "closing" in triggers:
        clauses.append("末行余额与期末余额对不上")
    if "totals" in triggers:
        clauses.append("期初＋收入－支出与期末余额对不上")
    if "account" in triggers:
        clauses.append("跨页账号不是同一个")
    return "流水链被判不可靠：" + "；".join(clauses) if clauses else ""


def value_rewrite_note(field: Dict) -> str:
    """这个值为什么和纸上印的不一样——单位折算与正负号都是抽取做过的改写。

    过去这两个事只活在 `field.meta` 里，两个出口一个字不说：屏幕上 12,000、纸上印
    「1.2 万元」，或屏幕上「-500.00」、纸上那格根本没有负号，复核员都会以为屏幕上就是
    纸上的数，然后按"值抄错了"去判这一格。
    """
    meta = field.get("meta") or {}
    notes: List[str] = []
    factor = meta.get("unit_factor")
    if factor and int(factor) != 1:
        unit = str(meta.get("unit_text") or "").strip()
        notes.append(f"值按正文单位「{unit or factor}」折算过（×{int(factor):,}），"
                     f"纸上印的是折算前的数")
    elif meta.get("unit_unknown"):
        notes.append("这一项要求按正文单位折算，但正文里没找到单位说明，"
                     "值就是纸面上的原数，单位对不对只能人看")
    if meta.get("sign_source") == "direction":
        column = str(meta.get("direction_column") or "").strip()
        printed = str(meta.get("direction_text") or "").strip()
        if column:
            notes.append(f"负号是按这笔金额落在「{column}」列判出来的，纸上这一格没有负号")
        elif printed:
            notes.append(f"负号是按「收支方向」列印的「{printed}」判出来的，纸上这一格没有负号")
        else:
            notes.append("负号是按本行的收支方向判出来的，纸上这一格没有负号")
    if meta.get("sign_conflict"):
        notes.append("纸上这一格带负号，方向却读成收入，值按纸上的负号保留")
    return "；".join(notes)


def review_actions(payloads: List[Dict], config: Dict) -> Dict:
    """每个观测到的原因码配一句"该去核对什么"；配置缺文案的码进 unmapped。"""
    table = review_action_table(config)
    counts: Counter = Counter()
    for doc in payloads:
        for field in doc["fields"].values():
            for reason in dict.fromkeys(field.get("review_reasons") or []):
                counts[str(reason)] += 1
    by_reason = {
        reason: {"action": table.get(reason, ""), "field_count": count}
        for reason, count in sorted(counts.items())
    }
    return {
        "by_reason": by_reason,
        "unmapped": [reason for reason, row in by_reason.items() if not row["action"]],
    }
