"""Helpers for the manual review JSONL contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

REVIEWED_ACCEPT = "reviewed_accept"
REVIEWED_CORRECTED = "reviewed_corrected"

# 复核动作词汇表的**唯一**代码来源：出厂值与 `config/scenario.yaml` 的
# `validation.delivery` 逐字相同（`deliveryaudit.DEFAULTS` 也从这里取，不再各写一份）。
ACTION_DEFAULTS = {
    "review_state_pairs": {"accept": REVIEWED_ACCEPT, "correct": REVIEWED_CORRECTED},
    "value_changing_actions": ["correct"],
}

UNREAD_SNIPPET = 120


def action_vocab(config: Optional[Dict] = None) -> Dict:
    """把 `validation.delivery` 的两个词汇键读成一份词汇表：谁能写台账、谁有权换值。

    以前这份词汇表有四个主人：配置（只有 `deliveryaudit` 读）、`serve.append_review` 的
    `{"accept", "correct"}`、`export._final_field` 与 `reviews` 里的 `== "correct"`、界面与
    单文件网页版的字面量。实测「只改配置那一半」的形态是：台账拒收新动作、成品照旧取值、
    而审计把一次合法的人工修正报成「越权改值」——配置声明能改的东西，代码里根本改不动。
    """
    section = ((config or {}).get("validation") or {}).get("delivery") or {}
    # 只在**键不存在或写着 null** 时退回出厂值；配成空表是操作者在说「一个动作都不改值」，
    # 用 `or` 会把这句合法的话读成「没配」，然后悄悄塞回出厂的那一套——配置写着空、代码当没写。
    def configured(key: str):
        value = section.get(key)
        return ACTION_DEFAULTS[key] if value is None else value

    pairs = {str(action): str(state) for action, state in configured("review_state_pairs").items()}
    changing = [str(action) for action in configured("value_changing_actions")]
    orphans = [action for action in changing if action not in pairs]
    if orphans:
        # 声明成「能换值」的动作如果自己都不在合法动作表里，写入侧永远产不出它，
        # 审计也就永远查不到它——两处打架时宁可当场报错，不要留一条静默的判据。
        raise ValueError(
            f"value_changing_actions 里的动作不在 review_state_pairs 里：{orphans}")
    confirming = [action for action in pairs if action not in changing]
    corrected_state = pairs[changing[0]] if changing else None
    return {"pairs": pairs, "actions": set(pairs), "changing": changing,
            "value_changing": set(changing), "confirming": confirming,
            "states": set(pairs.values()), "corrected_state": corrected_state,
            "confirming_state": pairs[confirming[0]] if confirming else REVIEWED_ACCEPT}


def read_review_ledger(out_dir: Path, doc_id: str) -> Dict:
    """台账的完整读法：能用的记录（每个路径最后一条）+ 原样读不出来的行。

    坏行不能算记录，但**必须数得出来**——静默跳过会让一次粘连写入同时吞掉两条人工修正，
    而界面照样说「已保存」、汇总里的 `reviewed_count` 也跟着少。`unread` 就是给写入侧和
    复核台看的第二个答案：这一格读不回来 ≠ 这一格没发生过。
    """
    path = out_dir / "review" / f"{doc_id}.review.jsonl"
    records: Dict[str, Dict] = {}
    usable: List[Dict] = []
    unread: List[str] = []
    stale: List[Dict] = []
    if not path.exists():
        return {"records": records, "all": usable, "unread": unread, "stale": stale, "lines": 0}
    source_hash = None
    document_path = out_dir / doc_id / f"{doc_id}.json"
    if document_path.exists():
        document = json.loads(document_path.read_text(encoding="utf-8"))
        source_hash = (document.get("source") or {}).get("sha256")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        text = line.strip()
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            # 一行合法 JSON 但不是对象（裸字符串、数组、null）同样进不了记录。
            unread.append(text[:UNREAD_SNIPPET])
            continue
        if not isinstance(record, dict) or not record.get("field_path"):
            unread.append(text[:UNREAD_SNIPPET])
            continue
        if record.get("source_sha256") and record["source_sha256"] != source_hash:
            stale.append(record)
            continue
        usable.append(record)
        records[str(record["field_path"])] = record
    return {"records": records, "all": usable, "unread": unread,
            "stale": stale, "lines": len(lines)}


def read_review_records(out_dir: Path, doc_id: str) -> Dict[str, Dict]:
    """只要能用的那部分；「有几行读不出来」请走 `read_review_ledger`。"""
    return read_review_ledger(out_dir, doc_id)["records"]


def known_records(fields: Dict[str, Dict], records: Dict[str, Dict]) -> Dict[str, Dict]:
    return {path: record for path, record in records.items() if path in fields}


def payload_vocab(config: Optional[Dict] = None) -> Dict:
    """词汇表里界面需要的那一份（只放能进 JSON 的形状）：复核台与单文件网页版都读它。

    多个确认/修正动作时取配置里的第一个——界面上只有两个按钮，取第一个是「按钮跟着配置走」，
    不是「再发明一套词汇」。
    """
    vocab = action_vocab(config)
    return {"confirming": list(vocab["confirming"]), "changing": list(vocab["changing"]),
            "pairs": dict(vocab["pairs"]), "corrected_state": vocab["corrected_state"],
            "confirming_state": vocab["confirming_state"],
            "confirm_action": vocab["confirming"][0] if vocab["confirming"] else None,
            "correct_action": vocab["changing"][0] if vocab["changing"] else None}


def status_from_vocab(vocab: Dict, record: Dict) -> str:
    """台账里的动作 → 成品上的复核状态；动作表只从 `action_vocab` 拿。

    表外的动作退回「确认通过」那一档，与旧实现（非 `correct` 一律算 `reviewed_accept`）同形。
    """
    return vocab["pairs"].get(str(record.get("action")), vocab["confirming_state"])


def review_status(record: Dict, config: Optional[Dict] = None) -> str:
    return status_from_vocab(action_vocab(config), record)


def summarize(fields: Dict[str, Dict], records: Dict[str, Dict],
              config: Optional[Dict] = None) -> Dict:
    vocab = action_vocab(config)
    known = known_records(fields, records)
    corrected = sum(1 for record in known.values()
                    if str(record.get("action")) in vocab["value_changing"])
    accepted = sum(1 for record in known.values()
                   if str(record.get("action")) in set(vocab["confirming"]))
    review_count = sum(1 for field in fields.values() if field.get("decision") == "review")
    reviewed_pending = sum(
        1 for path, field in fields.items()
        if field.get("decision") == "review" and path in known
    )
    return {
        "auto_review_count": review_count,
        "reviewed_count": len(known),
        "reviewed_accept_count": accepted,
        "reviewed_corrected_count": corrected,
        "pending_review_count": review_count - reviewed_pending,
        "correction_rate": round(corrected / review_count, 4) if review_count else 0.0,
    }
