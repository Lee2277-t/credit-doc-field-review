"""交付审计：出货的每一格必须带着判决层的痕迹。

SCENARIO 的铁律「不把解析层输出当最终产品」以前只靠一条流水线顺序保证：`run.py` 先
`extract` 再 `assess`，最后才写 `out/`。顺序在人眼里很清楚，在产物里看不见——哪天新增
一个出口、或改了导出顺序绕过 `assess` 直接写抽取结果，字段照样有值、有标签、有溯源，
CSV 和网页版全都正常，只有「这一格有没有被判过」这件事静默消失。

这里问三个能在产物上直接回答的问题，判据全读 `validation.delivery`：

1. **每一格有没有判决痕迹**：`decision` / `confidence` / `review_reasons` 由判决层写
   （`trust.decide_field`），`grounding` 由抽取层写。查的是**键在不在**，不是值好不好：
   `review` 的格子可以没有坐标、置信度可以很低，但键必须由那一层亲手写下。
2. **状态是不是判决层会产出的状态**：`decision` 必须落在 `decided_states` 里。出现
   `accepted`、`auto`、空串这类值，等于有人在导出层自己发明了一套判决词汇。
3. **三层计数对不对得上**：每份文档的 JSON、`summary.json` 的 `documents[]`、
   `summary.json` 的 `overall`。对不上就是「明细里一份、汇总里另一份」——这种分裂最容易
   在加了新出口后出现，而指标数字看着还是合理的。

第四个问题只在 `--zip` 传了交付包时才问，但权重与前三条一样：**这个包自己说得出装了哪些
文件吗**。清单必须逐字记全每个成员和它的内容哈希；包里有而清单没记、清单记了而包里没有、
哈希验不出这份内容，任一条都算失败。打包时点名拒收的遗留文件（`skipped_members`）非空，
就说明交付目录里混进了本轮没导出的东西——那些东西以前会直接跟着包出去。

外加一条覆盖下限（`min_delivered_documents` / `min_delivered_fields`）：一次没审到任何
交付文档的审计，输出和「全部通过」长得一模一样，所以扫空了必须算失败而不是算干净。

第五问把 `review/*.review.jsonl`（§7.2 的人工复核台账）和 `final/`（交付成品）逐格配一遍，
问的是**人签过的字有没有真的落到成品里、成品里宣称的「已复核」查不查得到是谁签的**。
这条以前只由流水线顺序保证：`export()` 读台账、盖 `effective_value` 与 `review_status`。
哪天台账少一条、或导出侧改了取值规则，成品里那一格照样有值、有状态、有溯源，CSV 与网页版
全都正常，只有「这个数是张三改的还是机器蒙的」这件事静默变了。拆成能各自失败的五件事：

1. **成品里有状态、台账里查无此条**（`unbacked_status`）——等于导出层自己发明了「人工已复核」。
2. **台账里有这条判决、成品里对应不上任何格**（`record_without_cell`）——人改过了，成品没带出去。
3. **同一条被复核过两次，成品跟了旧的那条**（`stale_winner`）；跟了谁都不是（`unapplied_correction`）。
4. **不该改值的动作把值改了**（`value_changed_without_correction`）、**状态与动作不配对**
   （`status_action_mismatch`）、**状态落在词汇表之外**（`unknown_state`）。
5. **复核侧三层计数**：逐格数出来的 `reviewed_count` / `reviewed_corrected_count` 必须等于
   `final_summary.json` 的 `documents[]` 与 `overall`（判据与第四问同源，只是问的是修正侧）。

台账在这里**独立解析**（全量、按出现顺序），不复用 `reviews.read_review_records`——导出侧
只留每个路径的最后一条，两边共用一个函数就永远对不上，也看不见「同一条复核了两次」。
`review/` 与成品两边都没有修正痕迹时如实打印「没有内容可审」，不冒充通过：一批刚跑完、
没人复核过是合法状态，但「0 违规」与「0 条账」必须分得开。

第六问问的是**坐标画不画得出来**：VIII 查 `grounding` 这个键在不在，验收
`output.accepted_grounding` 查原始框 `bbox` 四元、面积为正；可人在前台点的那两层（复核台
`web/app.js`、单文件网页版 `report.py`）摆方框用的是**归一化坐标** `bbox_norm`。抽取层在页面
宽高缺失时把 `bbox_norm` 写成 `None`、页码越界的框被 `grounding.page === 当前页` 那个过滤器
挡掉，两种情况下格子有值、有状态、有原始坐标、上面两条判据全绿，只有人看不见它来自哪里。
`coordinates()` 把这一层的依赖拆成六件能各自失败的事（`no_grounding` / `norm_absent` /
`norm_shape` / `norm_out_of_bounds` / `norm_degenerate` / `page_out_of_range`），再补一条
`scale_conflict`：`bbox` 与 `bbox_norm` 必须推得出同一个页面尺度（容差取归一化四舍五入的半个
刻度），只改一套坐标——补读裁片、按页宽钳制这类改动——就会被这一条抓住。值缺失的空格子不在
这一问里（前台没有东西可画，不算看不见证据）。

第七问问的是**放行这件事有没有出处**。门槛（`thresholds.global_min_confidence` +
`thresholds.risk_tiers` + `thresholds.field_overrides`）只在判决层读一次，产物里留下的是
`confidence` 与 `decision` 两个数——它们之间没有任何东西再对一遍。新增一个出口硬写
`accept`、导出层重算或覆写 `confidence`、或某格的 `risk` 落进没有配置的档，格子照样有值、
有溯源、有判决痕迹、上面六问全绿，只有「这一格凭什么放行」静默变成没有出处的数字。第七问
拿产物里的 `path`/`risk`/`confidence` 独立查一遍配置（不叫 `config.risk_threshold`，否则那个
函数改了口径这里跟着一起改，永远对得上），拆成六件能各自失败的事：`confidence_absent`（该
核对的格子拿不出一个能比较的数）、`below_global_min`、`below_risk_tier`、`unknown_risk_class`
（这一档根本没配门槛）、`accept_with_review_reasons`（判决说放行、理由却还挂着——两者不是同
一次写的）、`path_identity_split`（`fields` 的字典键与格子里的 `path` 不一致，门槛按哪个查
说不清）。两张词汇表各管各的：`gate_states` 里的状态有解释义务（`review` 的格子本来就低，那是
它进队列的原因，不是矛盾），`reason_free_states` 里的状态必须不带理由。路径覆盖出厂是空表，
所以 `field_overrides` 那一支只有夹具证明；`thresholds` 整段读不到时如实打印「没有内容可审」，
不冒充通过。

只读产物、只诊断，不参与任何判决，也不写任何文件。
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MANIFEST_NAME = "export_manifest.json"

DEFAULTS = {
    "required_field_keys": ["decision", "confidence", "review_reasons", "grounding"],
    "decided_states": ["accept", "review"],
    # 「扫到东西了没有」的下限：一次都没覆盖到交付面的审计和通过长得一模一样，
    # 所以这两条也被验收读，缺文档/缺字段就直接判失败。
    "min_delivered_documents": 5,
    "min_delivered_fields": 1000,
    # 第四问（交付包）的判据：清单必须逐字记全，且至少记到这么多条成员。
    "required_manifest_keys": ["source_out", "workspace", "exported_at", "document_count",
                               "overall", "report", "contents", "member_count", "members",
                               "skipped_members"],
    "min_package_members": 20,
    # 第五问（人工修正对账）的判据：§7.2 的台账契约、动作→状态配对、哪些动作会改值、
    # 以及复核侧的计数键。动作与状态都从这里读，代码里不留第二套词汇。
    "required_review_keys": ["document_id", "field_path", "original_value", "action",
                             "reviewer", "timestamp"],
    "review_state_pairs": {"accept": "reviewed_accept", "correct": "reviewed_corrected"},
    "value_changing_actions": ["correct"],
    # 复核侧的三层计数键：汇总里的键 → 它数的是哪个复核状态（null＝凡带状态的格都算）。
    "review_count_keys": {"reviewed_count": None, "reviewed_corrected_count": "reviewed_corrected"},
    # 第六问（坐标可核性）：人在前台点的方框画的是归一化坐标，不是原始框——「键在、四元、
    # 面积为正」全都成立时，界面那一层仍然可能什么框都不画。这三条把那个依赖写成判据。
    "drawn_coordinate_keys": ["bbox_norm"],
    "raw_coordinate_key": "bbox",
    "coordinate_tolerance": 5e-05,
    # 第七问（门自证）：哪些判决状态必须能被 `thresholds` 解释，以及拿什么键去解释。状态名与
    # 键名都从配置读，代码里不留第二套词汇——`decided_states` 那一口的教训是词汇表一旦在两处
    # 各写一遍，收窄其中一处就只有一半的人会报警。
    "gate_states": ["accept"],
    # 解释义务与「必须闭嘴」是两件事，分开两个键：前者跟 `thresholds` 比数，后者只看判决与
    # 理由是否同一次写的。合成一个键时，换掉解释对象会让后者静默停摆（详见 gates() 的注释）。
    "reason_free_states": ["accept"],
    "gate_confidence_key": "confidence",
    "gate_reason_key": "review_reasons",
}


def options(config: Dict) -> Dict:
    """`validation.delivery` 整段；缺键时退回与出厂配置逐字相同的默认值。"""
    section = ((config.get("validation") or {}).get("delivery")) or {}
    merged = dict(DEFAULTS)
    merged.update({key: value for key, value in section.items() if value is not None})
    return merged


def document_files(out_dir: Path) -> List[Path]:
    """交付目录里「每份文档一份 JSON」的那些文件；`final/`、`render/` 是派生产物，不算。"""
    found = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir() or sub.name in {"final", "render"}:
            continue
        own = sub / f"{sub.name}.json"
        if own.exists():
            found.append(own)
    return found


def field_traces(fields: Dict, opts: Dict) -> List[Dict]:
    """每格必须带着判决层与抽取层亲手写下的键；缺键或出现陌生状态就记一条。"""
    required = [str(key) for key in opts["required_field_keys"]]
    decided = {str(state) for state in opts["decided_states"]}
    offenders: List[Dict] = []
    for path, field in (fields or {}).items():
        if not isinstance(field, dict):
            offenders.append({"path": path, "problem": "not_an_object"})
            continue
        missing = [key for key in required if key not in field]
        if missing:
            offenders.append({"path": path, "problem": "missing_keys", "missing": missing})
            continue
        if field.get("decision") not in decided:
            offenders.append({"path": path, "problem": "unknown_state",
                              "decision": str(field.get("decision"))})
    return offenders


def counts(fields: Dict, opts: Dict) -> Dict[str, int]:
    """按判决状态数这一份文档的格子；`field_count` 是分母，与状态无关。"""
    states = [str(state) for state in opts["decided_states"]]
    result = {state: 0 for state in states}
    for field in (fields or {}).values():
        state = field.get("decision") if isinstance(field, dict) else None
        if state in result:
            result[state] += 1
    result["field_count"] = len(fields or {})
    return result


def package_report(zip_path: Path, opts: Dict) -> Dict:
    """第四问：交付包自己说得出装了哪些文件吗？清单与包内成员必须逐字对得上。

    打包原本是「照着 `final/` 里恰好有什么」扫，所以上一轮遗留的调试表、别人放进交付目录的
    原始 PDF 都会跟着一起出去，而包里只有一句粗粒度的 `contents`，事后查不出来。现在拆成
    四件能各自失败的事：清单**缺字段**（`missing_keys`）、包里有而清单没记（`unlisted`）、
    清单记了而包里没（`phantom`）、清单里的哈希验不出这份内容（`bad_digest`）；再加打包时
    点名拒收的 `skipped_members`——非空就是交付目录里混了不属于本轮导出的东西。
    """
    required = [str(key) for key in opts["required_manifest_keys"]]
    with zipfile.ZipFile(zip_path) as archive:
        names = {name for name in archive.namelist() if name != MANIFEST_NAME}
        raw = archive.read(MANIFEST_NAME) if MANIFEST_NAME in archive.namelist() else b""
        payloads = {name: archive.read(name) for name in sorted(names)}
    manifest = json.loads(raw.decode("utf-8")) if raw else {}
    entries = [entry for entry in (manifest.get("members") or []) if isinstance(entry, dict)]
    listed = {str(entry.get("name")) for entry in entries}
    bad_digest = [str(entry.get("name")) for entry in entries
                  if str(entry.get("name")) in payloads
                  and hashlib.sha256(payloads[str(entry.get("name"))]).hexdigest()
                  != str(entry.get("sha256"))]
    count = len(entries) + (1 if raw else 0)
    return {
        "zip": zip_path.name,
        "member_count": count,
        "declared_count": int(manifest.get("member_count") or 0),
        "missing_keys": [key for key in required if key not in manifest],
        "unlisted": sorted(names - listed),
        "phantom": sorted(listed - names),
        "bad_digest": sorted(bad_digest),
        "skipped": manifest.get("skipped_members") or [],
        "thin": count < int(opts["min_package_members"]),
    }


def _mismatch(mismatches: List[Dict], document_id: str, layer: str, problem: str,
              expected, actual) -> None:
    mismatches.append({"document_id": document_id, "layer": layer, "problem": problem,
                       "expected": expected, "actual": actual})


def _reported(entry: Dict, key: str, default) -> int:
    """汇总侧的计数键：`field_count` 原样，状态计数带 `_count` 后缀。"""
    return int(entry.get(key if key == "field_count" else f"{key}_count", default) or 0)


def _payload_ids(files: List[Path]) -> Dict[str, Dict]:
    """文档号 → 那份 JSON 的 `fields`；一次读盘，三层计数都从它算。"""
    payloads = {}
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        payloads[str(payload.get("document_id") or file.parent.name)] = payload.get("fields") or {}
    return payloads


def final_documents(out_dir: Path) -> List[Dict]:
    """交付成品侧的每份文档：`final/<文档>/<文档>.json`（应用过人工修正的那一版）。"""
    final_dir = out_dir / "final"
    loaded = []
    for sub in sorted(final_dir.iterdir()) if final_dir.is_dir() else []:
        own = sub / f"{sub.name}.json"
        if sub.is_dir() and own.exists():
            payload = json.loads(own.read_text(encoding="utf-8"))
            loaded.append({"document_id": str(payload.get("document_id") or sub.name),
                           "fields": payload.get("fields") or {}})
    return loaded


def review_ledger(out_dir: Path, doc_id: str) -> List[Dict]:
    """台账按文件出现顺序**全量**解析；坏行原样带出来，不静默跳过。

    这里故意不复用 `reviews.read_review_records`：导出侧只留每个路径的最后一条，两边共用
    一个函数就永远对得上，也就永远看不见「同一条被复核了两次、成品跟的是旧的那条」。
    """
    path = out_dir / "review" / f"{doc_id}.review.jsonl"
    if not path.exists():
        return []
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = None
        records.append(parsed if isinstance(parsed, dict) else {"_line": line.strip()[:80]})
    return records


def ledger_by_path(records: List[Dict], doc_id: str, opts: Dict, problems: List[Dict]) -> Dict:
    """合法记录按 `field_path` 归组（保留顺序）；契约缺键、动作越表、坏行都在这里点名。"""
    required = [str(key) for key in opts["required_review_keys"]]
    actions = {str(action) for action in (opts["review_state_pairs"] or {})}
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        if "_line" in record:
            problems.append({"document_id": doc_id, "path": None, "problem": "malformed_line",
                             "expected": "一行合法的 JSON 对象", "actual": record["_line"]})
            continue
        missing = [key for key in required if key not in record]
        if missing:
            problems.append({"document_id": doc_id, "path": record.get("field_path"),
                             "problem": "missing_record_keys", "expected": required,
                             "actual": missing})
            continue
        if str(record.get("action")) not in actions:
            problems.append({"document_id": doc_id, "path": record.get("field_path"),
                             "problem": "unknown_action", "expected": sorted(actions),
                             "actual": str(record.get("action"))})
            continue
        grouped.setdefault(str(record.get("field_path")), []).append(record)
    return grouped


def reconcile(out_dir: Path, opts: Dict) -> Dict:
    """第五问：人签的字有没有真的落到成品里，成品宣称的「已复核」查不查得到是谁签的。"""
    pairs = {str(action): str(state)
             for action, state in (opts["review_state_pairs"] or {}).items()}
    states = set(pairs.values())
    overwriting = {str(action) for action in (opts["value_changing_actions"] or [])}
    count_map = dict(opts["review_count_keys"])
    summary_path = out_dir / "final" / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    listed = {str(item.get("document_id")): item for item in summary.get("documents") or []}
    overall = summary.get("overall") or {}

    problems: List[Dict] = []
    mismatches: List[Dict] = []
    # 配置自己打架：声明成「能换值」的动作如果不在合法动作表里，写入侧永远产不出它，
    # 那台账里每一条真改了值的记录都会被下面那条 `value_changed_without_correction`
    # 报成越权——假指控的责任其实在词汇表，不在人。先把它点名出来。
    unpaired = sorted(overwriting - set(pairs))
    if unpaired:
        problems.append({"document_id": "*", "path": "*",
                         "problem": "value_changing_action_unpaired",
                         "expected": sorted(pairs), "actual": unpaired})
    detail = {key: 0 for key in count_map}
    via_summary = {key: 0 for key in count_map}
    record_count = 0
    documents = final_documents(out_dir)
    for document in documents:
        doc_id, fields = document["document_id"], document["fields"]
        ledger = review_ledger(out_dir, doc_id)
        record_count += len(ledger)
        grouped = ledger_by_path(ledger, doc_id, opts, problems)
        statuses = {path: field.get("review_status") for path, field in fields.items()
                    if isinstance(field, dict) and field.get("review_status")}
        for path, records in sorted(grouped.items()):
            if path not in statuses:
                field = fields.get(path)
                problems.append({"document_id": doc_id, "path": path,
                                 "problem": "record_without_cell",
                                 "expected": pairs.get(str(records[-1].get("action"))),
                                 "actual": "字段不在成品里" if field is None
                                 else f"成品里这一格没有复核状态（值 {field.get('value')!r}）"})
        for path, status in sorted(statuses.items()):
            field = fields[path]
            if status not in states:
                problems.append({"document_id": doc_id, "path": path,
                                 "problem": "unknown_state", "expected": sorted(states),
                                 "actual": str(status)})
                continue
            records = grouped.get(path) or []
            if not records:
                problems.append({"document_id": doc_id, "path": path,
                                 "problem": "unbacked_status",
                                 "expected": "台账里应有一条键齐的记录",
                                 "actual": f"成品标 {status}、值 {field.get('effective_value')!r}"})
                continue
            winner = records[-1]
            action = str(winner.get("action"))
            if pairs.get(action) != status:
                problems.append({"document_id": doc_id, "path": path,
                                 "problem": "status_action_mismatch",
                                 "expected": pairs.get(action), "actual": status})
            effective = field.get("effective_value")
            if action in overwriting:
                wanted = winner.get("corrected_value")
                if effective != wanted:
                    stale = [str(record.get("corrected_value")) for record in records[:-1]
                             if record.get("corrected_value") == effective]
                    problems.append({"document_id": doc_id, "path": path,
                                     "problem": "stale_winner" if stale else "unapplied_correction",
                                     "expected": wanted, "actual": effective,
                                     "note": f"成品跟的是更早一条记录的值 {stale[0]}" if stale else ""})
            elif effective != field.get("value"):
                problems.append({"document_id": doc_id, "path": path,
                                 "problem": "value_changed_without_correction",
                                 "expected": field.get("value"), "actual": effective,
                                 "note": f"动作 {action} 不在 value_changing_actions 里"})
        per_document = _review_counts(statuses.values(), count_map)
        for key, value in per_document.items():
            detail[key] += value
        entry = listed.get(doc_id)
        if entry is None:
            _mismatch(mismatches, doc_id, "final_summary.documents", "not_listed",
                      None, per_document)
            continue
        for key, value in per_document.items():
            via_summary[key] += _declared(entry, key, 0)
            if _declared(entry, key) != value:
                _mismatch(mismatches, doc_id, "final_summary.documents", key,
                          _declared(entry, key), value)
    # 没有 `final/` 交付侧就没有「声明」可验：汇总里查无此键是这一问没内容，不是对不上。
    if documents:
        for key, value in detail.items():
            if _declared(overall, key) != value:
                _mismatch(mismatches, "*", "final_summary.overall", key,
                          _declared(overall, key), value)
            if via_summary[key] != _declared(overall, key, 0):
                _mismatch(mismatches, "*", "documents_vs_overall", key, via_summary[key],
                          _declared(overall, key, 0))
    return {"documents": len(documents), "record_count": record_count,
            "cells": detail, "problems": problems, "mismatches": mismatches,
            "no_trail": not documents or (record_count == 0 and not any(detail.values()))}


def _review_counts(statuses, count_map: Dict) -> Dict[str, int]:
    """按 `review_count_keys` 的「汇总键 → 复核状态」映射数这一份文档的格子。"""
    result = {}
    for key, state in count_map.items():
        result[str(key)] = (len(list(statuses)) if state is None
                            else sum(1 for status in statuses if status == state))
    return result


def _declared(entry: Dict, key: str, missing: int = -1) -> int:
    """汇总侧声明的复核计数：缺键或非数字算 `missing`。合法的 `0` 不许算成缺键——第一版写成
    `int(entry.get(key) or -1)`，于是出厂产物 10 份文档里 20 个「0 条修正」被当成对不上。"""
    try:
        return int(entry.get(key))
    except (TypeError, ValueError):
        return missing


def _four_numbers(values) -> bool:
    """四元坐标框：长度为 4 的数字列表；`bool` 不算数字（JSON 里 true 会被 isinstance(int) 收下）。"""
    return (isinstance(values, (list, tuple)) and len(values) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values))


def _scale_window(value: float, rounded: float, tol: float) -> Optional[Tuple[float, float]]:
    """`round(value / 页面尺度) == rounded` 时页面尺度的可行区间；`tol` 是产物里四舍五入的半个刻度。

    归一化值贴 0 或贴 1 时区间只剩一边，推不出唯一尺度——这种格子不算违反一致性（返回的区间
    照样能和另一条边求交），拿不到结论就不下结论。
    """
    if value <= 0 or rounded + tol <= 0:
        return None
    lo, hi = rounded - tol, rounded + tol
    lower = value / hi if hi > 0 else 0.0
    upper = value / lo if lo > 0 else float("inf")
    return (lower, upper)


def _intersect(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> bool:
    """两条边各自推出的页面尺度区间**不相交**时为真（＝两套坐标不是同一次归一化写出来的）。

    任何一条边推不出区间（框贴 0 或贴页面边缘）就不下结论——「看不见证据」不等于「有矛盾」，
    把取不到结论算成违规会让一整批合法格子假报警。
    """
    if a is None or b is None:
        return False
    return a[1] < b[0] or b[1] < a[0]


def _page_limit(payload: Dict) -> Optional[int]:
    """`source.page_count`；拿不到就不判页码越界（产物结构缺失不等于坐标错）。"""
    try:
        limit = int((payload.get("source") or {}).get("page_count"))
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


def coordinates(out_dir: Path, opts: Dict) -> Dict:
    """第六问：交付的每一格坐标，人在前台那两层画得出来吗？

    VIII 问的是 `grounding` 这个**键在不在**，验收 `output.accepted_grounding` 问的是原始框
    `bbox` 有四元、面积为正。可复核台（`web/app.js`）与单文件网页版（`report.py`）摆方框用的是
    **归一化坐标**：`_grounding` 在页面宽高缺失时把 `bbox_norm` 写成 `None`，界面那个
    `field.grounding.bbox_norm` 过滤器就把这一格静默跳过——格子有值、有状态、有原始坐标、
    所有现存判据全绿，只有人看不见它来自哪里。页码越界是同一个形态（前台按
    `grounding.page === 当前页` 过滤）。拆成能各自失败的六件事：

    1. `no_grounding`——有值却没有 grounding（前台连"哪一页"都不知道）。
    2. `norm_absent` / `norm_shape`——前台要画的那套坐标没写出来，或写的不是四个数。
    3. `norm_out_of_bounds`——归一化值跑到 0..1 之外（框会画到页面外面）。
    4. `norm_degenerate`——按页面尺度宽或高为 0，点不到。
    5. `page_out_of_range`——页码不是 1..`source.page_count` 的整数。
    6. `scale_conflict`——原始框与归一化框推不出同一个页面尺度（改了 `bbox` 没重算
       `bbox_norm`，比如补读裁片或按页宽钳制只动了一套）。

    空格子（值缺失）不在这一问里：前台没有东西可画，不算看不见证据。
    """
    drawn = [str(key) for key in opts["drawn_coordinate_keys"]]
    raw_key = str(opts["raw_coordinate_key"])
    tol = float(opts["coordinate_tolerance"])
    offenders: List[Dict] = []
    counts = {"cells": 0, "audited": 0, "accepted": 0}
    verdicts: Dict[str, int] = {}

    def flag(document_id: str, path: str, problem: str, expected: str, actual: str,
             decision) -> None:
        verdicts[problem] = verdicts.get(problem, 0) + 1
        offenders.append({"document_id": document_id, "path": path, "problem": problem,
                          "expected": expected, "actual": actual, "decision": decision})

    for payload_file in document_files(out_dir):
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
        document_id = str(payload.get("document_id") or payload_file.parent.name)
        limit = _page_limit(payload)
        for path, field in (payload.get("fields") or {}).items():
            if not isinstance(field, dict) or field.get("value") in (None, "", [], {}):
                continue
            counts["cells"] += 1
            decision = field.get("decision")
            counts["accepted"] += int(decision == "accept")
            grounding = field.get("grounding")
            if not isinstance(grounding, dict) or not grounding:
                flag(document_id, path, "no_grounding", "一个能画的框", "没有 grounding", decision)
                continue
            counts["audited"] += 1
            page = grounding.get("page")
            if not isinstance(page, int) or isinstance(page, bool) or page < 1 \
                    or (limit is not None and page > limit):
                flag(document_id, path, "page_out_of_range",
                     f"1..{limit if limit is not None else '正整数'}", repr(page), decision)
            for key in drawn:
                norm = grounding.get(key)
                if norm is None:
                    flag(document_id, path, "norm_absent", f"{key} 四元框", "缺失或为 null", decision)
                    continue
                if not _four_numbers(norm):
                    flag(document_id, path, "norm_shape", f"{key} 四元框", repr(norm), decision)
                    continue
                if any(v < 0 or v > 1 for v in norm):
                    flag(document_id, path, "norm_out_of_bounds", "0..1", list(norm), decision)
                    continue
                if norm[2] <= norm[0] or norm[3] <= norm[1]:
                    flag(document_id, path, "norm_degenerate", "面积为正", list(norm), decision)
            bbox = grounding.get(raw_key)
            norm = next((grounding.get(key) for key in drawn if _four_numbers(grounding.get(key))),
                        None)
            if norm is not None and _four_numbers(bbox):
                x = _intersect(_scale_window(float(bbox[0]), float(norm[0]), tol),
                               _scale_window(float(bbox[2]), float(norm[2]), tol))
                y = _intersect(_scale_window(float(bbox[1]), float(norm[1]), tol),
                               _scale_window(float(bbox[3]), float(norm[3]), tol))
                if x or y:
                    flag(document_id, path, "scale_conflict",
                         f"{raw_key} 与 {drawn} 由同一个页面尺度除得出来（容差 {tol}）",
                         f"{raw_key}={list(bbox)}、{drawn[0] if drawn else 'norm'}={list(norm)}",
                         decision)
    return {"drawn_keys": drawn, "raw_key": raw_key, "tolerance": tol,
            "counts": counts, "verdicts": verdicts, "offenders": offenders}


def _gate_number(value) -> Optional[float]:
    """能拿来跟门槛比的数：`bool` 与字符串都不算（JSON 里 `true` 会被 `isinstance(int)` 收下）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalized(path: str) -> str:
    """行下标折成 `[i]`——`thresholds.field_overrides` 的键就是这个形状（判决层同款写法）。"""
    return re.sub(r"\[\d+\]", "[i]", str(path))


def gates(out_dir: Path, opts: Dict, config: Dict) -> Dict:
    """第七问：每一格「自动接受」，出厂配置里的门槛解释得通吗？

    判决层（`trust.decide_field`）拿 `confidence` 跟两个数比过才敢写 `accept`：全局下限
    `thresholds.global_min_confidence`，和按风险档取的 `thresholds.risk_tiers[<risk>]`（路径
    覆盖 `thresholds.field_overrides` 优先）。可产物里只留下 `confidence` 与 `decision` 两个
    字段，这中间**没有人再对一遍**——所以「先算门槛、后写判决」这件事和第八问的判决痕迹一样
    只在代码里看得见。这里独立查一遍配置：不叫 `config.risk_threshold`，因为那个函数改了口径
    这条判据会跟着一起改，两边永远对得上、也就永远看不见「产物里的放行没有门槛支撑」。

    六件能各自失败的事：

    1. `confidence_absent`——有解释义务的格子拿不出一个能比较的数（缺键、字符串、布尔）。
    2. `below_global_min`——低于 `thresholds.global_min_confidence`。
    3. `below_risk_tier`——低于该风险档（或该路径覆盖）的门槛。
    4. `unknown_risk_class`——`risk` 不在 `risk_tiers` 里、路径又没有覆盖，等于这一格压根
       没有配门槛（判决层在这种情况下会 `KeyError`，所以这条主要防「覆盖表被人随手加一条」）。
    5. `accept_with_review_reasons`——判决说放行、`review_reasons` 却还挂着：判决与理由不是
       同一次写的（`decide_field` 在 accept 分支会把理由清空）。
    6. `path_identity_split`——`fields` 的字典键与格子里的 `path` 不一致，门槛按哪个查说不清。

    两张词汇表各管各的：`gate_states` 决定谁有**解释义务**（`review` 的格子本来就低，那是它
    进队列的原因，不是矛盾），`reason_free_states` 决定谁必须**闭嘴**（放行就不该再挂理由）。
    把两件事塞进一个键里，换一张表就会让另一件事静默停摆——第一版正是这样，`gate_states` 被
    换成 `review` 时复核格子反而因为「带着理由」被判违规。`thresholds` 整段读不到时第 1~4、6
    这四件**弃权**（只数格子、一条不报，`no_gate` 交给调用方如实打印「这一问没有内容可审」），
    否则「谁都没配门槛」会变成整批 `unknown_risk_class` 那面假墙；第 5 件跟配置无关，照查。
    判决层在 `risk` 缺失时退回 `optional_low_risk`，这里**刻意不跟着退**：出厂产物 2858 个放行格
    每一格都写着 `risk`，缺了就是「按一个没人声明过的档放行」，正是要露头的那种事。
    """
    section = (config.get("thresholds") or {}) if isinstance(config, dict) else {}
    tiers = {str(key): float(value) for key, value in (section.get("risk_tiers") or {}).items()}
    overrides = {str(key): float(value)
                 for key, value in (section.get("field_overrides") or {}).items()}
    global_min = _gate_number(section.get("global_min_confidence"))
    states = {str(state) for state in opts["gate_states"]}
    silent = {str(state) for state in opts["reason_free_states"]}
    conf_key = str(opts["gate_confidence_key"])
    reason_key = str(opts["gate_reason_key"])
    offenders: List[Dict] = []
    verdicts: Dict[str, int] = {}
    counts = {"cells": 0, "gated": 0}
    # 门槛整体读不到时解释那一半弃权；理由与判决是否同一次写的这一半跟配置无关，继续查。
    no_gate = not tiers and not overrides and global_min is None

    def flag(document_id: str, path: str, problem: str, expected: str, actual: str,
             decision) -> None:
        verdicts[problem] = verdicts.get(problem, 0) + 1
        offenders.append({"document_id": document_id, "path": path, "problem": problem,
                          "expected": expected, "actual": actual, "decision": decision})

    for payload_file in document_files(out_dir):
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
        document_id = str(payload.get("document_id") or payload_file.parent.name)
        for key, field in (payload.get("fields") or {}).items():
            if not isinstance(field, dict):
                continue
            decision = field.get("decision")
            gated = decision in states
            quiet = decision in silent
            if not gated and not quiet:
                continue
            path = str(field.get("path") if field.get("path") is not None else key)
            reasons = field.get(reason_key)
            if quiet and isinstance(reasons, (list, tuple)) and reasons:
                flag(document_id, path, "accept_with_review_reasons",
                     f"放行就不该有理由（{reason_key} 为空）", list(reasons), decision)
            if not gated:
                continue
            counts["gated"] += 1
            if no_gate:
                continue
            if str(key) != path:
                flag(document_id, path, "path_identity_split",
                     "字典键与格子里的 path 同一个", f"键 {key}、path {path}", decision)
            if tiers.get(str(field.get("risk"))) is None \
                    and overrides.get(_normalized(path)) is None \
                    and overrides.get(path) is None:
                flag(document_id, path, "unknown_risk_class",
                     f"{sorted(tiers)} 之一或路径覆盖", f"risk={field.get('risk')!r}", decision)
            counts["cells"] += 1
            confidence = _gate_number(field.get(conf_key))
            if confidence is None:
                flag(document_id, path, "confidence_absent", "一个能跟门槛比的数",
                     repr(field.get(conf_key)), decision)
                continue
            if global_min is not None and confidence < global_min:
                flag(document_id, path, "below_global_min", f"≥ {global_min}", confidence,
                     decision)
                continue
            gate = overrides.get(path, overrides.get(_normalized(path)))
            if gate is None:
                gate = tiers.get(str(field.get("risk")))
            if gate is not None and confidence < gate:
                flag(document_id, path, "below_risk_tier",
                     f"{field.get('risk')} 档 ≥ {gate}", confidence, decision)
    return {"states": sorted(states), "global_min": global_min, "tiers": tiers,
            "overrides": sorted(overrides), "counts": counts, "verdicts": verdicts,
            "offenders": offenders, "no_gate": no_gate}


def audit(out_dir: Path, config: Dict) -> Dict:
    """对着一份已交付的 `out/<数据区>` 问那三个问题。"""
    opts = options(config)
    states = [str(state) for state in opts["decided_states"]]
    keys = states + ["field_count"]
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    listed = {str(item.get("document_id")): item
              for item in summary.get("documents") or []}
    overall = summary.get("overall") or {}
    undecided, mismatches = [], []
    payloads = _payload_ids(document_files(out_dir))
    detail = {key: 0 for key in keys}
    via_summary = {key: 0 for key in keys}
    for document_id, fields in payloads.items():
        offenders = field_traces(fields, opts)
        if offenders:
            undecided.append({"file": f"{document_id}/{document_id}.json",
                              "document_id": document_id, "offender_count": len(offenders),
                              "offenders": offenders[:4]})
        actual = counts(fields, opts)
        for key in keys:
            detail[key] += actual[key]
        entry = listed.get(document_id)
        if entry is None:
            _mismatch(mismatches, document_id, "summary.documents", "not_listed",
                      None, actual["field_count"])
            continue
        for key in keys:
            via_summary[key] += _reported(entry, key, 0)
            if _reported(entry, key, -1) != actual[key]:
                _mismatch(mismatches, document_id, "summary.documents", key,
                          _reported(entry, key, -1), actual[key])
    for document_id in sorted(set(listed) - set(payloads)):
        _mismatch(mismatches, document_id, "out/<id>/<id>.json", "no_payload",
                  _reported(listed[document_id], "field_count", 0), 0)
    for key in keys:
        if _reported(overall, key, 0) != detail[key]:
            _mismatch(mismatches, "*", "summary.overall", key,
                      _reported(overall, key, 0), detail[key])
        if via_summary[key] != _reported(overall, key, 0):
            _mismatch(mismatches, "*", "documents_vs_overall", key,
                      via_summary[key], _reported(overall, key, 0))
    return {"documents": len(payloads), "fields": detail["field_count"],
            "by_state": {state: detail[state] for state in states},
            "undecided": undecided, "mismatches": mismatches, "options": opts}


def main(argv=None) -> int:
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(prog="python -m docpipe.deliveryaudit")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=None,
                        help="顺带审一个交付包：清单必须逐字对得上包内成员与内容哈希")
    args = parser.parse_args(argv)
    config = load_config()
    result = audit(args.out.resolve(), config)
    opts = result["options"]
    actuals = {"min_delivered_documents": result["documents"],
               "min_delivered_fields": result["fields"]}
    thin = [(key, actuals[key], opts[key]) for key in actuals if actuals[key] < opts[key]]
    print(f"文档 {result['documents']} 份 / 字段 {result['fields']} 格 / "
          f"状态分布 {result['by_state']}")
    print(f"必需痕迹 {opts['required_field_keys']} / 允许状态 {opts['decided_states']} / "
          f"覆盖下限 {opts['min_delivered_documents']} 份 · {opts['min_delivered_fields']} 格")
    if thin:
        for key, actual, floor in thin:
            print(f"  覆盖不足: {key} 要求 {floor}，实得 {actual}——这一轮什么都没审到")
    print(f"缺判决痕迹的文档: {len(result['undecided'])}")
    for item in result["undecided"]:
        print(f"  {item['file']} ({item['document_id']}) 共 {item['offender_count']} 格："
              f"{item['offenders']}")
    print(f"三层计数不一致: {len(result['mismatches'])}")
    for item in result["mismatches"]:
        print(f"  {item['document_id']} {item['layer']} {item['problem']} "
              f"{item['expected']} vs {item['actual']}")
    package_fail = 0
    recon = reconcile(args.out.resolve(), opts)
    print(f"人工修正对账: 交付成品 {recon['documents']} 份 / 台账 {recon['record_count']} 条 / "
          f"成品侧计数 {recon['cells']}")
    if recon["no_trail"]:
        print("  两边都没有修正痕迹：这一问没有内容可审，别当成通过")
    print(f"台账与成品对不上: {len(recon['problems'])}")
    for item in recon["problems"]:
        print(f"  {item['document_id']} {item['path']} {item['problem']} "
              f"期望 {item['expected']} 实得 {item['actual']}"
              + (f"｜{item['note']}" if item.get("note") else ""))
    print(f"复核侧计数不一致: {len(recon['mismatches'])}")
    for item in recon["mismatches"]:
        print(f"  {item['document_id']} {item['layer']} {item['problem']} "
              f"{item['expected']} vs {item['actual']}")
    coords = coordinates(args.out.resolve(), opts)
    print(f"坐标可核性: 有值格 {coords['counts']['cells']} 格"
          f"（其中自动接受 {coords['counts']['accepted']}）/ 前台画 {coords['drawn_keys']} / "
          f"与 {coords['raw_key']} 比同一尺度（容差 {coords['tolerance']}）")
    if not coords["counts"]["cells"]:
        print("  有值的格子一个都没审到：这一问没有内容可审，别当成通过")
    print(f"画不出来或两套坐标打架: {len(coords['offenders'])}")
    for item in coords["offenders"][:10]:
        print(f"  {item['document_id']} {item['path']} {item['problem']} "
              f"期望 {item['expected']} 实得 {item['actual']}"
              f"｜decision={item['decision']}")
    if len(coords["offenders"]) > 10:
        print(f"  …另有 {len(coords['offenders']) - 10} 条，判据同前")
    coordinate_fail = bool(coords["offenders"])
    gate = gates(args.out.resolve(), opts, config)
    print(f"门槛自证: 需解释的状态 {gate['states']} / 全局下限 {gate['global_min']} / "
          f"风险档 {len(gate['tiers'])} 个（路径覆盖 {len(gate['overrides'])} 条）/ "
          f"参与核对 {gate['counts']['gated']} 格")
    if gate["no_gate"]:
        print("  thresholds 段一个数都没读到：这一问没有内容可审，别当成通过")
    print(f"放行解释不通: {len(gate['offenders'])}")
    for item in gate["offenders"][:10]:
        print(f"  {item['document_id']} {item['path']} {item['problem']} "
              f"期望 {item['expected']} 实得 {item['actual']}"
              f"｜decision={item['decision']}")
    if len(gate["offenders"]) > 10:
        print(f"  …另有 {len(gate['offenders']) - 10} 条，判据同前")
    gate_fail = bool(gate["offenders"])
    if args.zip_path:
        package = package_report(args.zip_path.resolve(), opts)
        print(f"交付包 {package['zip']}: 成员 {package['member_count']} 条 / "
              f"清单声明 {package['declared_count']} 条 / 下限 {opts['min_package_members']} 条")
        for label, items in (("清单缺字段", package["missing_keys"]),
                             ("包里有而清单没记", package["unlisted"]),
                             ("清单记了而包里没有", package["phantom"]),
                             ("内容哈希对不上", package["bad_digest"])):
            if items:
                print(f"  {label}: {items}")
        if package["skipped"]:
            print(f"  打包时拒收的遗留文件: "
                  f"{[(item.get('name'), item.get('reason')) for item in package['skipped']]}")
        if package["thin"]:
            print(f"  覆盖不足: 成员 {package['member_count']} 条 < 下限 "
                  f"{opts['min_package_members']} 条——这个包基本没被审到")
        package_fail = bool(package["missing_keys"] or package["unlisted"]
                            or package["phantom"] or package["bad_digest"]
                            or package["skipped"] or package["thin"]
                            or package["member_count"] != package["declared_count"])
    return 0 if not thin and not result["undecided"] and not result["mismatches"] \
        and not recon["problems"] and not recon["mismatches"] and not coordinate_fail \
        and not gate_fail and not package_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
