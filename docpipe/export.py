"""Merge manual review records into a final, review-applied export.

Automatic decisions in out/<id>/<id>.json stay untouched. This command writes
effective values and review statuses to a separate directory so the audit trail
remains intact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, load_config, resolve_path
from .review_actions import (field_actions, review_action_table, review_actions,
                             value_rewrite_note)
from .reviews import (action_vocab, read_review_records, status_from_vocab,
                      summarize)
from .textutil import csv_cell, now_iso

COLUMNS = ["document_id", "doc_type", "field_path", "value", "effective_value",
           "value_type", "confidence", "confidence_basis", "decision", "review_status", "page", "bbox",
           "extractor_id", "review_reasons", "review_actions", "value_rewrite", "chain_evidence"]

ORDINAL_BASES = {"heuristic", "calibrated_low_samples", "unknown"}


def _final_field(field: Dict, record: Optional[Dict], vocab: Dict) -> Dict:
    final = dict(field)
    final["effective_value"] = (
        record.get("corrected_value") if record
        and str(record.get("action")) in vocab["value_changing"]
        else field.get("value")
    )
    final["review_status"] = status_from_vocab(vocab, record) if record else None
    if record:
        final["review"] = {
            "action": record.get("action"),
            "reviewer": record.get("reviewer"),
            "timestamp": record.get("timestamp"),
            "original_value": record.get("original_value"),
            "corrected_value": record.get("corrected_value"),
        }
    return final


def _group_rows(documents: List[Dict], key: str, corrected_state: str) -> List[Dict]:
    groups: Dict[str, List[Dict]] = {}
    for doc in documents:
        name = doc["doc_type"] if key == "doc_type" else doc["document_id"]
        groups.setdefault(name, []).extend(doc["fields"].values())
    rows = []
    for name, fields in sorted(groups.items()):
        accept = sum(1 for field in fields if field["decision"] == "accept")
        # `value_changing_actions` 配成空表时没有任何状态算「人工修正」，直接数 0——
        # 拿 None 去比会把成品里所有未复核的格子（`review_status` 也是 None）算成修正。
        corrected = 0 if not corrected_state else sum(
            1 for field in fields if field.get("review_status") == corrected_state)
        rows.append({
            "doc_type" if key == "doc_type" else "document_id": name,
            "field_count": len(fields),
            "accept_count": accept,
            "review_count": len(fields) - accept,
            "reviewed_count": sum(1 for field in fields if field.get("review_status")),
            "reviewed_corrected_count": corrected,
        })
    return rows


def load_final_documents(out_dir: Path,
                         config: Optional[Dict] = None) -> tuple:
    """Return review-applied documents plus the summary they were built from."""
    config = config or load_config()
    out_dir = out_dir.resolve()
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"summary.json not found under {out_dir}; run docpipe.run first")
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    vocab = action_vocab(config)

    documents = []
    for item in source_summary.get("documents", []):
        doc_id = item["document_id"]
        document = json.loads((out_dir / item["path"]).read_text(encoding="utf-8"))
        records = read_review_records(out_dir, doc_id)
        fields = {path: _final_field(field, records.get(path), vocab)
                  for path, field in document["fields"].items()}
        document["fields"] = fields
        document["review_summary"] = summarize(fields, records, config)
        document["exported_at"] = now_iso()
        document["source_run"] = {
            "pipeline_version": source_summary.get("run", {}).get("pipeline_version"),
            "started_at": source_summary.get("run", {}).get("started_at"),
        }
        documents.append(document)

    all_fields = [field for doc in documents for field in doc["fields"].values()]
    accept = sum(1 for field in all_fields if field["decision"] == "accept")
    reviewed = sum(1 for field in all_fields if field.get("review_status"))
    corrected = sum(1 for field in all_fields
                    if vocab["corrected_state"]
                    and field.get("review_status") == vocab["corrected_state"])
    auto_review = len(all_fields) - accept
    basis = Counter(str(field.get("confidence_basis") or "unknown") for field in all_fields)
    ordinal = sum(count for name, count in basis.items() if name in ORDINAL_BASES)
    notes = (config.get("calibration", {}) or {}).get("reliability_notes") or {}
    source_calibration = source_summary.get("calibration") or {}
    final_summary = {
        "export": {
            "exported_at": now_iso(),
            "source_out": str(out_dir),
            "document_count": len(documents),
            "failed_count": len(source_summary.get("failures") or []),
            "pipeline_version": source_summary.get("run", {}).get("pipeline_version"),
        },
        "failures": source_summary.get("failures") or [],
        "calibration": {
            "calibrated": bool(source_calibration.get("calibrated")),
            "by_confidence_basis": dict(sorted(basis.items())),
            "ordinal_count": ordinal,
            "ordinal_share": round(ordinal / len(all_fields), 4) if all_fields else 0.0,
            "note": notes.get("probability" if all_fields and not ordinal else "ordinal", ""),
        },
        "review_actions": review_actions(documents, config),
        "overall": {
            "field_count": len(all_fields),
            "accept_count": accept,
            "review_count": auto_review,
            "refused_document_count": sum(1 for doc in documents
                                          if doc.get("document_refusal")),
            "reviewed_count": reviewed,
            "reviewed_corrected_count": corrected,
            "pending_review_count": sum(1 for field in all_fields
                                        if field["decision"] == "review" and not field.get("review_status")),
            "correction_rate": round(corrected / auto_review, 4) if auto_review else 0.0,
        },
        "by_doc_type": _group_rows(documents, "doc_type", vocab["corrected_state"]),
        "documents": [
            {
                "document_id": doc["document_id"],
                "doc_type": doc["doc_type"],
                **({"document_refusal": doc["document_refusal"]}
                   if doc.get("document_refusal") else {}),
                "field_count": len(doc["fields"]),
                "accept_count": sum(1 for field in doc["fields"].values()
                                    if field["decision"] == "accept"),
                "review_count": sum(1 for field in doc["fields"].values()
                                    if field["decision"] == "review"),
                **doc["review_summary"],
            }
            for doc in documents
        ],
    }
    return documents, final_summary, source_summary


def export(out_dir: Path, dest: Optional[Path] = None, config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    out_dir = out_dir.resolve()
    dest = (dest or out_dir / "final").resolve()
    documents, final_summary, _ = load_final_documents(out_dir, config)
    dest.mkdir(parents=True, exist_ok=True)

    for document in documents:
        doc_id = document["document_id"]
        target_dir = dest / doc_id
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{doc_id}.json").write_text(
            json.dumps(document, ensure_ascii=False,
                       indent=int(config.get("output", {}).get("json_indent", 2))),
            encoding="utf-8")

    (dest / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False,
                   indent=int(config.get("output", {}).get("json_indent", 2))), encoding="utf-8")

    encoding = config.get("output", {}).get("csv_encoding", "utf-8-sig")
    actions = review_action_table(config)
    with (dest / "final_fields.csv").open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for doc in documents:
            for path, field in doc["fields"].items():
                grounding = field.get("grounding") or {}
                writer.writerow({
                    "document_id": csv_cell(doc["document_id"]),
                    "doc_type": doc["doc_type"],
                    "field_path": path,
                    "value": csv_cell(field.get("value"), field.get("value_type")),
                    "effective_value": csv_cell(field.get("effective_value"), field.get("value_type")),
                    "value_type": field.get("value_type", ""),
                    "confidence": field.get("confidence", ""),
                    "confidence_basis": field.get("confidence_basis") or "unknown",
                    "decision": field.get("decision", ""),
                    "review_status": field.get("review_status") or "",
                    "page": grounding.get("page", ""),
                    "bbox": json.dumps(grounding.get("bbox")) if grounding.get("bbox") else "",
                    "extractor_id": field.get("extractor_id", ""),
                    "review_reasons": "|".join(field.get("review_reasons") or []),
                    "review_actions": "|".join(field_actions(field, actions)),
                    "value_rewrite": value_rewrite_note(field),
                    "chain_evidence": field.get("chain", ""),
                })
    return final_summary


def _readme_text(final_summary: Dict, report_name: str) -> str:
    overall = final_summary["overall"]
    calibration = final_summary.get("calibration") or {}
    return "\n".join([
        "信贷资料结构化结果 · 使用说明",
        "=" * 40,
        "",
        "1. 双击打开「%s」即可查看全部结果（离线可读，不需要联网、不需要装任何软件）。" % report_name,
        "   压缩包不解开也行：双击压缩包里的这个文件，Windows 会自动单独打开它。",
        "   - 左侧按资料类型分类，点文档名直接跳转；",
        "   - 中间是字段表：值、类型、复核状态、置信度、页码；人工修正过的会以删除线保留自动识别值；",
        "   - 右侧是原文页，点任意字段行会在原文上框出它的位置；",
        "   - 顶部可搜索、可只看「待复核 / 人工修正 / 人工已复核」，也可「打印 / 存 PDF」交给别人。",
        "2. 需要表格：final/final_fields.csv（Excel 直接打开，含最终值与复核状态）。",
        "3. 需要机器读取：final/<文档>/<文档>.json（结构化字段 + 溯源坐标）。",
        "4. 需要审计：review/<文档>.review.jsonl 逐条记录谁在什么时候改了哪个字段、原值和改后的值。",
        "",
        "本次导出：文档 %d 份，字段 %d 个，自动接受 %d 个，待复核 %d 个，人工修正 %d 个。" % (
            final_summary["export"]["document_count"], overall["field_count"],
            overall["accept_count"], overall["pending_review_count"],
            overall["reviewed_corrected_count"]),
        "解析失败 %d 份，不包含在字段结果中；详情见 final/final_summary.json 的 failures。" % (
            final_summary["export"].get("failed_count", 0)),
        "导出时间：%s" % str(final_summary["export"]["exported_at"]).replace("T", " ")[:16],
        "",
        "注意：置信度只表示自动决策的可靠程度，标为「待复核」的字段请人工确认后再使用。",
        "置信度口径：%s" % (calibration.get("note") or "见 final_summary.json 的 calibration 段"),
    ])


def _member_entry(name: str, payload: bytes) -> Dict:
    """交付清单里的一条成员：名字、字节数、内容哈希。"""
    return {"name": name, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def expected_members(final_summary: Dict) -> List[str]:
    """本轮 `export()` 亲手写出的那些交付文件（相对包内路径，用正斜杠）。

    打包不许照着磁盘上「恰好有什么」扫——交付目录里可能留着上一轮的文件、甚至别人放进
    来的原始材料（PDF/图片）。只有本轮导出的东西才是这一份交付物（决策 46）。
    """
    listed = [str(item.get("document_id")) for item in final_summary.get("documents") or []]
    return [f"final/{doc_id}/{doc_id}.json" for doc_id in listed] + [
        "final/final_summary.json", "final/final_fields.csv"]


def export_zip(out_dir: Path, dest: Optional[Path] = None,
               config: Optional[Dict] = None) -> Dict:
    """Package the human-readable report, final data and review logs into one ZIP."""
    from .report import build_report

    config = config or load_config()
    out_dir = out_dir.resolve()
    final_dir = (dest or out_dir / "final").resolve()
    final_summary = export(out_dir, final_dir, config)
    report = build_report(out_dir, final_dir, config)

    exports_dir = resolve_path(config["paths"]["work"], config) / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{out_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = exports_dir / f"{stem}.zip"
    counter = 1
    while target.exists():
        counter += 1
        target = exports_dir / f"{stem}-{counter}.zip"

    expected = set(expected_members(final_summary))
    exported_ids = {str(item.get("document_id")) for item in final_summary.get("documents") or []}
    report_path = Path(report["path"])
    members: List[Dict] = []
    skipped: List[Dict] = []

    def keep(name: str, path: Path, reason: str) -> None:
        skipped.append({"name": name, "bytes": path.stat().st_size, "reason": reason})

    guide_bytes = _readme_text(final_summary, report["name"]).encode("utf-8-sig")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report["name"])
        members.append(_member_entry(report["name"], report_path.read_bytes()))
        archive.writestr("怎么看.txt", guide_bytes)
        members.append(_member_entry("怎么看.txt", guide_bytes))
        for path in sorted(final_dir.rglob("*")):
            if not path.is_file() or path.resolve() == report_path.resolve():
                continue
            name = f"final/{path.relative_to(final_dir).as_posix()}"
            if name in expected:
                archive.write(path, name)
                members.append(_member_entry(name, path.read_bytes()))
            else:
                keep(name, path, "not_exported_this_run")
        review_dir = out_dir / "review"
        for path in sorted(review_dir.glob("*.review.jsonl")):
            name = f"review/{path.name}"
            if path.name[: -len(".review.jsonl")] in exported_ids:
                archive.write(path, name)
                members.append(_member_entry(name, path.read_bytes()))
            else:
                keep(name, path, "document_not_in_export")
        manifest = {
            "source_out": str(out_dir),
            "workspace": out_dir.name,
            "exported_at": final_summary["export"]["exported_at"],
            "document_count": final_summary["export"]["document_count"],
            "failed_count": final_summary["export"]["failed_count"],
            "failures": final_summary["failures"],
            "overall": final_summary["overall"],
            "report": {"file": report["name"], "size": report["size"],
                       "embedded_pages": report["embedded_pages"]},
            "contents": [report["name"], "怎么看.txt", "final/", "review/",
                         "export_manifest.json"],
            "member_count": len(members) + 1,
            "members": members,
            "skipped_members": skipped,
        }
        archive.writestr("export_manifest.json",
                         json.dumps(manifest, ensure_ascii=False,
                                    indent=int(config.get("output", {}).get("json_indent", 2))))
    return {
        "name": target.name,
        "path": str(target),
        "size": target.stat().st_size,
        "documents": manifest["document_count"],
        "failed_count": manifest["failed_count"],
        "fields": manifest["overall"]["field_count"],
        "dir": str(final_dir),
        "report": report,
        "members": members,
        "skipped_members": skipped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply manual reviews to a final export")
    parser.add_argument("--out", default=str(ROOT / "out"),
                        help="directory produced by docpipe.run")
    parser.add_argument("--dest", default=None,
                        help="export directory (default: <out>/final)")
    parser.add_argument("--zip", action="store_true", dest="as_zip",
                        help="package final/ + review/ into work/exports/<name>.zip")
    args = parser.parse_args(argv)
    dest = Path(args.dest) if args.dest else None
    if args.as_zip:
        result = export_zip(Path(args.out), dest)
        payload = {key: result[key] for key in ("name", "path", "size", "documents", "fields")}
        payload["report"] = result["report"]["path"]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("written:", result["path"])
        print("查看:", result["report"]["path"])
        return
    summary = export(Path(args.out), dest if dest else Path(args.out) / "final")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("written:", dest if dest else Path(args.out) / "final")


if __name__ == "__main__":
    main()
