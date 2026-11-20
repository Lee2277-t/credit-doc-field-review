"""End-to-end run: PDF folder -> parsed -> extracted -> trust -> out/.

Produces out/<document_id>/<document_id>.json, out/summary.json,
out/fields.csv and out/review/<document_id>.review.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, load_config, resolve_path
from .extract import extract_document
from .parse import document_id_for, iter_inputs, parse_document, sha256_file
from .review_actions import (chain_note, field_actions, review_action_table, review_actions,
                             value_rewrite_note)
from .reviews import read_review_records, summarize as summarize_reviews
from .textutil import csv_cell, now_iso
from .trust import Calibrator, assess

PIPELINE_VERSION = "0.4.0"
SCHEMA_VERSION = "1.0"

HARD_REASON_EXCLUDE = {"LOW_CONFIDENCE"}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _collect_documents(input_dir: Path, config: Dict,
                       out_dir: Optional[Path] = None) -> List[tuple]:
    files = iter_inputs(input_dir, config)
    manifest = None
    manifest_path = input_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = {}
    unreviewed = {}
    reserved = set()
    reviewed = set()
    if out_dir and out_dir.exists():
        for ledger in (out_dir / "review").glob("*.review.jsonl"):
            doc_id = ledger.name.removesuffix(".review.jsonl")
            reserved.add(doc_id)
            if ledger.stat().st_size:
                reviewed.add(doc_id)
        for directory in sorted(out_dir.iterdir()):
            stored = directory / f"{directory.name}.json"
            if not directory.is_dir() or not stored.is_file():
                continue
            reserved.add(directory.name)
            document = json.loads(stored.read_text(encoding="utf-8"))
            source = document.get("source") or {}
            if source.get("path") and source.get("sha256"):
                source_path = Path(source["path"]).resolve()
                previous[(source_path, source["sha256"])] = directory.name
                if directory.name not in reviewed:
                    unreviewed[source_path] = directory.name
    bases = {path: document_id_for(path, manifest) for path in files}
    assigned = {}
    for path in files:
        try:
            identity = (path.resolve(), sha256_file(path))
        except OSError:
            continue
        if identity in previous:
            assigned[path] = previous[identity]
        elif identity[0] in unreviewed:
            assigned[path] = unreviewed[identity[0]]
    used = reserved | set(assigned.values())
    for path in files:
        if path in assigned:
            continue
        base = bases[path]
        doc_id = base
        index = 2
        while doc_id in used or (doc_id != base and doc_id in bases.values()):
            doc_id = f"{base}_{index}"
            index += 1
        assigned[path] = doc_id
        used.add(doc_id)
    return [(assigned[path], path) for path in files]


def _document_payload(extracted: Dict, parsed: Dict, source_path: Path,
                      timing: Optional[Dict] = None, config: Optional[Dict] = None) -> Dict:
    quality = parsed.get("quality") or {}
    warnings = []
    if float(quality.get("parse_score") or 0.0) < float(
            (config or load_config()).get("parsing", {}).get("min_page_quality", 0.5)):
        warnings.append("PARSE_LOW_QUALITY")
    photo_flags = list(quality.get("photo_flags") or [])
    if photo_flags:
        warnings.append("PHOTO_QUALITY")
    low_resolution_pages = list(quality.get("low_resolution_pages") or [])
    if low_resolution_pages:
        warnings.append("LOW_RESOLUTION")
    preprocess_pages = [{
        "page": page["page"],
        "flags": list(page.get("quality_flags") or []),
        "deskew_degrees": (page.get("preprocess") or {}).get("geometry", {}).get("deskew_degrees", 0.0),
        "warped": bool((page.get("preprocess") or {}).get("geometry", {}).get("warped")),
        "seal_suppressed": bool((page.get("preprocess") or {}).get("enhancement", {}).get("seal_suppressed")),
        # 归一化 [x0, y0, x1, y1]，复核台直接叠在页面上：人被要求核对「章压住的字」，
        # 却只能在整页里凭肉眼找那枚章，等于把解释成本转给了复核员。
        "seal_regions": [[float(v) for v in region]
                         for region in ((page.get("preprocess") or {}).get("enhancement", {})
                                        .get("seal_regions") or [])],
        # 残片与整章要分开说：贴到页边的色块是「章只拍到半边/一角」，压住的字段同样是复核，
        # 但复核员得知道这一页本身拍不全，不能按整枚章去判断骑缝对不对得上。
        "seal_edge_fragment": bool((page.get("preprocess") or {}).get("enhancement", {})
                                   .get("seal_edge_fragment")),
        "highlight_suppressed": bool((page.get("preprocess") or {}).get("enhancement", {})
                                     .get("highlight_suppressed")),
        # 影调改写是**整页**改写（对比度拉伸还会把彩色并成灰度），比章压住一角影响面大得多，
        # 过去却一个字都不说：复核员在页面上看到淡淡的笔画，不知道送去 OCR 的那一份被拉过。
        "contrast_lifted": bool((page.get("preprocess") or {}).get("enhancement", {})
                                .get("contrast_lifted")),
        "brightness_lifted": bool((page.get("preprocess") or {}).get("enhancement", {})
                                  .get("brightness_lifted")),
    } for page in parsed.get("pages", []) if (page.get("preprocess") or {}).get("rectified")
        or page.get("quality_flags") or (page.get("preprocess") or {}).get("enhancement")]
    rules = Counter()
    hard_fail = 0
    warn_count = 0
    # 链诊断是文档级的数，只落在真正被 CHAIN_UNRELIABLE 拦下的那些格上：
    # 人被推去复核 50 个字段时，第一个问题永远是「到底错了多少行」。
    chain = chain_note(extracted.get("chain_stats") or {})
    for field in extracted["fields"].values():
        note = value_rewrite_note(field)
        if note:
            field["rewrite"] = note
        if chain and "CHAIN_UNRELIABLE" in (field.get("review_reasons") or []):
            field["chain"] = chain
        mask = any(reason not in HARD_REASON_EXCLUDE for reason in field.get("review_reasons") or [])
        if field.get("decision") == "review" and mask:
            hard_fail += 1
        for item in field.get("validators") or []:
            rules[(item["id"], item["status"])] += 1
            if item["status"] == "warn":
                warn_count += 1
    page_count = parsed.get("page_count")
    timing_payload = None
    if timing:
        total = float(timing.get("parse_seconds") or 0.0) + float(timing.get("extract_seconds") or 0.0)
        timing_payload = {
            "parse_seconds": round(float(timing.get("parse_seconds") or 0.0), 3),
            "extract_seconds": round(float(timing.get("extract_seconds") or 0.0), 3),
            "total_seconds": round(total, 3),
            "seconds_per_page": round(total / page_count, 3) if page_count else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": extracted["document_id"],
        "doc_type": extracted["doc_type"],
        "doc_type_confidence": round(float(extracted.get("doc_type_confidence") or 0.0), 3),
        "source": {
            "file_name": source_path.name,
            "path": str(source_path.resolve()),
            "sha256": parsed.get("sha256"),
            "page_count": page_count,
        },
        "extraction_timestamp": now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "document_quality": {
            "parse_score": quality.get("parse_score"),
            "ocr_mean_confidence": quality.get("mean_confidence"),
            "photo_flags": photo_flags,
            "low_resolution_pages": low_resolution_pages,
            "rectified_pages": quality.get("rectified_pages", 0),
            "grounding_dropped": quality.get("grounding_dropped", 0),
            "preprocess": preprocess_pages,
            "warnings": warnings,
        },
        "timing": timing_payload,
        "fields": extracted["fields"],
        # 类型认不出的文档一个字段都没有，拒答只能挂在文档上（trust.document_refusal）。
        **({"document_refusal": extracted["document_refusal"]}
           if extracted.get("document_refusal") else {}),
        "validation_summary": {
            "hard_fail_count": hard_fail,
            "warn_count": warn_count,
            "rules": [{"id": rule_id, "status": status, "field_count": count}
                      for (rule_id, status), count in sorted(rules.items())],
        },
        "summary": {
            "field_count": len(extracted["fields"]),
            "accept_count": sum(1 for f in extracted["fields"].values() if f["decision"] == "accept"),
            "review_count": sum(1 for f in extracted["fields"].values() if f["decision"] == "review"),
            "by_confidence_basis": dict(sorted(Counter(
                str(f.get("confidence_basis") or "unknown") for f in extracted["fields"].values()).items())),
        },
    }


def _stats(payloads: List[Dict]) -> Dict:
    fields = [field for doc in payloads for field in doc["fields"].values()]
    total = len(fields)
    accept = sum(1 for field in fields if field["decision"] == "accept")
    # 与评估报告同一个口径（§6.5 第一行）：`auto_accept_rate` 的分母是必填字段，
    # 全部字段那一版叫 `field_accept_rate`。两个数一起发，谁也冒充不了谁。
    required = [field for field in fields if field.get("required")]
    required_total = len(required)
    required_accept = sum(1 for field in required if field["decision"] == "accept")
    confidences = [float(field.get("confidence") or 0.0) for field in fields]
    reviews = [doc.get("review_summary") or {} for doc in payloads]
    review_count = sum(item.get("auto_review_count", 0) for item in reviews)
    corrected = sum(item.get("reviewed_corrected_count", 0) for item in reviews)
    timings = [doc.get("timing") or {} for doc in payloads]
    elapsed = sum(float(item.get("total_seconds") or 0.0) for item in timings)
    pages = sum(int(doc["source"].get("page_count") or 0) for doc in payloads)
    basis = Counter(str(field.get("confidence_basis") or "unknown") for field in fields)
    return {
        "field_count": total,
        "accept_count": accept,
        "review_count": total - accept,
        # 一份「什么都没抽到」的文档和一个「全都自动接受」的文档不能在汇总里长得一样。
        "refused_document_count": sum(1 for doc in payloads if doc.get("document_refusal")),
        "required_field_count": required_total,
        "required_accept_count": required_accept,
        "auto_accept_rate": round(required_accept / required_total, 4) if required_total else 0.0,
        "field_accept_rate": round(accept / total, 4) if total else 0.0,
        "review_rate": round((total - accept) / total, 4) if total else 0.0,
        "avg_confidence": round(sum(confidences) / total, 4) if total else 0.0,
        "by_confidence_basis": dict(sorted(basis.items())),
        "grounding_missing_count": sum(1 for field in fields
                                       if "GROUNDING_MISSING" in (field.get("review_reasons") or [])),
        # SCENARIO §7.1：解析阶段丢弃的无坐标块/单元格数量（与上面的字段级计数是两件事）。
        "parse_grounding_dropped_count": sum(
            int((doc.get("document_quality") or {}).get("grounding_dropped") or 0)
            for doc in payloads),
        # 编造率：原文回查覆盖到的字段里，有多少串字在文档里根本找不到。
        "raw_audited_count": sum(1 for field in fields
                                 if (field.get("meta") or {}).get("raw_grounded")
                                 or (field.get("meta") or {}).get("raw_unverified")),
        "raw_unverified_count": sum(1 for field in fields
                                    if (field.get("meta") or {}).get("raw_unverified")),
        "raw_unverified_rate": round(
            sum(1 for field in fields if (field.get("meta") or {}).get("raw_unverified"))
            / max(1, sum(1 for field in fields
                         if (field.get("meta") or {}).get("raw_grounded")
                         or (field.get("meta") or {}).get("raw_unverified"))), 4),
        "hard_fail_count": sum(
            1 for field in fields
            if field["decision"] == "review"
            and any(reason not in HARD_REASON_EXCLUDE for reason in field.get("review_reasons") or [])),
        "reviewed_count": sum(item.get("reviewed_count", 0) for item in reviews),
        "reviewed_accept_count": sum(item.get("reviewed_accept_count", 0) for item in reviews),
        "reviewed_corrected_count": corrected,
        "pending_review_count": sum(item.get("pending_review_count", 0) for item in reviews),
        "correction_rate": round(corrected / review_count, 4) if review_count else 0.0,
        "document_seconds_total": round(elapsed, 3),
        "avg_document_seconds": round(elapsed / len(payloads), 3) if payloads else 0.0,
        "avg_page_seconds": round(elapsed / pages, 3) if pages else 0.0,
    }


def _by_group(payloads: List[Dict], key: str) -> List[Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for doc in payloads:
        for field in doc["fields"].values():
            if key == "doc_type":
                groups[doc["doc_type"]].append(field)
            elif key == "extractor":
                groups[str(field.get("extractor_id") or "none").split(":")[0]].append(field)
            else:
                groups[field["path"]].append(field)
    rows = []
    name_key = {"doc_type": "doc_type", "extractor": "extractor"}.get(key, "field_path")
    for name, fields in sorted(groups.items()):
        accept = sum(1 for field in fields if field["decision"] == "accept")
        required = [field for field in fields if field.get("required")]
        required_accept = sum(1 for field in required if field["decision"] == "accept")
        confidences = [float(field.get("confidence") or 0.0) for field in fields]
        meta = [field.get("meta") or {} for field in fields]
        rows.append({
            name_key: name,
            "field_count": len(fields),
            "accept_count": accept,
            "review_count": len(fields) - accept,
            "required_field_count": len(required),
            "required_accept_count": required_accept,
            "auto_accept_rate": round(required_accept / len(required), 4) if required else 0.0,
            "field_accept_rate": round(accept / len(fields), 4) if fields else 0.0,
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "confidence_basis": Counter(str(field.get("confidence_basis") or "unknown")
                                        for field in fields).most_common(1)[0][0] if fields else "unknown",
            # 每条通路自己扛了多少、又有多少是靠兜底通路/补读救回来的：
            # 一批件里兜底占比高，说明这批件的版式根本不该走自动接受。
            "fallback_count": sum(1 for item in meta
                                  if item.get("extractor_fallback") or item.get("line_fallback")
                                  or item.get("matrix_fallback")),
            "recrop_count": sum(1 for item in meta if item.get("local_recrop")),
        })
    return rows


def _calibration_report(payloads: List[Dict], config: Dict, calibrated: bool) -> Dict:
    """How much of the reported confidence is an actual probability."""
    fields = [field for doc in payloads for field in doc["fields"].values()]
    basis = Counter(str(field.get("confidence_basis") or "unknown") for field in fields)
    ordinal = basis.get("heuristic", 0) + basis.get("calibrated_low_samples", 0)
    notes = config.get("calibration", {}).get("reliability_notes") or {}
    return {
        "calibrated": calibrated,
        "min_samples_per_field": int(config.get("calibration", {}).get("min_samples_per_field", 30)),
        "by_confidence_basis": dict(sorted(basis.items())),
        "ordinal_count": ordinal,
        "ordinal_share": round(ordinal / len(fields), 4) if fields else 0.0,
        "note": notes.get("probability" if calibrated and not ordinal else "ordinal", ""),
    }


def _write_summary(out_dir: Path, payloads: List[Dict], input_dir: Path,
                   started_at: str, config: Optional[Dict] = None,
                   calibrated: bool = False, failures: Optional[List[Dict]] = None) -> Dict:
    config = config or load_config()
    failures = failures or []
    documents = []
    for doc in payloads:
        required = [field for field in doc["fields"].values() if field.get("required")]
        required_accept = sum(1 for field in required if field["decision"] == "accept")
        field_count = doc["summary"]["field_count"]
        accept_count = doc["summary"]["accept_count"]
        documents.append({
            "document_id": doc["document_id"],
            "doc_type": doc["doc_type"],
            **({"document_refusal": doc["document_refusal"]}
               if doc.get("document_refusal") else {}),
            "source": doc["source"]["file_name"],
            "page_count": doc["source"]["page_count"],
            "parse_score": doc["document_quality"]["parse_score"],
            "photo_flags": doc["document_quality"].get("photo_flags") or [],
            "field_count": field_count,
            "accept_count": accept_count,
            "review_count": doc["summary"]["review_count"],
            "required_field_count": len(required),
            "required_accept_count": required_accept,
            "auto_accept_rate": round(required_accept / len(required), 4) if required else 0.0,
            "field_accept_rate": round(accept_count / field_count, 4) if field_count else 0.0,
            "by_confidence_basis": doc["summary"].get("by_confidence_basis") or {},
            "reviewed_count": (doc.get("review_summary") or {}).get("reviewed_count", 0),
            "pending_review_count": (doc.get("review_summary") or {}).get("pending_review_count", 0),
            "reviewed_corrected_count": (doc.get("review_summary") or {}).get("reviewed_corrected_count", 0),
            "timing": doc.get("timing"),
            "path": f"{doc['document_id']}/{doc['document_id']}.json",
        })
    summary = {
        "run": {
            "started_at": started_at,
            "finished_at": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "input_dir": _rel(input_dir),
            "document_count": len(payloads),
            "input_count": len(payloads) + len(failures),
            "failed_count": len(failures),
        },
        "overall": _stats(payloads),
        "calibration": _calibration_report(payloads, config, calibrated),
        "review_actions": review_actions(payloads, config),
        "by_doc_type": _by_group(payloads, "doc_type"),
        "by_extractor": _by_group(payloads, "extractor"),
        "by_field_path": _by_group(payloads, "field_path"),
        "documents": documents,
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False,
                   indent=int(config.get("output", {}).get("json_indent", 2))), encoding="utf-8")
    return summary


def _write_csv(out_dir: Path, payloads: List[Dict], config: Dict) -> None:
    output = config.get("output", {}) or {}
    encoding = output.get("csv_encoding", "utf-8-sig")
    columns = list(output.get("csv_columns") or []) + list(output.get("csv_appendix") or [])
    actions = review_action_table(config)
    with (out_dir / "fields.csv").open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for doc in payloads:
            for path, field in doc["fields"].items():
                grounding = field.get("grounding") or {}
                writer.writerow({
                    "document_id": csv_cell(doc["document_id"]),
                    "doc_type": doc["doc_type"],
                    "field_path": path,
                    "value": csv_cell(field.get("value"), field.get("value_type")),
                    "value_type": field.get("value_type", ""),
                    "confidence": field.get("confidence", ""),
                    "decision": field.get("decision", ""),
                    "page": grounding.get("page", ""),
                    "bbox": json.dumps(grounding.get("bbox")) if grounding.get("bbox") else "",
                    "extractor_id": field.get("extractor_id", ""),
                    "review_reasons": "|".join(field.get("review_reasons") or []),
                    "review_actions": "|".join(field_actions(field, actions)),
                    "value_rewrite": value_rewrite_note(field),
                    "chain_evidence": field.get("chain", ""),
                })


def run(input_dir: Path, out_dir: Path, config: Optional[Dict] = None,
        force: bool = False) -> Dict:
    config = config or load_config()
    started_at = now_iso()
    work_parsed = resolve_path(config["paths"]["work"], config) / "parsed"
    documents = _collect_documents(input_dir, config, out_dir)
    if not documents:
        raise SystemExit(f"{input_dir} 下没有找到可导入的资料（支持 PDF 与图片）")

    # 一份一份串行跑：并发实测在这条管线上没有收益（见 README「批量导入的并发」），
    # OCR 会话自己已经把核心用满，多开只会互相抢内存带宽。
    extracted_docs, parsed_by_id, sources, timings = [], {}, {}, {}
    failures = []
    for doc_id, source in documents:
        started = time.perf_counter()
        try:
            parsed = parse_document(source, doc_id,
                                    dpi=int(config["parsing"].get("render_dpi", 300)),
                                    force=force, config=config, out_dir=work_parsed)
            parsed_at = time.perf_counter()
            extracted = extract_document(parsed, config)
        except (OSError, ValueError, RuntimeError) as error:
            failures.append({"document_id": doc_id, "file_name": source.name,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        finished = time.perf_counter()
        timings[doc_id] = {
            "parse_seconds": parsed_at - started,
            "extract_seconds": finished - parsed_at,
        }
        extracted["source_path"] = str(source.resolve())
        extracted_docs.append(extracted)
        parsed_by_id[doc_id] = parsed
        sources[doc_id] = source

    assess(extracted_docs, parsed_by_id, config, cross_document=len(extracted_docs) > 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    payloads = []
    for extracted in extracted_docs:
        doc_id = extracted["document_id"]
        payload = _document_payload(extracted, parsed_by_id[doc_id], sources[doc_id],
                                    timings.get(doc_id), config)
        payload["review_summary"] = summarize_reviews(
            payload["fields"], read_review_records(out_dir, doc_id), config)
        document_dir = out_dir / doc_id
        document_dir.mkdir(parents=True, exist_ok=True)
        (document_dir / f"{doc_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False,
                       indent=int(config.get("output", {}).get("json_indent", 2))),
            encoding="utf-8")
        review_file = out_dir / "review" / f"{doc_id}.review.jsonl"
        if not review_file.exists():
            review_file.write_text("", encoding="utf-8")
        payloads.append(payload)

    summary = _write_summary(out_dir, payloads, input_dir, started_at, config,
                             calibrated=Calibrator(config).ready, failures=failures)
    _write_csv(out_dir, payloads, config)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the document review pipeline")
    parser.add_argument("--input", default=str(ROOT / "samples" / "dev"))
    parser.add_argument("--out", default=str(ROOT / "out"))
    parser.add_argument("--force", action="store_true", help="ignore parse cache")
    args = parser.parse_args(argv)
    summary = run(Path(args.input), Path(args.out), force=args.force)
    overall = summary["overall"]
    print(json.dumps({"overall": overall, "documents": summary["run"]["document_count"]},
                     ensure_ascii=False, indent=2))
    print("written:", Path(args.out))
    if summary["failures"]:
        print(json.dumps({"failures": summary["failures"]}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
