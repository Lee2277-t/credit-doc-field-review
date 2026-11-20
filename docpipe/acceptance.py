"""One-command acceptance check.

Validates the output contract, the Phase 4 gates and the review -> export loop
on a temporary copy of the output directory. The real out/ is never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from urllib.parse import quote
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .calibrate import (_class_balance, _require_two_classes, _stored_documents,
                        build_samples, merge_review_samples, review_sample_dirs,
                        review_samples, train)
from .config import ROOT, load_config, resolve_path
from .evaluate import claim_targets, evaluate_predictions
from .export import export, export_zip
from .review_actions import (chain_note, field_actions, review_action_table, value_rewrite_note)
from .serve import make_server
from .textutil import now_iso
from .trust import assess, field_features

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CSV_COLUMNS = ["document_id", "doc_type", "field_path", "value", "value_type",
               "confidence", "decision", "page", "bbox", "extractor_id", "review_reasons",
               "review_actions", "value_rewrite", "chain_evidence"]
REQUIRED_FIELD_KEYS = {"value", "value_type", "confidence", "confidence_basis", "decision",
                       "grounding", "extractor_id", "validators", "review_reasons"}
CONFIDENCE_BASES = {"calibrated", "calibrated_low_samples", "heuristic"}


class Report:
    def __init__(self):
        self.checks: List[Dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(item["ok"] for item in self.checks)


# 验收里所有本机 HTTP 请求的超时（秒）。/api/import/run 会当场跑完整流水线，
# 冷启动实测约 7.5 秒；写死 10 秒会和前面几项检查抢 CPU，所以这个值由
# config.acceptance.http_timeout_seconds 决定（run_acceptance 在读到配置后覆盖）。
HTTP_TIMEOUT_SECONDS = 30.0


def _get(port: int, path: str):
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def _post(port: int, path: str, payload: Dict):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _wait_ready(port: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, _, _ = _get(port, "/api/summary")
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.1)
    return False


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _check_contract(report: Report, out_dir: Path, labels_dir: Path) -> Optional[Dict]:
    summary_path = out_dir / "summary.json"
    if not report.check("output.summary_exists", summary_path.exists(), str(summary_path)):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report.check("output.summary_keys",
                 {"run", "overall", "by_doc_type", "by_field_path", "documents"} <= set(summary),
                 ",".join(summary.keys()))

    header = None
    csv_path = out_dir / "fields.csv"
    if report.check("output.csv_exists", csv_path.exists(), str(csv_path)):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
        report.check("output.csv_header", header == CSV_COLUMNS, ",".join(header or []))

    documents = summary.get("documents") or []
    label_count = len(list(labels_dir.glob("*.json")))
    report.check("output.document_count", len(documents) == label_count,
                 f"documents={len(documents)} labels={label_count}")

    accepted_bad_grounding = 0
    ordinal_fields = 0
    schema_detail = ""
    for item in documents:
        doc_path = out_dir / item["path"]
        if not report.check(f"output.document.{item['document_id']}", doc_path.exists(), str(doc_path)):
            continue
        document = json.loads(doc_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "1.0":
            schema_detail += f"{item['document_id']}:schema;"
        for path, field in document.get("fields", {}).items():
            missing = REQUIRED_FIELD_KEYS - set(field)
            if missing:
                schema_detail += f"{item['document_id']}:{path}:{','.join(sorted(missing))};"
            if field.get("decision") not in {"accept", "review"}:
                schema_detail += f"{item['document_id']}:{path}:decision;"
            if field.get("confidence_basis") not in CONFIDENCE_BASES:
                schema_detail += f"{item['document_id']}:{path}:confidence_basis;"
            if field.get("confidence_basis") != "calibrated":
                ordinal_fields += 1
            confidence = field.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                schema_detail += f"{item['document_id']}:{path}:confidence;"
            if field.get("decision") == "accept":
                grounding = field.get("grounding") or {}
                bbox = grounding.get("bbox") or []
                if grounding.get("page") is None or len(bbox) != 4 \
                        or bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                    accepted_bad_grounding += 1
    report.check("output.field_schema", not schema_detail, schema_detail[:500])
    report.check("output.accepted_grounding", accepted_bad_grounding == 0,
                 f"accepted_without_grounding={accepted_bad_grounding}")
    calibration = summary.get("calibration") or {}
    report.check("output.confidence_honesty",
                 bool(calibration.get("by_confidence_basis"))
                 and bool(calibration.get("note"))
                 and calibration.get("ordinal_count") == ordinal_fields
                 and (calibration.get("calibrated")
                      or calibration.get("ordinal_share") == 1.0),
                 f"basis={calibration.get('by_confidence_basis')} calibrated={calibration.get('calibrated')}")
    return summary


def _check_gates(report: Report, evaluation: Dict, config: Dict) -> None:
    gates = claim_targets(config)
    overall = evaluation["summary"]
    accept_rate = float(overall.get("auto_accept_rate") or 0.0)
    error_rate = float(overall.get("accepted_error_rate") or 0.0)
    monetary_error = float(overall.get("monetary_accepted_error_rate") or 0.0)
    grounding_missing = int(overall.get("accepted_grounding_missing") or 0)
    grounding_hit = overall.get("grounding_hit_rate")
    report.check("gate.auto_accept_rate",
                 accept_rate >= gates["min_auto_accept_rate"],
                 f"{accept_rate:.4f} >= {gates['min_auto_accept_rate']}")
    report.check("gate.accepted_error_rate",
                 error_rate <= gates["max_accepted_error_rate"],
                 f"{error_rate:.4f} <= {gates['max_accepted_error_rate']}")
    report.check("gate.monetary_error_rate",
                 monetary_error <= gates["max_monetary_accepted_error_rate"],
                 f"{monetary_error:.4f} <= {gates['max_monetary_accepted_error_rate']}")
    report.check("gate.accepted_grounding_missing",
                 grounding_missing <= gates["max_accepted_grounding_missing"],
                 f"{grounding_missing} <= {gates['max_accepted_grounding_missing']}")
    report.check("gate.grounding_hit_rate",
                 grounding_hit is not None
                 and float(grounding_hit) >= gates["min_grounding_hit_rate"],
                 f"{grounding_hit} >= {gates['min_grounding_hit_rate']}")

    transactions = overall.get("transactions")
    if transactions:
        floor = float((config.get("acceptance") or {}).get(
            "min_transaction_content_match_rate", 0.98))
        report.check("gate.transaction_content_match",
                     float(transactions.get("content_match_rate") or 0.0) >= floor,
                     f"{transactions.get('content_match_rate')} >= {floor}")

    # SCENARIO §6.5：样本不足只能算暂定结论。以前这条在 n>=100 时整条不出现（看不出守没守），
    # 而 100 硬写在代码里。现在判决跟着指标走，检查也恒定发出。
    claim = overall.get("claim") or {}
    report.check("gate.sample_size", claim.get("status") == "met",
                 f"status={claim.get('status')} accepted={claim.get('accept_count')} "
                 f"floor={claim.get('min_accept_for_claim')} "
                 f"confidence={claim.get('confidence')} "
                 f"error_ci={overall.get('accepted_error_ci')}")
    # 判决在两个层级都要守：按类型分组是给人看达标没有，按风险分组是这条规则原来最容易漏的地方
    # （金额那一维在纯文本分组里既不是成绩、也不该把整组拖成「样本不够」）。
    groups = [("by_doc_type", row) for row in overall.get("by_doc_type") or []]
    groups += [("by_risk", row) for row in overall.get("by_risk") or []]
    short_groups = [f"{level}:{row['name']}" for level, row in groups
                    if int(row.get("accept_count") or 0) < gates["min_accept_for_claim"]
                    and (row.get("claim") or {}).get("status") == "met"]
    report.check("gate.small_group_not_claimed", not short_groups,
                 f"分母不足却自称 met 的分组={short_groups}（每个指标按自己的分母判，不看总样本）")
    missing = [f"{level}:{row['name']}" for level, row in groups
               if not (row.get("claim") or {}).get("status")]
    report.check("gate.claim_travels_with_stats", not missing,
                 f"没有判决的分组={missing}；判决必须随每一级指标一起发出")


def _check_review_flow(report: Report, out_dir: Path, config: Dict) -> Optional[Dict]:
    gates = config.get("acceptance") or {}
    if not gates.get("require_review_flow", True):
        return None
    work = resolve_path(config["paths"]["work"], config)
    work.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_accept_", dir=str(work)))
    temp_out = temp_root / "out"
    server = None
    thread = None
    result = None
    try:
        shutil.copytree(out_dir, temp_out, ignore=shutil.ignore_patterns("final", "render"))
        summary = json.loads((temp_out / "summary.json").read_text(encoding="utf-8"))
        documents = summary.get("documents") or []
        if not report.check("review.documents", bool(documents), f"documents={len(documents)}"):
            return None
        entry = next((item for item in documents if item.get("review_count")), documents[0])
        document = json.loads((temp_out / entry["path"]).read_text(encoding="utf-8"))
        fields = document["fields"]
        field = next((item for item in fields.values() if item["decision"] == "review"),
                     next(iter(fields.values())))
        field_path = field["path"]
        doc_id = document["document_id"]

        server = make_server(temp_out, "127.0.0.1", 0, config)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        if not report.check("review.server_ready", _wait_ready(port), f"port={port}"):
            return None

        status, content_type, body = _get(port, "/")
        report.check("review.index", status == 200 and "text/html" in content_type
                     and "信贷资料字段复核台".encode("utf-8") in body, f"status={status}")
        status, _, body = _get(port, "/static/app.js")
        report.check("review.static", status == 200 and b"function openDocument" in body,
                     f"status={status}")
        status, _, body = _get(port, f"/api/document/{doc_id}")
        report.check("review.document_api", status == 200 and doc_id.encode() in body,
                     f"status={status}")
        status, content_type, body = _get(port, f"/api/page/{doc_id}/1.png?dpi=150")
        report.check("review.page_render", status == 200 and body.startswith(PNG_MAGIC)
                     and "image/png" in content_type, f"status={status} bytes={len(body)}")

        jsonl = temp_out / "review" / f"{doc_id}.review.jsonl"
        before = len(_read_jsonl(jsonl))
        status, payload = _post(port, f"/api/review/{doc_id}",
                                {"field_path": field_path, "action": "accept",
                                 "reviewer": "acceptance"})
        records = _read_jsonl(jsonl)
        report.check("review.accept_api",
                     status == 200 and payload.get("ok") and len(records) == before + 1
                     and records[-1]["action"] == "accept",
                     f"status={status} records={len(records)}")
        report.check("review.accept_contract",
                     bool(records) and {"document_id", "field_path", "original_value",
                                        "corrected_value", "action", "reviewer",
                                        "timestamp"} <= set(records[-1]),
                     ",".join(records[-1].keys()) if records else "no record")

        marker = "__ACCEPTANCE_TEST__"
        status, payload = _post(port, f"/api/review/{doc_id}",
                                {"field_path": field_path, "action": "correct",
                                 "corrected_value": marker, "reviewer": "acceptance"})
        records = _read_jsonl(jsonl)
        report.check("review.correct_api",
                     status == 200 and len(records) == before + 2
                     and records[-1]["corrected_value"] == marker
                     and records[-1]["action"] == "correct",
                     f"status={status} records={len(records)}")
        status, _, body = _get(port, f"/api/document/{doc_id}")
        loaded = json.loads(body.decode("utf-8"))
        report.check("review.review_roundtrip",
                     status == 200 and len(loaded.get("reviews") or []) == before + 2,
                     f"status={status} reviews={len(loaded.get('reviews') or [])}")

        server.shutdown()
        server.server_close()
        server = None
        if thread:
            thread.join(timeout=3)
            thread = None

        export_summary = export(temp_out, temp_out / "final", config)
        final_doc_path = temp_out / "final" / doc_id / f"{doc_id}.json"
        final_doc = json.loads(final_doc_path.read_text(encoding="utf-8"))
        final_field = final_doc["fields"][field_path]
        report.check("export.effective_value",
                     final_field.get("effective_value") == marker
                     and final_field.get("review_status") == "reviewed_corrected",
                     f"status={final_field.get('review_status')}")
        report.check("export.summary", export_summary["overall"]["reviewed_corrected_count"] >= 1,
                     json.dumps(export_summary["overall"], ensure_ascii=False))
        with (temp_out / "final" / "final_fields.csv").open(
                "r", encoding="utf-8-sig", newline="") as handle:
            final_header = next(csv.reader(handle), None)
        report.check("export.csv", final_header is not None
                     and {"effective_value", "review_status", "confidence_basis"} <= set(final_header),
                     ",".join(final_header or []))
        report.check("review_contract.final_csv_column",
                     "review_actions" in (final_header or []),
                     ",".join(final_header or []))
        with (temp_out / "final" / "final_fields.csv").open(
                "r", encoding="utf-8-sig", newline="") as handle:
            final_rows = list(csv.DictReader(handle))
        _check_action_rows(report, "final", final_rows, review_action_table(config))
        final_summary = json.loads((temp_out / "final" / "final_summary.json")
                                   .read_text(encoding="utf-8"))
        _check_action_summary(report, "final", final_summary)
        calibration = final_summary.get("calibration") or {}
        zip_config = copy.deepcopy(config)
        zip_config["paths"]["work"] = str(temp_root / "work")
        packaged = export_zip(temp_out, temp_out / "final", zip_config)
        with zipfile.ZipFile(packaged["path"]) as archive:
            names = archive.namelist()
            report_name = zip_config.get("report", {}).get("file_name", "查看结果.html")
            group_title = (zip_config.get("report", {}).get("group_order") or [""])[0]
            html = archive.read(report_name).decode("utf-8") if report_name in names else ""
            guide = archive.read("怎么看.txt").decode("utf-8") if "怎么看.txt" in names else ""
        # 置信度能不能当概率读，必须在交付件里写清楚，而不是只留在 out/summary.json。
        report.check("export.calibration_note",
                     bool(calibration.get("note")) and bool(calibration.get("by_confidence_basis"))
                     and "置信度口径" in guide and calibration["note"] in guide
                     and calibration["note"] in html,
                     f"note={(calibration.get('note') or '')[:30]} basis={calibration.get('by_confidence_basis')}")
        report.check("export.zip",
                     packaged["documents"] >= 1
                     and any(name.startswith("final/") for name in names)
                     and any(name.startswith("review/") and name.endswith(".review.jsonl")
                             for name in names)
                     and "export_manifest.json" in names,
                     f"entries={len(names)} size={packaged['size']}")
        report.check("export.report_offline_html",
                     report_name in names and len(html) > 20_000
                     and 'src="http' not in html and 'href="http' not in html
                     and "<link" not in html and group_title in html
                     and "自动接受" in html and "未复核" not in html
                     and report_name in guide,
                     f"html={len(html)} guide={len(guide)}")
        _check_report_actions(report, final_rows, html)
        result = export_summary
        # 铁律审计 X：这棵树刚被复核台写过台账、被 export() 应用过修正，是逐格对账的真夹具。
        _check_review_reconcile(report, temp_out, config)
    except Exception as exc:  # acceptance must report failures, not crash
        report.check("review.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=3)
        shutil.rmtree(temp_root, ignore_errors=True)
    return result


def _check_image_ingest(report: Report, config: Dict) -> None:
    """End-to-end: an image upload is typed by its bytes, OCR-parsed and reported."""
    gates = config.get("acceptance") or {}
    if not gates.get("require_image_ingest", True):
        return

    import pymupdf as fitz

    from .importer import process_inbox, store_upload
    from .report import build_report

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_image_", dir=str(work)))
    try:
        cfg = copy.deepcopy(config)
        cfg["paths"]["work"] = str(temp_root / "work")
        cfg["paths"]["inbox"] = str(temp_root / "inbox")
        cfg["paths"]["user_out"] = str(temp_root / "out")
        samples = sorted((resolve_path(cfg["paths"]["samples"], cfg) / "dev").glob("*.pdf"))
        if not samples:
            report.check("ingest.sample_image", False, "no sample PDF to rasterise")
            return
        page = fitz.open(samples[0])[0]
        # 200dpi：栅格本身要「够用」，这组检查考的是图片导入与溯源，
        # 低分辨率页由 page_quality 门单独负责（见 _check_page_quality_gate）。
        png = page.get_pixmap(dpi=200).tobytes("png")
        jpeg = fitz.open(samples[1] if len(samples) > 1 else samples[0])[0] \
            .get_pixmap(dpi=200).tobytes("jpeg", jpg_quality=80)

        first = store_upload(io.BytesIO(png), "证件照片.pdf", len(png), cfg)
        second = store_upload(io.BytesIO(jpeg), "流水拍照.jpg", len(jpeg), cfg)
        # 扫描仪常见的多页 TIFF：同一份资料的连续两页必须是两页，不能只留第一页。
        from PIL import Image
        multipage = next((fitz.open(item) for item in samples
                          if fitz.open(item).page_count > 1), None)
        if multipage is None or multipage.page_count < 2:
            report.check("ingest.multiframe_sample", False, "no multi-page sample to rasterise")
            return
        frames = [Image.open(io.BytesIO(multipage[index].get_pixmap(dpi=200).tobytes("png")))
                  .convert("RGB") for index in (0, 1)]
        tiff_buffer = io.BytesIO()
        frames[0].save(tiff_buffer, save_all=True, append_images=frames[1:],
                       format="TIFF", compression="jpeg")
        tiff_bytes = tiff_buffer.getvalue()
        third = store_upload(io.BytesIO(tiff_bytes), "两页扫描件.tif", len(tiff_bytes), cfg)
        inbox = Path(cfg["paths"]["inbox"])
        report.check("ingest.image_signature",
                     first["suffix"] == "png" and first["name"] == "证件照片.png"
                     and second["suffix"] == "jpg" and third["suffix"] == "tif"
                     and (inbox / "证件照片.png").is_file()
                     and (inbox / "流水拍照.jpg").is_file()
                     and (inbox / "两页扫描件.tif").is_file(),
                     f"first={first['name']} second={second['name']} third={third['name']}")
        rejected = None
        try:
            store_upload(io.BytesIO("这是一段纯文本".encode("utf-8")), "说明.txt", 18, cfg)
        except ValueError as exc:
            rejected = str(exc)
        report.check("ingest.unknown_type_rejected", bool(rejected), (rejected or "")[:120])

        result = process_inbox(cfg)
        summary = result["summary"]
        documents = summary.get("documents") or []
        page_counts = sorted(int(item.get("page_count") or 0) for item in documents)
        accepted, orphan = [], 0
        for item in documents:
            fields = json.loads((Path(result["out_dir"]) / item["path"])
                                .read_text(encoding="utf-8"))["fields"]
            pages = int(item.get("page_count") or 1)
            for field in fields.values():
                if field.get("decision") != "accept":
                    continue
                grounding = field.get("grounding") or {}
                accepted.append(field)
                if not grounding.get("bbox") or not (
                        1 <= int(grounding.get("page") or 0) <= pages):
                    orphan += 1
        report.check("ingest.image_pipeline",
                     len(documents) == 3 and summary["overall"]["field_count"] > 0
                     and page_counts == [1, 1, 2]
                     and all(json.loads((Path(result["out_dir"]) / item["path"])
                                         .read_text(encoding="utf-8"))["fields"]
                             for item in documents),
                     f"documents={len(documents)} pages={page_counts} "
                     f"fields={summary['overall']['field_count']}")
        report.check("ingest.image_grounding",
                     bool(accepted) and orphan == 0,
                     f"accepted={len(accepted)} outside_page={orphan}")

        built = build_report(Path(result["out_dir"]), dest=Path(result["out_dir"]) / "final",
                             config=cfg)
        html = Path(built["path"]).read_text(encoding="utf-8")
        report.check("ingest.image_report_pages",
                     built["embedded_pages"] == sum(min(count, int(
                         cfg["report"].get("max_pages_per_document", 8) or 8))
                         for count in page_counts)
                     and built["page_level"] == 0
                     and html.count("data:image/jpeg;base64,") == built["embedded_pages"],
                     f"pages={built['embedded_pages']} size={built['size']}")
    except Exception as exc:
        report.check("ingest.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_photo_preprocess(report: Report, config: Dict) -> None:
    """Crooked, stamped, softly focused, glare-washed and finger-covered photos must be handled, then said out loud."""
    gates = config.get("acceptance") or {}
    if not gates.get("require_photo_preprocess", True):
        return

    import pymupdf as fitz
    from PIL import Image, ImageFilter

    from .importer import process_inbox, store_upload
    from .preprocess import FLAG_LABELS, enhance_options
    from .report import build_report

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_photo_", dir=str(work)))
    try:
        cfg = copy.deepcopy(config)
        cfg["paths"]["work"] = str(temp_root / "work")
        cfg["paths"]["inbox"] = str(temp_root / "inbox")
        cfg["paths"]["user_out"] = str(temp_root / "out")
        samples = sorted((resolve_path(cfg["paths"]["samples"], cfg) / "dev").glob("*.pdf"))
        if not report.check("photo.sample_available", bool(samples), f"samples={len(samples)}"):
            return
        canvas = Image.open(io.BytesIO(
            fitz.open(samples[0])[0].get_pixmap(dpi=200).tobytes("png"))).convert("RGB")

        # 骑缝章专用底座：章只有半边/一角入画时整页彩色占比在主门之下，只能靠「贴图片边缘」
        # 认出来，所以底座必须是一张本身无彩色的页面（证件页自带红章会把占比顶过主门）。
        options = enhance_options(cfg)
        edge_canvas = None
        for sample in samples:
            candidate = Image.open(io.BytesIO(
                fitz.open(sample)[0].get_pixmap(dpi=200).tobytes("png"))).convert("RGB")
            if _colored_ratio(candidate, int(options["seal_chroma_threshold"])) \
                    < float(options["edge_seal_min_colored_ratio"]):
                edge_canvas = candidate
                break

        tilt = float(gates.get("photo_skew_degrees", 7))
        shots = {
            "歪斜拍照.png": canvas.rotate(tilt, expand=True, fillcolor=(255, 255, 255)),
            "盖章拍照.png": _draw_seal(canvas),
            "模糊拍照.png": canvas.filter(ImageFilter.GaussianBlur(
                float(gates.get("photo_blur_radius", 6)))),
            "光照不均.png": _uneven_lighting(canvas),
            "拍暗拍照.png": _dark_photo(canvas, float(gates.get("photo_dark_scale", 0.40)),
                                        float(gates.get("photo_dark_offset", 20))),
            "反光拍照.png": _glare_photo(canvas),
            "遮挡拍照.png": _finger_photo(canvas),
            "骑缝章拍照.png": _draw_edge_seal(edge_canvas or canvas),
        }
        blobs = {name: _png_bytes(image) for name, image in shots.items()}
        for name, blob in blobs.items():
            store_upload(io.BytesIO(blob), name, len(blob), cfg)

        result = process_inbox(cfg)
        out_dir = Path(result["out_dir"])
        parsed_dir = Path(cfg["paths"]["work"]) / "parsed"
        documents = {item["document_id"]: json.loads(
            (out_dir / item["path"]).read_text(encoding="utf-8"))
            for item in result["summary"].get("documents") or []}
        parsed = {doc_id: json.loads((parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
                  for doc_id in documents}
        source_of = {name: doc_id for doc_id, doc in documents.items()
                     if (name := doc["source"]["file_name"]) in shots}

        def first_page(doc_id: str, key: str) -> Dict:
            pages = (parsed.get(doc_id) or {}).get("pages") or []
            return (pages[0].get("preprocess") or {}).get(key) or {} if pages else {}

        tilted = source_of.get("歪斜拍照.png")
        degrees = float(first_page(tilted, "geometry").get("deskew_degrees") or 0.0)
        report.check("photo.deskew_applied",
                     abs(abs(degrees) - tilt) <= float(gates.get("photo_deskew_tolerance", 1.5))
                     and (documents.get(tilted) or {}).get(
                         "document_quality", {}).get("rectified_pages", 0) >= 1,
                     f"deskew={degrees}° shot={tilt}°")

        sealed = source_of.get("盖章拍照.png")
        enhancement = first_page(sealed, "enhancement")
        report.check("photo.seal_suppressed_for_ocr",
                     bool(enhancement.get("seal_suppressed")) and bool(enhancement.get("seal_regions")),
                     f"regions={len(enhancement.get('seal_regions') or [])}")
        # 章压住哪几个字是人眼判断，程序只该把章的位置递到复核员眼前：
        # 出参里必须有归一化框，界面必须真去读它，否则重命名一个键就静默丢掉这条提示。
        payload_regions = [region
                           for page in ((documents.get(sealed) or {}).get("document_quality") or {}
                                        ).get("preprocess") or []
                           for region in page.get("seal_regions") or []]
        app_source = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
        style_source = (ROOT / "docpipe" / "web" / "style.css").read_text(encoding="utf-8")
        report.check("photo.seal_regions_reach_review_ui",
                     bool(payload_regions)
                     and all(len(region) == 4
                             and all(0.0 <= float(value) <= 1.0 for value in region)
                             for region in payload_regions)
                     and "seal_regions" in app_source and "seal-box" in app_source
                     and ".seal-box" in style_source,
                     f"regions={len(payload_regions)} wired="
                     f"{'seal_regions' in app_source and 'seal-box' in app_source}")

        # 骑缝章：整页占比低于主门，只有「色块贴图片边缘」这一条能认出来；
        # 认出来之后同样要把归一化章框递给复核台，旗标要把这条弱证据和整枚章区分开。
        fragment = source_of.get("骑缝章拍照.png")
        fragment_enhancement = first_page(fragment, "enhancement")
        fragment_regions = fragment_enhancement.get("seal_regions") or []
        fragment_pages = [page for page in ((documents.get(fragment) or {})
                                            .get("document_quality") or {}).get("preprocess") or []]

        def touches_edge(region: List[float]) -> bool:
            return region[0] <= 0.01 or region[1] <= 0.01 or region[2] >= 0.99 or region[3] >= 0.99

        report.check("photo.edge_clipped_seal",
                     bool(fragment_enhancement.get("seal_suppressed"))
                     and bool(fragment_enhancement.get("seal_edge_fragment"))
                     and bool(fragment_regions)
                     and all(touches_edge(region) for region in fragment_regions)
                     and float(fragment_enhancement.get("colored_ratio") or 0.0)
                     < float(options["min_colored_ratio"]),
                     f"fragment={bool(fragment_enhancement.get('seal_edge_fragment'))} "
                     f"ratio={fragment_enhancement.get('colored_ratio')} "
                     f"regions={len(fragment_regions)}")
        # 旗标只停在解析记录里等于没说：复核员看见页边一个红框，分不清是骑缝残片还是整枚章，
        # 就会拿「章没盖全」去判这页有问题。要求旗标走到出参，并且界面真的读它、画它。
        sealed_pages = [page for page in ((documents.get(sealed) or {})
                                          .get("document_quality") or {}).get("preprocess") or []]
        sealed_fragment = any(page.get("seal_edge_fragment") for page in sealed_pages)
        report.check("photo.edge_fragment_reaches_review_ui",
                     bool(fragment_pages)
                     and any(page.get("seal_edge_fragment") and page.get("seal_regions")
                             for page in fragment_pages)
                     and "seal_edge_fragment" in app_source
                     and "dataset.label" in app_source and "[data-label]" in style_source
                     and not sealed_fragment,
                     f"pages={len(fragment_pages)} wired={'seal_edge_fragment' in app_source} "
                     f"sealed_fragment={sealed_fragment}")

        blurred = source_of.get("模糊拍照.png")
        flags = (parsed.get(blurred) or {}).get("quality", {}).get("photo_flags") or []
        wanted = set(gates.get("photo_expected_flags") or ["PHOTO_BLUR"])
        report.check("photo.low_quality_flagged", bool(wanted & set(flags)),
                     f"flags={','.join(flags) or 'none'}")

        glare = source_of.get("反光拍照.png")
        control = source_of.get("光照不均.png")
        glare_enhancement = first_page(glare, "enhancement")
        report.check("photo.glare_suppressed_for_ocr",
                     bool(glare_enhancement.get("highlight_suppressed"))
                     and float(glare_enhancement.get("highlight_gain") or 0.0)
                     >= float(options["highlight_min_gain"])
                     and not bool(first_page(control, "enhancement").get("highlight_suppressed")),
                     f"gain={glare_enhancement.get('highlight_gain')} "
                     f"area={glare_enhancement.get('highlight_area')}")

        # 影调改写是**整页**改写（对比度拉伸还会把彩色并成灰度）：送去 OCR 的那一份和复核员
        # 在页面上看到的那一份已经不是同一张图。旗标必须一路走到复核台与报表，否则这个改写
        # 只有写代码的人知道，人只会以为原件本来就发淡。
        dark = source_of.get("拍暗拍照.png")
        dark_enhancement = first_page(dark, "enhancement")
        dark_pages = [page for page in (documents.get(dark) or {}).get(
            "document_quality", {}).get("preprocess") or []]
        bright_pages = [page for page in (documents.get(tilted) or {}).get(
            "document_quality", {}).get("preprocess") or []]
        tone_lifted = any(page.get("brightness_lifted") and page.get("contrast_lifted")
                          for page in dark_pages)
        control_clean = not any(page.get("brightness_lifted") or page.get("contrast_lifted")
                                for page in bright_pages)
        tone_wired = ("contrast_lifted" in app_source and "brightness_lifted" in app_source
                      and "对比度拉伸" in app_source and "暗页提亮" in app_source)
        report.check("photo.tone_rewrite_reaches_review_ui",
                     bool(dark_enhancement.get("brightness_lifted"))
                     and bool(dark_enhancement.get("contrast_lifted"))
                     and tone_lifted and control_clean and tone_wired,
                     f"bright={dark_enhancement.get('brightness_lifted')} "
                     f"contrast={dark_enhancement.get('contrast_lifted')} "
                     f"payload={tone_lifted} control_clean={control_clean} wired={tone_wired}")

        def field_count(doc_id: str) -> int:
            return len((documents.get(doc_id) or {}).get("fields") or {})

        share = float(gates.get("photo_glare_field_share", 0.8))
        report.check("photo.glare_still_extractable",
                     field_count(control) > 0 and field_count(glare) >= share * field_count(control),
                     f"glare={field_count(glare)} control={field_count(control)} share={share}")

        covered = source_of.get("遮挡拍照.png")
        occlusion = (parsed.get(covered) or {}).get("quality", {}).get("photo_flags") or []
        report.check("photo.occlusion_flagged", "PHOTO_OCCLUSION" in occlusion,
                     f"flags={','.join(occlusion) or 'none'}")

        report.check("photo.status_reach_output",
                     bool((documents.get(blurred) or {}).get("document_quality", {})
                          .get("photo_flags"))
                     and bool((documents.get(covered) or {}).get("document_quality", {})
                              .get("photo_flags"))
                     and any(page.get("deskew_degrees")
                             for page in (documents.get(tilted) or {}).get(
                                 "document_quality", {}).get("preprocess") or []),
                     f"flags={(documents.get(blurred) or {}).get('document_quality', {}).get('photo_flags')}")

        built = build_report(out_dir, dest=out_dir / "final", config=cfg)
        html = Path(built["path"]).read_text(encoding="utf-8")
        embedded = re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                             html, re.S)
        data = json.loads(embedded.group(1).replace("<\\/", "</")) if embedded else {}
        report_docs = {item["id"]: item for item in data.get("documents") or []}
        blurred_entry = report_docs.get(source_of.get("模糊拍照.png") or "", {})
        tilted_entry = report_docs.get(tilted or "", {})
        glare_entry = report_docs.get(glare or "", {})
        dark_entry = report_docs.get(dark or "", {})
        tone_in_report = any(page.get("contrast_lifted") and page.get("brightness_lifted")
                             for page in dark_entry.get("preprocess") or [])
        report.check("photo.status_reach_report",
                     bool(set(blurred_entry.get("photo_flags") or []) & wanted)
                     and any(page.get("deskew_degrees") for page in tilted_entry.get("preprocess") or [])
                     and FLAG_LABELS[sorted(wanted & set(blurred_entry.get("photo_flags") or []))[0]] in html
                     and ("拉正" in html or "拉平" in html) and "淡化" in html
                     and any(page.get("highlight_suppressed") for page in glare_entry.get("preprocess") or [])
                     and FLAG_LABELS["PHOTO_OCCLUSION"] in html and "反光压制" in html
                     and tone_in_report and "对比度拉伸" in html and "暗页提亮" in html,
                     f"flags={blurred_entry.get('photo_flags')} embedded={len(report_docs)} "
                     f"tone_in_report={tone_in_report}")
    except Exception as exc:
        report.check("photo.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_table_anomaly(report: Report, config: Dict) -> None:
    """A merged/over-wide cell must survive parse -> extract -> decide as review."""
    gates = config.get("acceptance") or {}
    if not gates.get("require_table_anomaly", True):
        return

    import random
    from datetime import date
    from decimal import Decimal

    import pymupdf as fitz

    from . import synth
    from .extract import extract_document
    from .parse import parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_table_", dir=str(work)))
    try:
        company = synth.make_company(random.Random(11), 0)
        txns = synth.make_bank_transactions(random.Random(12), date(2025, 7, 1),
                                            date(2025, 12, 31), Decimal("100000.00"), 5)
        data = {
            "bank_name": "示例银行",
            "account_number": "6222020200112233445",
            "period_start": date(2025, 7, 1),
            "period_end": date(2025, 12, 31),
            "opening_balance": Decimal("100000.00"),
            "transactions": txns,
        }
        source = temp_root / "table_anomaly.pdf"
        doc = fitz.open()
        synth.build_bank_statement_fixture(doc, company, data, extra="stray_note")
        doc.save(source, garbage=4, deflate=True)
        doc.close()

        parsed = parse_document(source, "TBL", force=True, config=config,
                                out_dir=temp_root / "parsed")
        extracted = extract_document(parsed, config)
        assessed = assess([extracted], {"TBL": parsed}, config, cross_document=False)[0]
        flagged = assessed["fields"]["bs.transactions[3].amount"]
        anomalies = [item for page in parsed["pages"] for table in page["tables"]
                     for item in table.get("anomalies", [])]
        # 一个框没有 rows 表示"整张表都受影响"，所以洁净行只能挑异常没碰到的那一行。
        touched = {row for item in anomalies for row in (item.get("rows") or range(len(txns)))}
        clean_index = next((index for index in range(len(txns)) if index not in touched), None)
        clean = assessed["fields"].get(f"bs.transactions[{clean_index}].amount")
        report.check("table_anomaly.detected",
                     any(item["reason"] == "column_mismatch" for item in anomalies),
                     f"anomalies={[(a['reason'], a.get('rows')) for a in anomalies]}")
        report.check("table_anomaly.row_forced_review",
                     flagged.get("decision") == "review"
                     and "TABLE_ANOMALY" in (flagged.get("review_reasons") or []),
                     f"decision={flagged.get('decision')} "
                     f"reasons={flagged.get('review_reasons')}")
        report.check("table_anomaly.clean_row_untouched",
                     clean is not None and "TABLE_ANOMALY" not in (clean.get("review_reasons") or []),
                     f"clean_index={clean_index} "
                     f"reasons={(clean or {}).get('review_reasons')}")

        merges = [item for item in anomalies if item["reason"] == "column_merge"]
        merge_index = next((row for item in merges for row in (item.get("rows") or [])), None)
        merge_field = assessed["fields"].get(f"bs.transactions[{merge_index}].amount")
        report.check("table_anomaly.column_merge_detected", bool(merges),
                     f"merges={[(m['columns'], m['rows']) for m in merges]}")
        report.check("table_anomaly.column_merge_forces_review",
                     merge_field is not None
                     and merge_field.get("decision") == "review"
                     and "TABLE_ANOMALY" in (merge_field.get("review_reasons") or [])
                     and (merge_field.get("meta") or {}).get("table_anomaly_columns"),
                     f"merge_index={merge_index} decision={(merge_field or {}).get('decision')} "
                     f"columns={(merge_field or {}).get('meta', {}).get('table_anomaly_columns')}")
        # 并框开关只关并框，纵向错列必须还在，否则说明两路证据被同一个闸门绑死了。
        merge_off = copy.deepcopy(config)
        merge_off["parsing"]["table_anomaly"]["detect_column_merge"] = False
        parsed_off = parse_document(source, "TBL-OFF", force=True, config=merge_off,
                                    out_dir=temp_root / "parsed-off")
        off_anomalies = [item for page in parsed_off["pages"] for table in page["tables"]
                         for item in table.get("anomalies", [])]
        report.check("table_anomaly.column_merge_configurable",
                     not any(item["reason"] == "column_merge" for item in off_anomalies)
                     and any(item["reason"] == "column_mismatch" for item in off_anomalies),
                     f"off={[(a['reason'], a.get('rows')) for a in off_anomalies]}")

        off = copy.deepcopy(config)
        off["validation"]["review_on_table_anomaly"] = False
        assessed_off = assess([copy.deepcopy(extracted)], {"TBL": parsed}, off,
                              cross_document=False)[0]
        report.check("table_anomaly.configurable",
                     "TABLE_ANOMALY" not in
                     (assessed_off["fields"]["bs.transactions[3].amount"]
                      .get("review_reasons") or []),
                     "review_on_table_anomaly=false")

        app_js = ROOT / "docpipe" / "web" / "app.js"
        report.check("table_anomaly.ui_label",
                     app_js.exists()
                     and bool(re.search(r"TABLE_ANOMALY\s*:", app_js.read_text(encoding="utf-8")))
                     and "table_anomaly_columns" in app_js.read_text(encoding="utf-8"),
                     str(app_js))
    except Exception as exc:
        report.check("table_anomaly.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_credit_report_matrix(report: Report, config: Dict) -> None:
    """Official credit-report layout: repeated column labels under a block header.

    The only honest evidence is the header sequence, the exact repeated sub-header
    and a row with exactly that many cells; the amount unit comes from the report
    note. A misaligned table must abstain, and a missing unit must not silently
    keep the un-converted number.
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_credit_matrix", True):
        return

    import pymupdf as fitz

    from . import synth
    from .extract import extract_document
    from .parse import parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_matrix_", dir=str(work)))
    try:
        headers = [("正常类", 171.0), ("关注类", 272.2), ("不良类", 373.4), ("合计", 479.0)]
        sub_x = [145.7, 200.9, 246.9, 302.0, 348.1, 403.1, 449.1, 504.3]
        row = [("合计", 86.3), ("12", 154.8), ("920.15", 195.3), ("3", 258.2),
               ("25.10", 300.9), ("2", 359.4), ("30.25", 399.8), ("17", 458.3),
               ("975.50", 498.8)]

        def build(path: Path, with_unit: bool = True, drop_cell: bool = False) -> None:
            doc = fitz.open()
            cover = synth._new_page(doc)
            synth.draw_text(cover, 200, 60, "中国人民银行征信中心", size=12)
            synth.draw_text(cover, 200, 80, "企业信用报告", size=18, bold=True)
            synth.draw_text(cover, 60, 130, "企业名称： 虚构演示企业", size=10)
            synth.draw_text(cover, 60, 160, "报告编号： 2024020300000001", size=10)
            synth.draw_text(cover, 60, 190, "报告时间： 2024-02-03T10:05:15", size=10)
            synth.draw_text(cover, 60, 220, "统一社会信用代码： 91000000DEMA000013", size=10)
            if with_unit:
                note = synth._new_page(doc)
                synth.draw_text(note, 50, 200,
                                "12. 如无特别说明，本报告中的金额类数据项单位均为万元。",
                                size=10)
            page = synth._new_page(doc)
            synth.draw_text(page, 74.7, 560, "未结清信贷及授信信息概要", size=10)
            for text, x in headers:
                synth.draw_text(page, x, 660, text, size=10)
            labels = ["账户数", "余额"] * 4
            if drop_cell:
                labels = labels[:-1]
            for text, x in zip(labels, sub_x):
                synth.draw_text(page, x, 680, text, size=10)
            for text, x in row:
                synth.draw_text(page, x, 750, text, size=10)
            doc.save(path, garbage=4, deflate=True)
            doc.close()

        def run(source: Path, tag: str):
            parsed = parse_document(source, tag, force=True, config=config,
                                    out_dir=temp_root / f"parsed-{tag}")
            extracted = extract_document(parsed, config)
            assessed = assess([extracted], {tag: parsed}, config, cross_document=False)[0]
            return parsed, assessed

        source = temp_root / "credit_matrix.pdf"
        build(source)
        parsed, assessed = run(source, "CM1")
        report.check("credit_matrix.doc_type",
                     parsed["doc_type"] == "CREDIT_REPORT_ENTERPRISE", parsed["doc_type"])
        count = assessed["fields"]["cr.loan_account_count"]
        balance = assessed["fields"]["cr.outstanding_loan_balance"]
        report.check("credit_matrix.count_read",
                     count.get("value") == 17 and count.get("extractor_id", "").startswith(
                         "header_matrix:"),
                     f"value={count.get('value')} extractor={count.get('extractor_id')}")
        report.check("credit_matrix.amount_converted",
                     balance.get("value") == "9755000.00"
                     and (balance.get("meta") or {}).get("unit_factor") == 10000,
                     f"value={balance.get('value')} meta={balance.get('meta')}")
        bbox = (balance.get("grounding") or {}).get("bbox") or []
        report.check("credit_matrix.grounding",
                     (balance.get("grounding") or {}).get("page") == 3
                     and len(bbox) == 4 and 0 <= bbox[0] < bbox[2] <= 595
                     and 0 <= bbox[1] < bbox[3] <= 842,
                     f"grounding={balance.get('grounding')}")
        report.check("credit_matrix.forced_review",
                     count.get("decision") == "review" and balance.get("decision") == "review"
                     and "MATRIX_FALLBACK" in (balance.get("review_reasons") or []),
                     f"count={count.get('decision')} balance={balance.get('decision')} "
                     f"reasons={balance.get('review_reasons')}")

        off = copy.deepcopy(config)
        off["validation"]["review_on_matrix_fallback"] = False
        assessed_off = assess([copy.deepcopy(
            extract_document(parsed, config))], {"CM1": parsed}, off,
            cross_document=False)[0]
        report.check("credit_matrix.configurable",
                     "MATRIX_FALLBACK" not in
                     (assessed_off["fields"]["cr.outstanding_loan_balance"]
                      .get("review_reasons") or []),
                     "review_on_matrix_fallback=false")

        bare = temp_root / "credit_matrix_no_unit.pdf"
        build(bare, with_unit=False)
        _, assessed_bare = run(bare, "CM2")
        bare_balance = assessed_bare["fields"]["cr.outstanding_loan_balance"]
        report.check("credit_matrix.unit_must_be_evidenced",
                     bare_balance.get("value") == "975.50"
                     and "AMBIGUOUS_UNIT" in (bare_balance.get("review_reasons") or [])
                     and bare_balance.get("decision") == "review",
                     f"value={bare_balance.get('value')} "
                     f"reasons={bare_balance.get('review_reasons')}")

        broken = temp_root / "credit_matrix_misaligned.pdf"
        build(broken, with_unit=True, drop_cell=True)
        parsed_broken = parse_document(broken, "CM3", force=True, config=config,
                                       out_dir=temp_root / "parsed-CM3")
        fields_broken = extract_document(parsed_broken, config)["fields"]
        report.check("credit_matrix.misalignment_abstains",
                     fields_broken["cr.loan_account_count"].get("value") is None
                     and fields_broken["cr.outstanding_loan_balance"].get("value") is None,
                     f"count={fields_broken['cr.loan_account_count'].get('value')} "
                     f"balance={fields_broken['cr.outstanding_loan_balance'].get('value')}")
    except Exception as exc:
        report.check("credit_matrix.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_ocr_engine(report: Report, config: Dict) -> None:
    """`parsing.ocr_engine` 一度是个装饰键：引擎写死在 parse.py 的 import 里。

    配置驱动的口径要求这个键真能换掉后端，包括换成「什么都不读」那一档——它是
    「交付物里有多少来自 OCR」的基线，也是以后每换一次引擎唯一能定价的对照。
    写了没注册的名字必须报错，不许悄悄退回默认引擎：那样指标照常，读的是别的东西。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_ocr_engine", True):
        return

    from . import ocr as backends
    from .parse import parse_document

    configured = str((config.get("parsing") or {}).get("ocr_engine") or "")
    if not report.check("ocr_engine.default_is_registered",
                        configured in backends.BUILDERS
                        and backends.get_engine(config).name == configured,
                        f"parsing.ocr_engine={configured} available={backends.engine_names()}"):
        return

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_ocr_engine_", dir=str(work)))
    try:
        source = resolve_path(config["paths"]["samples"], config) / "dev" / "D010.scan.pdf"
        if not report.check("ocr_engine.scan_sample_available", source.exists(), str(source)):
            return
        blind = copy.deepcopy(config)
        blind["parsing"]["ocr_engine"] = "null"
        scanned = parse_document(source, "OEM1", force=True, config=config,
                                 out_dir=temp_root / "default")
        no_ocr = parse_document(source, "OEM2", force=True, config=blind,
                                out_dir=temp_root / "null")
        report.check("ocr_engine.parse_record_follows_the_config",
                     scanned["engine"] == configured and no_ocr["engine"] == "null"
                     and scanned["quality"]["ocr_used"] and not no_ocr["quality"]["ocr_used"],
                     f"{configured}: engine={scanned['engine']} ocr_used={scanned['quality']['ocr_used']} "
                     f"null: engine={no_ocr['engine']} ocr_used={no_ocr['quality']['ocr_used']}")
        report.check("ocr_engine.null_baseline_reads_nothing",
                     no_ocr["quality"]["line_count"] == 0
                     and no_ocr["quality"]["recrop_parts"] == 0
                     and all(not page["lines"] for page in no_ocr["pages"]),
                     f"lines={no_ocr['quality']['line_count']} "
                     f"recrop={no_ocr['quality']['recrop_parts']}")

        refused = ""
        broken = copy.deepcopy(config)
        broken["parsing"]["ocr_engine"] = "ppocr"
        try:
            backends.get_engine(broken)
        except ValueError as exc:
            refused = str(exc)
        report.check("ocr_engine.unknown_id_is_refused",
                     "ppocr" in refused and "rapidocr" in refused and "null" in refused,
                     f"message={refused[:80]}" if refused else "未知引擎 id 被静默退回默认引擎")

        parse_text = (ROOT / "docpipe" / "parse.py").read_text(encoding="utf-8")
        report.check("ocr_engine.parse_defers_to_the_registry",
                     "ocr.get_engine" in parse_text and '== "rapidocr"' not in parse_text
                     and "rapidocr_onnxruntime" not in parse_text,
                     "parse.py 里还写着引擎名")
        report.check("ocr_engine.registry_lists_both_backends",
                     sorted(backends.BUILDERS) == backends.engine_names()
                     and {"rapidocr", "null"} <= set(backends.BUILDERS)
                     and backends.BUILDERS["null"].is_ocr is False
                     and backends.BUILDERS["rapidocr"].is_ocr is True,
                     f"registry={sorted(backends.BUILDERS)}")
    except Exception as exc:  # noqa: BLE001 - 验收要把异常变成一条检查，不能让整批检查中断
        report.check("ocr_engine.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_ocr_recrop(report: Report, config: Dict) -> None:
    """A degraded scan hides an isolated digit from full-page OCR.

    D010's page renders 「未结清贷款笔数：4」 but the detector drops the lone 4 at
    300dpi; cropping the blank right of the label reads it. The recovered value is
    still only evidence, so it must reach the reviewer instead of auto-accepting.
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_ocr_recrop", True):
        return

    from .extract import extract_document
    from .parse import _crop_ocr, gap_recrop_rect, open_document, parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_recrop_", dir=str(work)))
    try:
        source = resolve_path(config["paths"]["samples"], config) / "dev" / "D010.scan.pdf"
        if not report.check("ocr_recrop.sample_available", source.exists(), str(source)):
            return
        parsed = parse_document(source, "ORC1", force=True, config=config,
                                out_dir=temp_root / "parsed")
        first_page = parsed["pages"][0]
        label_part = next(
            (part for line in first_page["lines"] for part in line.get("parts") or []
             if "未结清贷款笔数" in str(part.get("text") or "")
             and str(part.get("text") or "").rstrip().endswith(("：", ":"))
             and part.get("engine") == "rapidocr"), None)
        extracted = extract_document(parsed, config)
        assessed = assess([copy.deepcopy(extracted)], {"ORC1": parsed}, config,
                          cross_document=False)[0]
        field = assessed["fields"]["cr.loan_account_count"]
        report.check("ocr_recrop.scan_value_present", field.get("value") == 4,
                     f"value={field.get('value')} status={field.get('status')}")
        recovered = bool((field.get("meta") or {}).get("local_recrop"))
        reasons = field.get("review_reasons") or []
        report.check("ocr_recrop.recovered_value_held",
                     (not recovered)
                     or (field.get("decision") == "review" and "LOCAL_RECROP" in reasons),
                     f"recovered={recovered} decision={field.get('decision')} reasons={reasons}")
        recrop_count = int((parsed.get("quality") or {}).get("recrop_parts") or 0)
        if recrop_count:
            report.check("ocr_recrop.plumbing",
                         recovered and any(line.get("recovered")
                                           for line in first_page["lines"]),
                         f"recrop={recrop_count} meta={field.get('meta')}")
        else:
            report.check("ocr_recrop.plumbing", True,
                         "full-page OCR read the token directly; nothing to recover")
        grounding = field.get("grounding") or {}
        bbox = grounding.get("bbox") or []
        report.check("ocr_recrop.grounding",
                     grounding.get("page") == 1 and len(bbox) == 4
                     and 0 <= bbox[0] < bbox[2] <= first_page["width"]
                     and 0 <= bbox[1] < bbox[3] <= first_page["height"],
                     f"grounding={grounding}")

        if label_part:
            options = (config.get("parsing") or {}).get("ocr_recrop") or {}
            label_line = {"text": label_part["text"], "bbox": label_part["bbox"]}
            rect = gap_recrop_rect(label_line, first_page["width"], first_page["height"],
                                   options)
            doc = open_document(source, config)
            try:
                candidates = _crop_ocr(doc[0], rect, int(options.get("dpi", 400)), config)
            finally:
                doc.close()
            texts = [str(text).strip() for _, text, _ in candidates]
            report.check("ocr_recrop.crop_reads_isolated_token", "4" in texts,
                         f"texts={texts}")
        else:
            report.check("ocr_recrop.crop_reads_isolated_token", True,
                         "full-page OCR read the label with its value")

        off = copy.deepcopy(config)
        off["validation"]["review_on_local_recrop"] = False
        off_field = assess([copy.deepcopy(extracted)], {"ORC1": parsed}, off,
                           cross_document=False)[0]["fields"]["cr.loan_account_count"]
        report.check("ocr_recrop.configurable",
                     "LOCAL_RECROP" not in (off_field.get("review_reasons") or []),
                     "review_on_local_recrop=false")

        app_js = ROOT / "docpipe" / "web" / "app.js"
        report.check("ocr_recrop.ui_label",
                     app_js.exists()
                     and bool(re.search(r"LOCAL_RECROP\s*:",
                                        app_js.read_text(encoding="utf-8"))),
                     str(app_js))
    except Exception as exc:
        report.check("ocr_recrop.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_extractor_crosscheck(report: Report, config: Dict) -> None:
    """SCENARIO 列为硬性原因的 EXTRACTOR_CONFLICT 必须有代码真能触发，也不能逢干净件就报。

    永远不触发的门是装饰，一触发就误伤的门会被关掉，所以两端一起验：配置里写的通路都得
    有实现，干净样张上第二条通路要能作证且不报冲突，把单元格换成邻列的数必须进复核。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_extractor_crosscheck", True):
        return

    from .extract import extract_document
    from .parse import parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_crosscheck_", dir=str(work)))
    try:
        options = (config.get("validation") or {}).get("extractor_crosscheck") or {}
        implemented = {"line_cell", "statement_line", "declared_fallback"}
        configured = [str(pair.get("alternate") or "")
                      for pair in options.get("alternates") or []]
        report.check("extractor_crosscheck.alternates_implemented",
                     bool(configured) and set(configured) <= implemented
                     and bool(options.get("enabled")),
                     f"alternates={configured} enabled={options.get('enabled')}")
        report.check("extractor_crosscheck.reason_declared",
                     "EXTRACTOR_CONFLICT" in (config.get("hard_abstain_reasons") or []),
                     "硬性原因表里没有 EXTRACTOR_CONFLICT")
        app_js = ROOT / "docpipe" / "web" / "app.js"
        report.check("extractor_crosscheck.ui_evidence_row",
                     app_js.exists()
                     and "extractorConflictRow" in app_js.read_text(encoding="utf-8"),
                     str(app_js))

        source = resolve_path(config["paths"]["samples"], config) / "dev" / "D007.pdf"
        if not report.check("extractor_crosscheck.sample_available", source.exists(),
                            str(source)):
            return
        parsed = parse_document(source, "XCH1", force=True, config=config,
                                out_dir=temp_root / "parsed")
        extracted = extract_document(parsed, config)
        verdicts = {path: (field.get("meta") or {}).get("extractor_alternates") or {}
                    for path, field in extracted["fields"].items()}
        covered = [path for path, item in verdicts.items() if item]
        report.check("extractor_crosscheck.witness_coverage", len(covered) >= 10,
                     f"covered={len(covered)} fields={len(verdicts)}")
        noisy = sorted(path for path, item in verdicts.items() if "conflict" in item.values())
        report.check("extractor_crosscheck.clean_sample_agrees", not noisy,
                     f"conflicts={noisy}")

        tampered = copy.deepcopy(parsed)
        table = next(item for page in tampered["pages"] for item in page["tables"]
                     if item["name"] == "income")
        row = next(item for item in table["rows"]
                   if "营业收入" in str((item["cells"].get("项目") or {}).get("text") or ""))
        row["cells"]["本期金额"]["text"] = row["cells"]["上期金额"]["text"]
        hurt = extract_document(tampered, config)
        field = hurt["fields"]["is.revenue"]
        conflict = (field.get("meta") or {}).get("extractor_conflict") or {}
        report.check("extractor_crosscheck.neighbour_column_conflicts", bool(conflict),
                     f"value={field.get('value')} meta={field.get('meta')}")
        assessed = assess([copy.deepcopy(hurt)], {"XCH1": tampered}, config,
                          cross_document=False)[0]["fields"]["is.revenue"]
        report.check("extractor_crosscheck.conflict_forces_review",
                     assessed.get("decision") == "review"
                     and "EXTRACTOR_CONFLICT" in (assessed.get("review_reasons") or []),
                     f"decision={assessed.get('decision')} "
                     f"reasons={assessed.get('review_reasons')}")
    except Exception as exc:
        report.check("extractor_crosscheck.flow_exception", False,
                     f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_raw_audit(report: Report, config: Dict) -> None:
    """编造率回查必须真的能咬人：一个从不触发的门只是装饰。

    干净件上应当零命中（原文都是抄来的，不是拼出来的），但把原文换成页面上没有的
    串就必须报出来，并且开关打开时进硬性复核——两端一起验。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_raw_text_audit", True):
        return

    from .extract import _raw_text_audit, extract_document
    from .parse import parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_raw_audit_", dir=str(work)))
    try:
        options = (config.get("validation") or {}).get("raw_text_audit") or {}
        report.check("raw_audit.configured", bool(options.get("enabled")),
                     json.dumps(options, ensure_ascii=False))
        app_text = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
        report.check("raw_audit.ui_evidence_row",
                     "rawUnverifiedRow" in app_text and "RAW_UNVERIFIED" in app_text,
                     "复核台没有原文回查的中文文案")

        source = resolve_path(config["paths"]["samples"], config) / "dev" / "D007.pdf"
        if not report.check("raw_audit.sample_available", source.exists(), str(source)):
            return
        parsed = parse_document(source, "RAW1", force=True, config=config,
                                out_dir=temp_root / "parsed")
        extracted = extract_document(parsed, config)
        audited = [path for path, field in extracted["fields"].items()
                   if (field.get("meta") or {}).get("raw_grounded")
                   or (field.get("meta") or {}).get("raw_unverified")]
        report.check("raw_audit.coverage", len(audited) >= 10,
                     f"audited={len(audited)} fields={len(extracted['fields'])}")
        invented = sorted(path for path, field in extracted["fields"].items()
                          if (field.get("meta") or {}).get("raw_unverified"))
        report.check("raw_audit.clean_sample_invents_nothing", not invented,
                     f"unverified={invented}")

        target_path = next((path for path, field in extracted["fields"].items()
                            if field.get("raw_text")), "")
        if not report.check("raw_audit.field_with_raw_text", bool(target_path),
                            "样张里没有带原文的字段"):
            return
        # 把原文换成页面上确实没有的一串字：回查必须认出来。值不动，只查有没有出处。
        hurt = copy.deepcopy(extracted)
        hurt["fields"][target_path]["raw_text"] = "9,999,999.00"
        (hurt["fields"][target_path].get("meta") or {}).pop("raw_grounded", None)
        _raw_text_audit(hurt["fields"], parsed, config)
        evidence = (hurt["fields"][target_path].get("meta") or {}).get("raw_unverified") or {}
        report.check("raw_audit.invented_raw_is_flagged", bool(evidence),
                     f"raw=9,999,999.00 meta={hurt['fields'][target_path].get('meta')}")

        # 出厂配置自己就得咬人：以前这一条靠手改 config 才成立，等于验收替代码补了一条
        # 它自己没有的保证——开关默认关、码又不在强制表里，信号是双重惰性的。
        assessed = assess([copy.deepcopy(hurt)], {"RAW1": parsed}, config,
                          cross_document=False)[0]["fields"][target_path]
        report.check("raw_audit.forced_review_by_shipped_config",
                     assessed.get("decision") == "review"
                     and "RAW_UNVERIFIED" in (assessed.get("review_reasons") or []),
                     f"decision={assessed.get('decision')} "
                     f"reasons={assessed.get('review_reasons')}")
        # 默认打开的前提是标注集实测零误报：干净样本上的命中数由上面那条把关，这里钉住
        # 「开关 + 强制表必须同时在位」，否则「默认打开」只是一句口号。
        report.check("raw_audit.default_armed_on_measured_zero_hits",
                     config["validation"].get("review_on_raw_unverified") is True
                     and "RAW_UNVERIFIED" in (config.get("hard_abstain_reasons") or []),
                     f"switch={config['validation'].get('review_on_raw_unverified')} "
                     f"data_area_unverified={len(invented)}")
    except Exception as exc:
        report.check("raw_audit.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_reason_ui_labels(report: Report, config: Dict) -> None:
    """每个硬性复核原因都要在复核台有中文文案：原因码是给人读的，不是给机器读的。

    P0-8 漏过一次——MATRIX_FALLBACK 进了 hard_abstain_reasons，界面却只显示裸代码。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_reason_ui_labels", True):
        return
    app_js = ROOT / "docpipe" / "web" / "app.js"
    if not app_js.exists():
        report.check("reason_labels.every_hard_reason_has_ui_label", False, f"缺少 {app_js}")
        return
    block = re.search(r"const REASON_LABELS = \{(.*?)\n\};",
                      app_js.read_text(encoding="utf-8"), re.S)
    if not block:
        report.check("reason_labels.every_hard_reason_has_ui_label", False,
                     "REASON_LABELS 表结构变了，同步改这条检查")
        return
    labelled = set(re.findall(r"^\s*([A-Z][A-Z0-9_]+):", block.group(1), re.M))
    missing = [reason for reason in config.get("hard_abstain_reasons", [])
               if reason not in labelled]
    report.check("reason_labels.every_hard_reason_has_ui_label", not missing,
                 f"missing={missing} labelled={len(labelled)}")


def _check_action_rows(report: Report, label: str, rows: List[Dict],
                       table: Dict[str, str]) -> None:
    flagged = [row for row in rows if (row.get("review_reasons") or "").strip()]
    if not report.check(f"review_contract.{label}_has_flagged_rows", bool(flagged),
                        f"rows={len(rows)}"):
        return
    mismatched = []
    for row in flagged:
        codes = dict.fromkeys((row.get("review_reasons") or "").split("|"))
        want = "|".join(field_actions({"review_reasons": list(codes)}, table))
        if (row.get("review_actions") or "") != want:
            mismatched.append(row.get("field_path"))
    report.check(f"review_contract.{label}_actions_resolved", not mismatched,
                 f"mismatched={mismatched[:5]} flagged={len(flagged)}")


def _check_report_actions(report: Report, rows: List[Dict], html: str) -> None:
    """单文件网页版能不能替复核台回答「那我该核什么」。

    表格里的一格（`bs.transactions[3].amount`）在网页上没有独立的行，所以「怎么办」
    挂在整行下面、同一行的多个原因去重合并——holdout 唯一被拦的就是这样一格，
    只核标量字段的话，这条检查在 holdout 上会空跑。
    """
    checked, missing = 0, []
    for row in rows:
        if not (row.get("review_reasons") or "").strip():
            continue
        checked += 1
        if any(text and text not in html for text in (row.get("review_actions") or "").split("|")):
            missing.append(row.get("field_path"))
    report.check("review_contract.report_actions_in_html", bool(checked) and not missing,
                 f"checked={checked} missing={missing[:5]}")


def _check_action_summary(report: Report, label: str, data: Dict) -> None:
    contract = data.get("review_actions") or {}
    by_reason = contract.get("by_reason") or {}
    rows_ok = all(str(row.get("action") or "").strip() and row.get("field_count")
                  for row in by_reason.values())
    reviews = int((data.get("overall") or {}).get("review_count") or 0)
    report.check(f"review_contract.{label}_summary_actions",
                 rows_ok and not contract.get("unmapped")
                 and (bool(by_reason) or reviews == 0),
                 f"reasons={len(by_reason)} unmapped={contract.get('unmapped')} "
                 f"review_count={reviews}")


def _check_review_contract(report: Report, out_dir: Path, config: Dict) -> None:
    """只有原因码，下游知道字段被拦了，却不知道人该去核什么。

    契约要求每个硬性原因都有一句可执行的话，并且这句话要真的出现在四个出口
    （复核台、out/fields.csv、final/final_fields.csv、单文件网页版）上，不能只活在配置里。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_review_contract", True):
        return

    table = review_action_table(config)
    missing = [reason for reason in config.get("hard_abstain_reasons", [])
               if not str(table.get(reason) or "").strip()]
    report.check("review_contract.every_hard_reason_has_action", not missing,
                 f"missing={missing} mapped={len(table)}")

    app_text = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    report.check("review_contract.ui_shows_action",
                 "reviewActionRow" in app_text and "review_actions" in app_text,
                 "复核台没有「怎么办」行")
    # 单文件网页版是第四个出口：拿到成品的人不在复核台前，也得在同一份配置里看到指引。
    report_text = (ROOT / "docpipe" / "report.py").read_text(encoding="utf-8")
    report.check("review_contract.report_shows_action",
                 "field_actions" in report_text and 'class="hint"' in report_text
                 and "怎么办" in report_text,
                 "单文件网页版没接拒绝契约")

    summary_path = out_dir / "summary.json"
    if report.check("review_contract.out_summary_available", summary_path.exists(),
                    str(summary_path)):
        _check_action_summary(report, "out",
                              json.loads(summary_path.read_text(encoding="utf-8")))

    csv_path = out_dir / "fields.csv"
    if report.check("review_contract.out_csv_available", csv_path.exists(), str(csv_path)):
        encoding = config.get("output", {}).get("csv_encoding", "utf-8-sig")
        with csv_path.open("r", encoding=encoding, newline="") as handle:
            _check_action_rows(report, "out", list(csv.DictReader(handle)), table)


def _check_value_rewrite_disclosure(report: Report, out_dir: Path, config: Dict) -> None:
    """抽取把 1.2 变成 12000、把 500 变成 -500 时，屏幕上必须说清是谁改的。

    单位折算和正负号都在 meta 里算过，但字段值本身看不出名堂：复核员拿「值抄错了」
    去判一次本来正确的折算，等于系统自己造了一个假阳性。所以这句话由 meta 现算，
    再跟着 payload、out/fields.csv、复核台、单文件网页版走一遍——验收对同一份 meta
    重算一次，少一条就算漏；句子本身必须点名纸面上的依据（印的单位 / 印的收支方向），
    只说「系统换算过」等于没说。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_value_rewrite_disclosure", True):
        return

    documents = _stored_documents(out_dir)
    missing: List[str] = []
    vague: List[str] = []
    counted: List[str] = []
    kinds: Dict[str, int] = {}
    for doc_id, doc in sorted(documents.items()):
        for path, field in sorted((doc.get("fields") or {}).items()):
            note = value_rewrite_note(field)
            if not note:
                if str(field.get("rewrite") or "").strip():
                    missing.append(f"{doc_id}.{path}=无依据的改写句")
                continue
            kinds["unit" if "折算" in note else "sign"] = \
                kinds.get("unit" if "折算" in note else "sign", 0) + 1
            if str(field.get("rewrite") or "").strip() != note:
                missing.append(f"{doc_id}.{path}")
                continue
            # 「×10,000」和「纸上这一格没有负号」是两类改写的点名依据，缺一样就算空话。
            if "折算" in note and ("×" not in note or "纸上" not in note):
                vague.append(f"{doc_id}.{path}")
            elif "负号" in note and "纸上" not in note:
                vague.append(f"{doc_id}.{path}")
            counted.append(f"{doc_id}.{path}")

    report.check("value_rewrite.every_rewrite_field_carries_the_note",
                 bool(counted) and not missing,
                 f"notes={len(counted)} kinds={kinds} bad={','.join(missing[:4]) or 'none'}")
    report.check("value_rewrite.note_names_the_printed_source", not vague,
                 f"vague={','.join(vague[:4]) or 'none'}")

    app_text = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    report_text = (ROOT / "docpipe" / "report.py").read_text(encoding="utf-8")
    run_text = (ROOT / "docpipe" / "run.py").read_text(encoding="utf-8")
    export_text = (ROOT / "docpipe" / "export.py").read_text(encoding="utf-8")
    csv_path = out_dir / "fields.csv"
    csv_hits = None
    if csv_path.exists():
        encoding = config.get("output", {}).get("csv_encoding", "utf-8-sig")
        with csv_path.open("r", encoding=encoding, newline="") as handle:
            rows = list(csv.DictReader(handle))
        csv_hits = sum(1 for row in rows if str(row.get("value_rewrite") or "").strip())
    outlets = {
        "复核台": "rewriteRow" in app_text and "field.rewrite" in app_text
                 and "值改写" in app_text,
        "网页版": "rewriteRow" in report_text and '"rewrite"' in report_text
                 and "值改写" in report_text,
        "out_csv": "value_rewrite_note" in run_text and "value_rewrite" in run_text,
        "final_csv": "value_rewrite_note" in export_text and "value_rewrite" in export_text,
        "csv_行数": csv_hits is not None and csv_hits == len(counted),
    }
    report.check("value_rewrite.note_reaches_every_outlet", all(outlets.values()),
                 " ".join(f"{name}={'ok' if hit else 'NO'}" for name, hit in outlets.items())
                 + f" csv_rows={csv_hits} payload_notes={len(counted)}")


def _chain_fixture(kind: str) -> tuple:
    """手工拼一份流水，只喂链条校验看得见的东西。

    交付集（dev/holdout）里链条全是健康的，0 条 CHAIN_UNRELIABLE，所以这条只能
    在进程内造证据：真样本覆盖不到这件事，就明说覆盖不到，不许拿零命中当通过。
    """
    def field(path: str, value: str, risk: str = "monetary") -> Dict:
        return {"path": path, "label": path.rsplit(".", 1)[-1], "value": value,
                "value_type": "decimal", "required": True, "risk": risk,
                "grounding": {"page": 1, "bbox": [10.0, 10.0, 80.0, 20.0]},
                "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                            "table_score": 1.0, "cross_field_score": 1.0,
                            "grounding_present": 1},
                "meta": {}, "decision": "review", "validators": [], "review_reasons": []}

    good = [("100.00", "1100.00"), ("100.00", "1200.00"), ("100.00", "1300.00")]
    rows, closing, totals, lines = good, "1300.00", None, []
    if kind == "ratio":
        rows, closing = [good[0]] + [("100.00", "999999.00")] * 5, None
    elif kind == "closing":
        closing = "1500.00"
    elif kind == "totals":
        totals = ("900.00", "0.00")
    elif kind == "account":
        lines = ["账号：6222020000000000"]
    fields = {"bs.account_number": field("bs.account_number", "6222021234567890", "identity"),
              "bs.opening_balance": field("bs.opening_balance", "1000.00")}
    if closing is not None:
        fields["bs.closing_balance"] = field("bs.closing_balance", closing)
    if totals is not None:
        fields["bs.total_income"] = field("bs.total_income", totals[0])
        fields["bs.total_expense"] = field("bs.total_expense", totals[1])
    for index, (amount, balance) in enumerate(rows):
        fields[f"bs.transactions[{index}].amount"] = field(
            f"bs.transactions[{index}].amount", amount)
        fields[f"bs.transactions[{index}].balance"] = field(
            f"bs.transactions[{index}].balance", balance)
        fields[f"bs.transactions[{index}].txn_date"] = field(
            f"bs.transactions[{index}].txn_date", f"2026-03-{index + 1:02d}", "date")
    doc = {"document_id": "CHAIN-" + kind.upper(), "doc_type": "BANK_STATEMENT",
           "fields": fields}
    parsed = {"document_id": doc["document_id"], "doc_type": "BANK_STATEMENT",
              "page_count": 1, "quality": {"parse_score": 0.99},
              "pages": [{"page": 1, "width": 595.0, "height": 842.0, "quality_flags": [],
                         "lines": [{"text": text} for text in lines], "tables": []}]}
    return doc, parsed


def _assess_chain(kind: str, config: Dict) -> Dict:
    """造一份流水，走完判决 + 出口句子这一遍，返回真正发给复核台的那个 payload。

    句子是由 `run._document_payload` 盖章的（和 `rewrite` 同一层）：判决层只出原因码，
    「说人话」归出口层。验收要盯的就是这一层有没有漏盖章，所以不能只看 assess 的结果。
    """
    from .run import _document_payload
    options = copy.deepcopy(config)
    options.setdefault("calibration", {})["artifact"] = None
    doc, parsed = _chain_fixture(kind)
    assessed = assess([doc], {doc["document_id"]: parsed}, options, cross_document=False)[0]
    payload = _document_payload(assessed, parsed, Path("chain.pdf"), config=options)
    payload["chain_stats"] = assessed.get("chain_stats") or {}
    return payload


def _check_chain_disclosure(report: Report, out_dir: Path, config: Dict) -> None:
    """流水链判坏时，界面要说出是「多少行里错了几行、被哪条证据判坏」。

    过去 CHAIN_UNRELIABLE 只是一个码：一次错 5 行和一次错 30 行推给同样一批字段，
    复核员看不出这份流水坏到什么程度，只能整份重看。诊断数（chain_failures/rows/
    ratio）本来就在 `validate_bank_statement` 里算过，算完就丢——本条盯的是「算了但
    不说」这一类，不是判决本身。触发码有四种（ratio/closing/totals/account），漏一种
    就等于那句话只说一半；链健康时必须一个字不出，否则每格都带提示，这句提示当天作废。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_chain_disclosure", True):
        return

    expected = {"ratio": ("6 行里 5 行", "83.3%"), "closing": ("末行余额与期末余额对不上",),
                "totals": ("期初＋收入－支出与期末余额对不上",), "account": ("跨页账号不是同一个",)}
    healthy = _assess_chain("healthy", config)
    healthy_stats = healthy.get("chain_stats") or {}
    healthy_notes = [path for path, field in healthy["fields"].items() if field.get("chain")]
    report.check("chain.healthy_statement_stays_silent",
                 healthy_stats.get("chain_unreliable") is False and not healthy_notes
                 and chain_note(healthy_stats) == "",
                 f"stats={healthy_stats} notes={','.join(healthy_notes[:4]) or 'none'}")

    wrong: List[str] = []
    for kind, needles in expected.items():
        doc = _assess_chain(kind, config)
        stats = doc.get("chain_stats") or {}
        blocked = {path for path, field in doc["fields"].items()
                   if "CHAIN_UNRELIABLE" in (field.get("review_reasons") or [])}
        notes = {path: str(field.get("chain") or "")
                 for path, field in doc["fields"].items() if field.get("chain")}
        if stats.get("triggers") != [kind]:
            wrong.append(f"{kind}:triggers={stats.get('triggers')}")
            continue
        if not blocked or set(notes) != blocked:
            wrong.append(f"{kind}:覆盖={len(notes)}/{len(blocked)}")
            continue
        if any(field.get("decision") != "review"
               for path, field in doc["fields"].items() if path in notes):
            wrong.append(f"{kind}:改写跟着了accepted字段")
            continue
        text = {note for note in notes.values()}
        if len(text) != 1:
            wrong.append(f"{kind}:句子不唯一={len(text)}")
            continue
        sentence = text.pop()
        if not sentence.startswith("流水链被判不可靠："):
            wrong.append(f"{kind}:没有点名是链判坏={sentence}")
            continue
        for needle in needles:
            if needle not in sentence:
                wrong.append(f"{kind}:缺依据「{needle}」={sentence}")

    report.check("chain.each_trigger_gets_its_own_sentence", not wrong,
                 f"kinds={len(expected)} bad={','.join(wrong[:4]) or 'none'}")

    app_text = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    report_text = (ROOT / "docpipe" / "report.py").read_text(encoding="utf-8")
    run_text = (ROOT / "docpipe" / "run.py").read_text(encoding="utf-8")
    export_text = (ROOT / "docpipe" / "export.py").read_text(encoding="utf-8")
    outlets = {
        "复核台": "chainRow" in app_text and "field.chain" in app_text
                 and "链诊断" in app_text,
        "网页版": "chainNote" in report_text and '"chain": _doc_chain_note(doc)' in report_text
                 and "链诊断" in report_text,
        "out_csv": "chain_note" in run_text and "chain_evidence" in run_text,
        "final_csv": "chain_evidence" in export_text,
    }
    report.check("chain.disclosure_reaches_every_outlet", all(outlets.values()),
                 " ".join(f"{name}={'ok' if hit else 'NO'}" for name, hit in outlets.items()))

    payload_notes = sum(1 for doc in _stored_documents(out_dir).values()
                        for field in (doc.get("fields") or {}).values() if field.get("chain"))
    csv_path = out_dir / "fields.csv"
    csv_hits = None
    if csv_path.exists():
        encoding = config.get("output", {}).get("csv_encoding", "utf-8-sig")
        with csv_path.open("r", encoding=encoding, newline="") as handle:
            csv_hits = sum(1 for row in csv.DictReader(handle)
                           if str(row.get("chain_evidence") or "").strip())
    report.check("chain.csv_and_payload_counts_agree",
                 csv_hits is not None and csv_hits == payload_notes,
                 f"csv_rows={csv_hits} payload_notes={payload_notes}"
                 "（本集链条全健康，0 条属正常；有判决覆盖靠的是上面那条进程内夹具）")


def _check_review_loop(report: Report, out_dir: Path, config: Dict) -> None:
    """人工判定只用一次就丢了，等于每份资料都从零学。

    复核台账里 accept / correct 是同一条特征向量上的人裁判决法，必须能变回校准
    样本。验收不重训模型（要保持快），验的是判决方向、去重规则、"样本不足整批不接"
    与"开关关着一个目录都不扫"这几条机械装置，外加单类池子必须报错拒训——缺负例
    只能靠真实人工判决补，不许靠调阈值蒙过去。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_review_loop", True):
        return
    options = config.get("calibration") or {}
    report.check("review_loop.gate_ships_disabled",
                 options.get("learn_from_review") is False
                 and bool(options.get("review_dirs"))
                 and int(options.get("review_min_samples") or 0) >= 10,
                 f"learn_from_review={options.get('learn_from_review')} "
                 f"dirs={options.get('review_dirs')} min={options.get('review_min_samples')}")
    report.check("review_loop.disabled_gate_scans_nothing",
                 not review_sample_dirs(config) and not review_samples([], config),
                 "开关关着却还是扫到了目录")

    fixture_root = resolve_path(config["paths"]["work"], config) / "eval" / "review_loop"
    shutil.rmtree(fixture_root, ignore_errors=True)
    stored = _stored_documents(out_dir)
    if not report.check("review_loop.stored_document_available", bool(stored), str(out_dir)):
        return
    document = stored[sorted(stored)[0]]
    fields = document.get("fields") or {}
    grounded = [path for path, field in sorted(fields.items()) if field.get("grounding")]
    if not report.check("review_loop.three_grounding_fields", len(grounded) >= 3,
                        f"found={len(grounded)}"):
        return
    flipped, positive, abandoned = grounded[0], grounded[1], grounded[2]
    workspace = fixture_root / "out"
    (workspace / "RX01").mkdir(parents=True)
    clone = copy.deepcopy(document)
    clone["document_id"] = "RX01"
    (workspace / "RX01" / "RX01.json").write_text(
        json.dumps(clone, ensure_ascii=False), encoding="utf-8")
    review_dir = workspace / "review"
    review_dir.mkdir(parents=True)

    def record(path, action, stamp, value=None):
        item = {"document_id": "RX01", "field_path": path, "action": action,
                "timestamp": stamp}
        if value is not None:
            item["corrected_value"] = value
        return item

    records = [
        record(flipped, "accept", "2026-10-26T09:00:00"),
        "{这不是 JSON",
        record("zz.not_a_field", "accept", "2026-10-26T09:00:01"),
        record(positive, "accept", "2026-10-26T09:00:02"),
        record(flipped, "correct", "2026-10-26T09:00:03", value="__reviewed_wrong__"),
        # 最后一句不是判决（撤回复核），前面那条 accept 就不算数：没裁完的字段不进样本池。
        record(positive, "accept", "2026-10-26T09:00:05"),
        record(abandoned, "accept", "2026-10-26T09:00:06"),
        record(abandoned, "pending", "2026-10-26T09:00:07"),
    ]
    (review_dir / "RX01.review.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8")

    samples = review_samples([workspace], config)
    by_path = {item["path"]: item for item in samples}
    report.check("review_loop.nonalterminal_and_broken_records_dropped",
                 len(samples) == 2 and set(by_path) == {flipped, positive},
                 f"paths={sorted(by_path)} records={len(records)}")
    report.check("review_loop.last_record_wins",
                 by_path.get(flipped, {}).get("review_action") == "correct"
                 and by_path.get(positive, {}).get("review_action") == "accept",
                 f"flipped={by_path.get(flipped, {}).get('review_action')} "
                 f"positive={by_path.get(positive, {}).get('review_action')}")
    report.check("review_loop.accept_is_positive_and_correct_is_negative",
                 by_path.get(flipped, {}).get("y") == 0 and by_path.get(positive, {}).get("y") == 1,
                 f"flipped_y={by_path.get(flipped, {}).get('y')} "
                 f"positive_y={by_path.get(positive, {}).get('y')}")
    replayed = all(item["features"] == field_features(fields[item["path"]],
                                                      document.get("doc_type", ""), config)
                   for item in samples)
    report.check("review_loop.features_replay_stored_signals", replayed,
                 "复现出的特征向量和结果目录里存下来的 signals 不一致")

    _, shadowed = merge_review_samples([{"document_id": "RX01", "path": flipped}],
                                       samples, config)
    report.check("review_loop.labels_shadow_review_records",
                 shadowed["shadowed_by_labels"] == 1 and not shadowed["enabled"],
                 f"stats={shadowed}")
    strict = copy.deepcopy(config)
    strict["calibration"]["review_min_samples"] = 20
    refused, stats = merge_review_samples([], samples, strict)
    report.check("review_loop.small_batch_refuses_retrain",
                 not stats["enabled"] and not refused and stats["usable"] == 2,
                 f"stats={stats}")
    loose = copy.deepcopy(config)
    loose["calibration"]["review_min_samples"] = 1
    folded, stats = merge_review_samples([], samples, loose)
    report.check("review_loop.threshold_allows_retrain",
                 stats["enabled"] and len(folded) == 2
                 and stats["accepted"] == 1 and stats["corrected"] == 1,
                 f"stats={stats}")
    # 铁律审计 XIV：台账签的是当时那一版的值与判决，结果目录重跑过就不许再当同一条证据。
    report.check("review_loop.anchorless_records_are_not_judged",
                 all(not item["drift"] for item in samples),
                 f"drift={[item['drift'] for item in samples]}")

    drift_root = fixture_root / "drift"
    (drift_root / "review").mkdir(parents=True)
    shutil.copytree(workspace / "RX01", drift_root / "RX01")
    before = {path: (fields[path].get("value"), fields[path].get("decision"))
              for path in (flipped, abandoned, positive)}
    drifted = copy.deepcopy(document)
    drifted["fields"][flipped]["value"] = "重跑之后这一版换了值"
    drifted["fields"][abandoned]["decision"] = "review"
    (drift_root / "RX01" / "RX01.json").write_text(
        json.dumps(drifted, ensure_ascii=False), encoding="utf-8")
    ledger = []
    for path in (flipped, abandoned, positive):
        value, decision = before[path]
        ledger.append({"document_id": "RX01", "field_path": path, "action": "accept",
                       "timestamp": "2026-10-26T09:10:00",
                       "original_value": value, "auto_decision": decision})
    (drift_root / "review" / "RX01.review.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ledger),
        encoding="utf-8")
    drift_samples = {item["path"]: item for item in review_samples([drift_root], config)}
    report.check("review_loop.rewritten_product_is_caught_by_the_ledger_anchors",
                 set(drift_samples) == {flipped, abandoned, positive}
                 and drift_samples[flipped]["drift"] == "value"
                 and drift_samples[abandoned]["drift"] == "decision"
                 and drift_samples[positive]["drift"] == "",
                 f"drift={ {path: item['drift'] for path, item in drift_samples.items()} }")
    folded_clean, drift_stats = merge_review_samples([], list(drift_samples.values()), loose)
    report.check("review_loop.shipped_policy_drops_only_the_drifted_ones",
                 drift_stats["drift_policy"] == "drop" and drift_stats["drift"] == 2
                 and drift_stats["drift_value"] == 1 and drift_stats["drift_decision"] == 1
                 and drift_stats["usable"] == 1 and drift_stats["enabled"]
                 and [item["path"] for item in folded_clean] == [positive],
                 f"stats={ {key: drift_stats[key] for key in ('drift', 'drift_value', 'drift_decision', 'usable')} }")
    permissive = copy.deepcopy(loose)
    permissive["calibration"]["review_drift"] = "keep"
    kept_all, keep_stats = merge_review_samples([], list(drift_samples.values()),
                                                permissive)
    report.check("review_loop.keep_policy_folds_them_but_still_counts_them",
                 keep_stats["drift_policy"] == "keep" and keep_stats["drift_kept"] == 2
                 and keep_stats["drift_dropped"] == 0 and keep_stats["usable"] == 3
                 and len(kept_all) == 3,
                 f"stats={ {key: keep_stats[key] for key in ('drift_kept', 'drift_dropped', 'usable')} }")

    second = fixture_root / "second"
    (second / "review").mkdir(parents=True)
    shutil.copytree(workspace / "RX01", second / "RX01")
    older = {"document_id": "RX01", "field_path": positive, "action": "correct",
             "timestamp": "2026-10-26T08:00:00"}
    (second / "review" / "RX01.review.jsonl").write_text(
        json.dumps(older, ensure_ascii=False) + "\n", encoding="utf-8")
    across = {item["path"]: item for item in review_samples([workspace, second], config)}
    report.check("review_loop.newest_record_wins_across_dirs",
                 len(across) == 2 and across.get(positive, {}).get("review_action") == "accept",
                 f"paths={sorted(across)}")

    # 这一条问的是「池子只有一类时必须拒训」，与哪个 split 无关；固定用开发集，
    # 因为留出集按 SCENARIO 第 108 行不许被拟合（`paths.evaluation_only_splits`）。
    split = "dev"
    pool = build_samples(split, config)
    positives, negatives = _class_balance(pool)
    single_class = positives == 0 or negatives == 0
    refused: Optional[str] = None
    try:
        train(split, config)
    except SystemExit as error:
        refused = str(error)
    # 全正例（或全负例）的池子拟合不出概率，必须当场报错说清为什么，而不是把
    # sklearn 的 traceback 丢给操作者，更不许靠调阈值蒙过去。
    report.check("review_loop.single_class_pool_refuses_to_fit",
                 bool(refused) == single_class
                 and (not single_class or ("负例" in refused and "复核台账" in refused)),
                 f"positives={positives} negatives={negatives} "
                 f"refused={bool(refused)} message={refused or ''}")

    review_negatives = [item for item in samples if item["y"] == 0]
    mixed, mixed_stats = merge_review_samples(pool, review_negatives, loose)
    _, mixed_negatives = _class_balance(mixed)
    fits_after = True
    try:
        _require_two_classes(*_class_balance(mixed), "验收 折入复核负例之后")
    except SystemExit:
        fits_after = False
    report.check("review_loop.review_negative_makes_the_pool_fittable",
                 fits_after and mixed_stats["enabled"]
                 and mixed_negatives >= negatives + (1 if review_negatives else 0),
                 f"pool_negatives={negatives} mixed_negatives={mixed_negatives} "
                 f"stats={mixed_stats}")


def _check_table_structure(report: Report, out_dir: Path, labels_dir: Path,
                           config: Dict) -> Optional[Dict]:
    """格子取对没有——值读对不代表格子取对（P1-6 / G4，对标 docTR TableCellMetric）。

    这份数字只从 labels/ 与 work/parsed 算出来：不重跑解析、不碰 out/。
    它没有给任何字段放行的能力，最后一条检查锁的就是这件事。
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_table_structure", True):
        return

    from . import tablescore

    work = resolve_path(config["paths"]["work"], config)
    parsed_dir = work / "parsed"
    # 退化归因只有难例集才有：dev/holdout 没有这份 manifest，整个分片都算干净件。
    manifest = ROOT / "samples" / labels_dir.name / "manifest.json"

    def fingerprint(root: Path) -> Dict[str, str]:
        if not root.exists():
            return {}
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*")) if path.is_file()}

    before = fingerprint(out_dir)
    payload = tablescore.score_split(labels_dir, parsed_dir, config,
                                     manifest if manifest.exists() else None)
    after = fingerprint(out_dir)
    documents = payload["documents"]
    overall = payload["overall"]

    params = (config.get("evaluation") or {}).get("table_structure") or {}
    module_source = (ROOT / "docpipe" / "tablescore.py").read_text(encoding="utf-8")
    unread = [key for key in sorted(params) if f'"{key}"' not in module_source]
    report.check("table_structure.thresholds_live_in_config",
                 bool(params) and not unread,
                 f"keys={sorted(params)} unread={unread}")

    report.check("table_structure.scores_the_split",
                 payload["documents_scored"] > 0 and overall["gt_cells"] > 0
                 and not payload["cache_missing"],
                 f"docs={payload['documents_scored']} gt={overall['gt_cells']} "
                 f"det={overall['det_cells']} cache_missing={payload['cache_missing'][:5]}")

    # 结构分读的是共享解析缓存，缓存按 doc_id 命名；doc_id 对不上就是拿别人的网格打分。
    mismatched = []
    for document in documents:
        parsed_path = parsed_dir / f"{document['document_id']}.json"
        parsed = json.loads(parsed_path.read_text(encoding="utf-8")) if parsed_path.exists() else {}
        if parsed.get("document_id") != document["document_id"] \
                or parsed.get("doc_type") != document["doc_type"]:
            mismatched.append((document["document_id"], parsed.get("document_id"),
                               parsed.get("doc_type")))
    report.check("table_structure.cache_matches_labels", not mismatched,
                 f"docs={len(documents)} mismatched={mismatched[:3]}")

    leaks = [(document["document_id"], name) for document in documents
             for name, rec in document["tables"].items()
             if rec["aligned"] + rec["misassigned"] + rec["missed_cell"] != rec["gt_cells"]]
    report.check("table_structure.denominator_conserved", not leaks,
                 f"tables={sum(len(d['tables']) for d in documents)} leaks={leaks[:3]}")

    undeclared = sorted({reason for document in documents
                         for reason in list(document["skipped"]) + list(document["detected_skipped"])
                         if reason not in tablescore.DECLARED_SKIP_REASONS})
    report.check("table_structure.skipped_reasons_are_declared", not undeclared,
                 f"undeclared={undeclared} vocabulary={len(tablescore.DECLARED_SKIP_REASONS)}")

    clean = [document for document in documents if not document["degradation"]]
    clean_totals = tablescore._totals([rec for d in clean for rec in d["tables"].values()])
    degraded_errors = sum(rec["misassigned"] + rec["missed_cell"]
                          for d in documents if d["degradation"] for rec in d["tables"].values())
    report.check("table_structure.clean_docs_have_no_structure_error",
                 clean_totals["misassigned"] == 0 and clean_totals["missed_cell"] == 0
                 and (clean_totals["aligned"] == 0
                      or clean_totals["structure_exact_rate"] == 1.0),
                 f"clean_docs={len(clean)} aligned={clean_totals['aligned']} "
                 f"misassigned={clean_totals['misassigned']} missed={clean_totals['missed_cell']} "
                 f"struct={clean_totals['structure_exact_rate']} "
                 f"degraded_errors={degraded_errors} "
                 f"manifest={'yes' if manifest.exists() else 'no'}")

    # 反例：把一份当前完美的网格在内存里改坏，结构分必须看得见（看不见就是度量瞎了）。
    # 优先用干净件，退化件里那张恰好还完美的网格也行——这里要的是「现在全对」，不是「没退化」。
    def _has_intact_grid(document: Dict) -> bool:
        return any(rec["gt_cells"] > 0 and rec["det_cells"] > 0
                   and rec["misassigned"] == 0 and rec["missed_cell"] == 0
                   for rec in document["tables"].values())

    intact = [document for document in documents if _has_intact_grid(document)]
    target = next((d for d in intact if not d["degradation"]), intact[0] if intact else None)
    if target is None:
        report.check("table_structure.sees_a_broken_grid", False,
                     f"没有一张完整网格可供破坏（docs={len(documents)}）")
    else:
        doc_id = target["document_id"]
        labels = json.loads((labels_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        parsed = json.loads((parsed_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        name = max((rec for rec in target["tables"].values() if rec["det_cells"]),
                   key=lambda rec: rec["gt_cells"])["table"]

        def _instances(record: Dict) -> List[Dict]:
            return [table for page in record.get("pages") or []
                    for table in page.get("tables") or [] if table.get("name") == name]

        dropped = copy.deepcopy(parsed)
        for table in _instances(dropped):
            table["rows"] = table.get("rows")[1:]
        dropped_totals = tablescore._totals(
            [rec for rec in tablescore.score_document(
                doc_id, target["doc_type"], labels, dropped, config,
                tablescore._params(config))["tables"].values()])

        # 互换同一行里两格的盒子：行号没变、列号串了，正是「余额串进交易金额位」的形状。
        grid = tablescore.canonical_column_order(target["doc_type"], config)[name]
        gt_cells, _skipped = tablescore.ground_truth_cells(labels, target["doc_type"], config)
        pair = next(((a, b) for a in gt_cells for b in gt_cells
                     if a["table"] == name and b["table"] == name
                     and a["page"] == b["page"] and a["row"] == b["row"]
                     and a["col"] < b["col"]), None)
        swapped = copy.deepcopy(parsed)
        touched = None
        if pair:
            left, right = grid[pair[0]["col"]], grid[pair[1]["col"]]
            for page in swapped.get("pages") or []:
                for table in page.get("tables") or []:
                    if table.get("name") != name or page.get("page") != pair[0]["page"]:
                        continue
                    for row in table.get("rows") or []:
                        cells = row.get("cells") or {}
                        if left in cells and right in cells:
                            cells[left], cells[right] = cells[right], cells[left]
                            touched = (left, right)
                            break
                    if touched:
                        break
                if touched:
                    break
        swapped_totals = tablescore._totals(
            [rec for rec in tablescore.score_document(
                doc_id, target["doc_type"], labels, swapped, config,
                tablescore._params(config))["tables"].values()]) if touched else None
        report.check("table_structure.sees_a_broken_grid",
                     dropped_totals["missed_cell"] + dropped_totals["misassigned"] > 0
                     and (dropped_totals["cell_found_rate"] or 0) < 1.0
                     and swapped_totals is not None
                     and swapped_totals["misassigned"] > 0
                     and (swapped_totals["structure_exact_rate"] or 0) < 1.0,
                     f"doc={doc_id} table={name} 掉一行→missed={dropped_totals['missed_cell']} "
                     f"misassigned={dropped_totals['misassigned']} "
                     f"found={dropped_totals['cell_found_rate']}；"
                     f"互换列={touched} →misassigned={(swapped_totals or {}).get('misassigned')} "
                     f"struct={(swapped_totals or {}).get('structure_exact_rate')}")

    # 结构错格只采样到 examples（每表 max_examples 条），这里能查的都是有明细的那几格。
    errors = [(document["document_id"], example["path"]) for document in documents
              for rec in document["tables"].values() for example in rec["examples"]]
    auto_accepted = []
    for doc_id, path in errors:
        pred_path = out_dir / doc_id / f"{doc_id}.json"
        field = (json.loads(pred_path.read_text(encoding="utf-8")).get("fields") or {}).get(path) \
            if pred_path.exists() else None
        if field is None:
            auto_accepted.append((doc_id, path, "missing"))
        elif field.get("decision") == "accept":
            auto_accepted.append((doc_id, path, field.get("decision")))
    report.check("table_structure.structure_error_never_auto_accepted", not auto_accepted,
                 f"sampled_errors={len(errors)} auto_accepted={auto_accepted[:5]}")

    decision_path = [ROOT / "docpipe" / "trust.py", ROOT / "docpipe" / "extract.py",
                     ROOT / "docpipe" / "run.py", ROOT / "docpipe" / "evaluate.py",
                     ROOT / "docpipe" / "serve.py"]
    wired = [str(path.relative_to(ROOT)) for path in decision_path
             if path.exists() and "tablescore" in path.read_text(encoding="utf-8")]
    report.check("table_structure.is_diagnostic_only", not wired and after == before,
                 f"imported_by_decision_path={wired} out_changed={len(set(before) ^ set(after))} "
                 f"out_files={len(before)}")
    return {"metric_version": payload["metric_version"], "params": payload["params"],
            "labels_dir": payload["labels_dir"], "parsed_dir": payload["parsed_dir"],
            "documents_scored": payload["documents_scored"],
            "cache_missing": payload["cache_missing"], "overall": payload["overall"],
            "by_table": payload["by_table"], "by_degradation": payload["by_degradation"],
            "note": payload["note"]}


def _check_page_quality_gate(report: Report, config: Dict) -> None:
    """A 120dpi scan does not fail loudly: OCR reads blurred digits with high confidence.

    The only honest defence is the page's own sampling density, so the gate has to
    survive parse -> extract -> decide and be switchable from the config.
    """
    gates = config.get("acceptance") or {}
    if not gates.get("require_page_quality_gate", True):
        return

    import pymupdf as fitz

    from .extract import extract_document
    from .parse import parse_document
    from .trust import assess

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_raster_", dir=str(work)))
    try:
        cfg = copy.deepcopy(config)
        cfg["paths"]["work"] = str(temp_root / "work")
        samples = sorted((resolve_path(cfg["paths"]["samples"], cfg) / "dev").glob("*.pdf"))
        if not report.check("page_quality.sample_available", bool(samples),
                            f"samples={len(samples)}"):
            return
        source = samples[0]

        def run(dpi: int, doc_id: str):
            path = temp_root / f"{doc_id}.pdf"
            original = fitz.open(source)
            raster = fitz.open()
            for page in original:
                target = raster.new_page(width=page.rect.width, height=page.rect.height)
                target.insert_image(target.rect,
                                    pixmap=page.get_pixmap(dpi=dpi, alpha=False))
            raster.save(path, garbage=4, deflate=True)
            raster.close()
            original.close()
            parsed = parse_document(path, doc_id, force=True, config=cfg,
                                    out_dir=temp_root / "parsed")
            extracted = extract_document(parsed, cfg)
            assessed = assess([copy.deepcopy(extracted)], {doc_id: parsed}, cfg,
                              cross_document=False)[0]
            return parsed, extracted, assessed

        low_dpi = int(gates.get("page_quality_fixture_dpi", 100))
        parsed, _extracted, doc = run(low_dpi, "PQLOW")
        quality = parsed.get("quality") or {}
        wanted_pages = list(range(1, int(parsed.get("page_count") or 0) + 1))
        report.check("page_quality.low_flagged",
                     "LOW_RESOLUTION" in (quality.get("photo_flags") or [])
                     and quality.get("low_resolution_pages") == wanted_pages
                     and all(page["raster"] and page["raster"]["low_resolution"]
                             for page in parsed["pages"]),
                     f"dpi={low_dpi} effective="
                     f"{[page['raster']['effective_dpi'] if page['raster'] else None for page in parsed['pages']]}")

        grounded = [field for field in doc["fields"].values()
                    if (field.get("grounding") or {}).get("page")]
        gated = [field for field in grounded
                 if "LOW_RESOLUTION" in (field.get("review_reasons") or [])
                 and field.get("decision") == "review"]
        report.check("page_quality.forces_review",
                     bool(grounded) and len(gated) == len(grounded),
                     f"gated={len(gated)}/{len(grounded)}")

        off = copy.deepcopy(cfg)
        off["validation"]["review_on_low_resolution"] = False
        reassessed = assess([copy.deepcopy(_extracted)], {parsed["document_id"]: parsed},
                            off, cross_document=False)[0]
        report.check("page_quality.configurable",
                     not any("LOW_RESOLUTION" in (field.get("review_reasons") or [])
                             for field in reassessed["fields"].values()),
                     "review_on_low_resolution=false")

        # 分级档位必须两头都能动：够用的字段放出去，要求更高的字段比整页门更早进复核。
        measured = min(page["raster"]["effective_dpi"] for page in parsed["pages"]
                       if page.get("raster"))
        tiered = copy.deepcopy(cfg)
        tiers = tiered["parsing"]["page_quality"].setdefault("risk_min_dpi", {})
        tiers.update({"enabled": True, "default": measured - 5, "monetary": measured + 5})
        tier_doc = assess([copy.deepcopy(_extracted)], {parsed["document_id"]: parsed},
                          tiered, cross_document=False)[0]
        grounded_fields = [field for field in tier_doc["fields"].values()
                           if (field.get("grounding") or {}).get("page")]
        exempt = [field for field in grounded_fields if field.get("risk") != "monetary"]
        held = [field for field in grounded_fields if field.get("risk") == "monetary"]
        report.check("page_quality.risk_tier_grades_the_gate",
                     bool(grounded_fields) and bool(exempt) and bool(held)
                     and not any("LOW_RESOLUTION" in (field.get("review_reasons") or [])
                                 for field in exempt)
                     and all("LOW_RESOLUTION" in (field.get("review_reasons") or [])
                             for field in held),
                     f"measured={round(measured, 1)} exempt={len(exempt)} held={len(held)}")

        clean_parsed, _clean_extracted, clean_doc = run(300, "PQCLEAN")
        clean_quality = clean_parsed.get("quality") or {}
        report.check("page_quality.clean_scan_untouched",
                     "LOW_RESOLUTION" not in (clean_quality.get("photo_flags") or [])
                     and clean_quality.get("low_resolution_pages") == []
                     and any(field.get("decision") == "accept"
                             for field in clean_doc["fields"].values()),
                     f"flags={clean_quality.get('photo_flags')} "
                     f"effective={[page['raster']['effective_dpi'] if page['raster'] else None for page in clean_parsed['pages']]}")

        app_js = ROOT / "docpipe" / "web" / "app.js"
        report.check("page_quality.ui_label",
                     app_js.exists()
                     and bool(re.search(r"LOW_RESOLUTION\s*:", app_js.read_text(encoding="utf-8"))),
                     str(app_js))
    except Exception as exc:
        report.check("page_quality.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _post_bytes(port: int, path: str, body: bytes):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _zip_bytes(entries, compress: int = zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compress) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _png_variant(blob: bytes, color) -> bytes:
    """Same page, different bytes, so the inbox de-duplicator does not merge the two."""
    from PIL import Image, ImageDraw

    with Image.open(io.BytesIO(blob)) as image:
        canvas = image.convert("RGB")
    ImageDraw.Draw(canvas).rectangle([0, 0, 60, 60], fill=tuple(color))
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG")
    return buffer.getvalue()


def _check_ingest_flow(report: Report, config: Dict) -> None:
    """Upload -> review -> remove on a throwaway workspace; removal must be reversible and scoped."""
    gates = config.get("acceptance") or {}
    if not gates.get("require_ingest_flow", True):
        return

    import pymupdf as fitz

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_ingest_", dir=str(work)))
    server = thread = None
    try:
        cfg = copy.deepcopy(config)
        cfg["paths"]["work"] = str(temp_root / "work")
        cfg["paths"]["inbox"] = str(temp_root / "inbox")
        cfg["paths"]["user_out"] = str(temp_root / "out")
        samples = sorted((resolve_path(cfg["paths"]["samples"], cfg) / "dev").glob("*.pdf"))
        if not report.check("ingest.flow_sample", bool(samples), f"samples={len(samples)}"):
            return
        blob = fitz.open(samples[0])[0].get_pixmap(dpi=120).tobytes("png")
        out_dir = Path(cfg["paths"]["user_out"])
        out_dir.mkdir(parents=True, exist_ok=True)

        server = make_server(out_dir, "127.0.0.1", 0, cfg)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        if not report.check("ingest.flow_server", _wait_ready(port), f"port={port}"):
            return

        uploads = [_post_bytes(port, f"/api/import?filename={quote(name)}", blob)
                   for name in ("验收甲.png", "验收乙.png")]
        status, payload = _post(port, "/api/import/run", {})
        ids = sorted(payload.get("document_ids") or [])
        report.check("ingest.flow_import",
                     all(item[0] == 200 for item in uploads) and status == 200 and len(ids) == 2,
                     f"documents={ids}")

        def source_name(doc_id: str) -> str:
            return json.loads((out_dir / doc_id / f"{doc_id}.json").read_text(
                encoding="utf-8"))["source"]["file_name"]

        doomed = next((doc_id for doc_id in ids if source_name(doc_id) == "验收甲.png"), ids[0])
        kept = next(doc_id for doc_id in ids if doc_id != doomed)
        field_path = next(iter(json.loads(
            (out_dir / doomed / f"{doomed}.json").read_text(encoding="utf-8"))["fields"]))
        # 资料标识可能是中文，URL 里必须转义，否则请求行直接编不出来。
        doomed_url, kept_url = quote(doomed), quote(kept)
        _post(port, f"/api/review/{doomed_url}",
              {"field_path": field_path, "action": "accept", "reviewer": "acceptance"})
        stale = Path(cfg["paths"]["work"]) / "trash" / "19990101-000000"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "很久以前删的.png").write_bytes(blob)
        status, payload = _post(port, f"/api/document/{doomed_url}/remove", {})
        trash = Path(payload.get("trash") or "")
        moved = {Path(item).name for item in payload.get("moved") or []}
        remaining = (payload.get("summary") or {}).get("run", {}).get("document_count")
        report.check("ingest.flow_remove",
                     status == 200 and payload.get("remaining") == 1 and remaining == 1
                     and not (out_dir / doomed).exists() and (out_dir / kept).is_dir(),
                     f"status={status} remaining={remaining}")
        report.check("ingest.flow_trash_retention_applied",
                     payload.get("pruned") == [stale.name] and not stale.exists()
                     and trash.is_dir() and (trash / "很久以前删的.png").exists() is False,
                     f"pruned={payload.get('pruned')}")
        trashed = {item.name for item in trash.rglob("*") if item.is_file()} if trash.is_dir() else set()
        report.check("ingest.flow_trash_is_reversible",
                     trash.is_dir() and {f"{doomed}.json", f"{doomed}.review.jsonl"} <= trashed
                     and f"{doomed}" in moved and "验收甲.png" in moved,
                     f"trashed={len(trashed)} moved={len(moved)}")

        status, payload = _post(port, f"/api/document/{kept_url}/remove", {})
        inbox_left = sorted(item.name for item in Path(cfg["paths"]["inbox"]).glob("*"))
        report.check("ingest.flow_empty_after_last_removal",
                     status == 200 and payload.get("remaining") == 0
                     and not (out_dir / "summary.json").exists() and not inbox_left
                     and not (out_dir / "final" / "查看结果.html").exists(),
                     f"remaining={payload.get('remaining')} inbox={inbox_left} "
                     f"stale_report={(out_dir / 'final' / '查看结果.html').exists()}")

        # 压缩包：只把里面真能识别的资料放进收件箱，条目路径压平，压缩包本身不落盘。
        packed = _zip_bytes([
            ("第一页/甲.png", blob),
            ("../../越界.png", _png_variant(blob, (240, 245, 250))),
            ("说明.txt", b"notes"),
            ("内层.zip", _zip_bytes([("再里面.png", blob)])),
        ])
        status, payload = _post_bytes(port, f"/api/import?filename={quote('资料包.zip')}", packed)
        inbox_after = sorted(item.name for item in Path(cfg["paths"]["inbox"]).glob("*"))
        report.check("ingest.zip_unpack",
                     status == 200 and payload.get("extracted_count") == 2
                     and payload.get("skipped") == 2
                     and inbox_after == sorted(["甲.png", "越界.png"])
                     and not any(item.suffix == ".zip" for item in
                                 Path(cfg["paths"]["inbox"]).glob("*")),
                     f"status={status} extracted={payload.get('extracted')} skipped={payload.get('skipped')}")
        report.check("ingest.zip_stays_inside_inbox",
                     not list(temp_root.glob("*.png")) and not list(Path(cfg["paths"]["work"]).glob("*.png")),
                     f"stray={len(list(temp_root.glob('*.png')))}")
        status, payload = _post(port, "/api/import/run", {})
        zip_ids = sorted(payload.get("document_ids") or [])
        report.check("ingest.zip_documents_usable",
                     status == 200 and len(zip_ids) == 2
                     and all((out_dir / doc_id / f"{doc_id}.json").exists() for doc_id in zip_ids),
                     f"documents={zip_ids}")
        bomb = _zip_bytes([("炸弹.png", PNG_MAGIC + b"\x00" * (3 * 1024 * 1024))], zipfile.ZIP_DEFLATED)
        refusal = None
        try:
            _post_bytes(port, f"/api/import?filename={quote('炸弹.zip')}", bomb)
        except urllib.error.HTTPError as exc:
            refusal = json.loads(exc.read().decode("utf-8")).get("error")
        report.check("ingest.zip_bomb_refused",
                     bool(refusal) and "疑似异常" in refusal
                     and not (Path(cfg["paths"]["inbox"]) / "炸弹.png").exists(),
                     f"error={refusal}")

        _post(port, "/api/workspace", {"id": "dev"})
        sample_ids = [item["document_id"] for item in
                      (json.loads((ROOT / "out" / "dev" / "summary.json").read_text(
                          encoding="utf-8")).get("documents") or [])]
        refusal = None
        try:
            _post(port, f"/api/document/{quote(sample_ids[0])}/remove", {})
        except urllib.error.HTTPError as exc:
            refusal = json.loads(exc.read().decode("utf-8")).get("error")
        report.check("ingest.flow_sample_area_is_protected",
                     bool(refusal) and "我的资料" in refusal, f"error={(refusal or '')[:80]}")
    except Exception as exc:
        report.check("ingest.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=3)
        shutil.rmtree(temp_root, ignore_errors=True)


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_seal(canvas):
    from PIL import Image, ImageDraw

    sealed = canvas.copy()
    ImageDraw.Draw(sealed).ellipse(
        [int(sealed.width * 0.3), int(sealed.height * 0.24),
         int(sealed.width * 0.7), int(sealed.height * 0.6)],
        outline=(205, 30, 45), width=max(6, sealed.width // 90))
    return sealed


def _draw_edge_seal(canvas):
    """A seal running off the bottom-left corner: only a crescent is in frame."""
    from PIL import ImageDraw

    clipped = canvas.copy()
    ImageDraw.Draw(clipped).ellipse(
        [-int(clipped.width * 0.10), int(clipped.height * 0.80),
         int(clipped.width * 0.20), int(clipped.height * 1.10)],
        outline=(205, 30, 45), width=max(6, clipped.width // 90))
    return clipped


def _colored_ratio(image, chroma_threshold: int) -> float:
    """Share of coloured pixels the seal gate would see, computed like preprocess does."""
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    spread = array.max(axis=2) - array.min(axis=2)
    return float((spread > chroma_threshold).mean())


def _uneven_lighting(canvas, low: float = 0.80, high: float = 0.95):
    """A photographed page is never evenly lit: the lamp slops across it in a gradient."""
    from PIL import Image

    array = np.asarray(canvas.convert("RGB"), dtype=np.float32)
    ramp = np.linspace(low, high, array.shape[1], dtype=np.float32)
    return Image.fromarray(
        np.clip(array * ramp[None, :, None], 0, 255).astype(np.uint8), "RGB")


def _dark_photo(canvas, scale: float, offset: float):
    """A page shot with the lamp behind it: the whole sheet is dim, not just one band."""
    return canvas.point(lambda value: int(value * scale) + int(offset))


def _glare_photo(canvas, alpha: float = 0.90):
    """A specular band over the left of the page: the ink underneath is still there, just washed out."""
    from PIL import Image, ImageDraw

    photo = _uneven_lighting(canvas)
    band = Image.new("L", photo.size, 0)
    ImageDraw.Draw(band).polygon(
        [(int(photo.width * 0.30), 0), (int(photo.width * 0.55), 0),
         (int(photo.width * 0.20), photo.height), (0, photo.height)], fill=255)
    return Image.composite(Image.new("RGB", photo.size, (255, 255, 255)), photo,
                           band.point(lambda value: int(value * alpha)))


def _finger_photo(canvas):
    """The shooter's own finger across the bottom-left corner."""
    from PIL import ImageDraw

    photo = _uneven_lighting(canvas)
    ImageDraw.Draw(photo).line(
        [(-int(photo.width * 0.05), int(photo.height * 1.02)),
         (int(photo.width * 0.34), int(photo.height * 0.42))],
        fill=(223, 177, 148), width=max(8, int(photo.width * 0.085)), joint="curve")
    return photo


def _check_config_thresholds(report: Report, config: Dict) -> None:
    """铁律「所有阈值和规则可配置」的两条可复查证据。

    1) 改配置真的会移动判据（不是只在 YAML 里写个数字好看）；
    2) 配置里不许留没有代码读取的键——装饰性配置键骗过人两次（ocr_engine 起初没人读，
       min_page_quality 写了几轮而判决处硬编码 0.5）。
    这两条都不设开关：把「检查阈值是否可配」做成可关闭的，就等于把铁律本身可关闭。
    """
    import copy as _copy

    from .trust import Calibrator, decide_field

    field = {
        "path": "bs.opening_balance", "label": "期初余额", "value": "100.00",
        "value_type": "decimal", "required": True, "risk": "monetary",
        "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
        "extractor_id": "table_cell:期初余额",
        "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                    "table_score": 0.4, "cross_field_score": 1.0, "grounding_present": 1},
        "meta": {}, "decision": "review", "validators": [], "review_reasons": [],
    }
    parsed = {"page_count": 1, "quality": {"parse_score": 0.45},
              "pages": [{"page": 1, "width": 200, "height": 200}]}
    doc = {"document_id": "CFG1", "doc_type": "BANK_STATEMENT"}

    def reasons(tweaks: Dict) -> List[str]:
        tuned = _copy.deepcopy(config)
        tuned["calibration"]["artifact"] = None
        for section, values in tweaks.items():
            tuned[section].update(values)
        fresh = json.loads(json.dumps(field))
        return decide_field(fresh, doc, parsed, tuned, [],
                            Calibrator(tuned))["review_reasons"]

    default = reasons({})
    page_loose = reasons({"parsing": {"min_page_quality": 0.2}})
    table_loose = reasons({"validation": {"min_table_score": 0.2}})
    report.check("config_keys.page_quality_threshold_moves_the_gate",
                 "PARSE_LOW_QUALITY" in default and "PARSE_LOW_QUALITY" not in page_loose
                 and "TEMPLATE_MISMATCH" in page_loose,
                 f"default={default} page_loose={page_loose}")
    report.check("config_keys.table_score_threshold_moves_the_gate",
                 "TEMPLATE_MISMATCH" in default and "TEMPLATE_MISMATCH" not in table_loose
                 and "PARSE_LOW_QUALITY" in table_loose,
                 f"default={default} table_loose={table_loose}")

    # OCR 置信度这条线曾经在判决层（trust.py 读 low_ocr_threshold）和归因层
    # （evaluate.error_category 硬写 0.9）各写一遍：改配置只移动判决，错误归因
    # 还停在旧数上。这里用同一个 0.7 的 ocr_conf，只抬这一个键，两处必须同时翻面。
    from .evaluate import error_category
    ocr_field = json.loads(json.dumps(field))
    ocr_field["signals"]["ocr_conf"] = 0.7
    ocr_record = {"pred": "100.00", "extractor_id": "table_cell:期初余额",
                  "reasons": [], "signals": json.loads(json.dumps(ocr_field["signals"]))}

    def ocr_pair(threshold: Optional[float] = None):
        tuned = _copy.deepcopy(config)
        tuned["calibration"]["artifact"] = None
        if threshold is not None:
            tuned["validation"]["low_ocr_threshold"] = threshold
        verdict = decide_field(json.loads(json.dumps(ocr_field)), doc, parsed, tuned, [],
                               Calibrator(tuned))["review_reasons"]
        return ("LOW_OCR" in verdict), error_category(ocr_record, tuned)

    shipped = float(config["validation"]["low_ocr_threshold"])
    loose_gate, loose_attr = ocr_pair()
    strict_gate, strict_attr = ocr_pair(0.8)
    report.check("config_keys.low_ocr_threshold_moves_the_attribution",
                 not loose_gate and loose_attr != "OCR"
                 and strict_gate and strict_attr == "OCR",
                 f"{shipped}=({loose_gate},{loose_attr}) 0.80=({strict_gate},{strict_attr})")

    # 兜底默认值也是「同一个数的另一处字面量」：YAML 缺键时由它决定行为，和出厂值
    # 不同就说明「配置缺失」与「配置出厂」是两种产品。标量兜底必须等于出厂值；
    # 容器/None 兜底只要求这个键在配置里存在（没有值可逐字比）。
    from .configaudit import audit as _audit_defaults
    fallback = _audit_defaults(ROOT, config)
    # 比对按配置路径做：出厂配置里有 23 个叶子键名在不同段下取值不同，只按键名比会
    # 让「段定位错」的兜底值撞上另一段的同值而放行。解析不到段的接收者必须如实记账，
    # 现在只剩 2 处（config.py 里下标是变量的两处），涨了就是出现了新的间接访问形态。
    report.check("config_keys.code_defaults_match_shipped_config",
                 not fallback["mismatches"] and not fallback["unknown_keys"]
                 and fallback["checked"] > 200 and len(fallback["unresolved"]) <= 2,
                 f"checked={fallback['checked']} "
                 f"mismatches={fallback['mismatches'][:3]} "
                 f"unknown={fallback['unknown_keys'][:3]} "
                 f"unresolved={len(fallback['unresolved'])}")

    # 文档类型判定有两道门（绝对分、与第二名的差距）和一条置信度公式，过去 3 / 1 /
    # 0.55+0.04×分 全硬写在 parse.py 里。这里同样用隔离法：默认配置下必须认出类型，
    # 只抬 min_margin、只抬 min_score 各一次，都必须退回 UNKNOWN——两次都动才不会
    # 出现「改一个数两个门一起亮，看不出坏的是哪一条」。
    from .parse import detect_doc_type
    probe = [{"text": "银行流水 交易明细 账号 户名 期末余额"}]
    typed, typed_conf = detect_doc_type(probe, config)
    strict_margin = _copy.deepcopy(config)
    strict_margin["parsing"]["doc_type"]["min_margin"] = 99
    margin_type, _ = detect_doc_type(probe, strict_margin)
    strict_score = _copy.deepcopy(config)
    strict_score["parsing"]["doc_type"]["min_score"] = 99
    score_type, _ = detect_doc_type(probe, strict_score)
    report.check("config_keys.doc_type_threshold_moves_the_gate",
                 typed == "BANK_STATEMENT" and margin_type == "UNKNOWN"
                 and score_type == "UNKNOWN",
                 f"default=({typed},{typed_conf}) strict_margin={margin_type} "
                 f"strict_score={score_type}")

    sources = "\n".join(path.read_text(encoding="utf-8")
                        for path in sorted((ROOT / "docpipe").glob("*.py")))
    dead = []
    for section in ("parsing", "validation", "output", "security",
                                "cross_document", "evaluation"):
        for key, value in (config.get(section) or {}).items():
            # code_patterns 按文档类型、cross_document.rules 按规则 id 索引：这些键名是数据，
            # 不是「有没有人读的判决开关」，展开成键名检查只会把注册表本身报成死键。
            names = [key] + (list(value) if isinstance(value, dict)
                             and key not in ("code_patterns", "rules") else [])
            dead += [f"{section}.{name}" for name in names
                     if f'"{name}"' not in sources and f"'{name}'" not in sources]
    report.check("config_keys.no_unread_judgement_keys", not dead,
                 "dead=" + (",".join(dead) or "none"))


def _check_schema_paths(report: Report, config: Dict) -> None:
    """SCENARIO 第 102 行「字段路径一经发布不得随实现变化」的静态门槛。

    改名在流水线上一个字都不会报：台账按旧名记着，产物里已是新名，于是那条人工
    修正对不上任何格子——导出侧 `records.get(path)` 取不到、折池侧 `if not field`
    直接 continue，两处都静默。基线 `config/schema_paths.json` 把发布过的名字冻住，
    改名当场变成 `missing`（旧名还在基线里），新名变成 `added` 逼人显式登记。

    反向证明同一处做：把一条已发布路径改名后，旧名必须出现在 `missing`、新名必须
    出现在 `added`——扫描器哪天静默返回空表，这条会先红。
    """
    from .schemapaths import audit as _audit_paths, compare as _compare

    result = _audit_paths(config)
    problems = []
    if result["problem"]:
        problems.append(result["problem"])
    problems += [f"改名或删除:{name}" for name in result["missing"]]
    problems += [f"未登记的新增:{name}" for name in result["added"]]
    report.check("schema.field_paths_append_only", not problems,
                 f"paths={len(result['shipped'])} "
                 f"baseline={len(result['baseline'] or [])} "
                 f"problems={problems[:4]}")

    sample = (result["shipped"] or ["bl.company_name"])[0]
    old = sample.rsplit(".", 1)[0] + ".renamed_probe"
    bitten = _compare([sample], [old])
    report.check("schema.field_paths_rename_is_named",
                 bitten["missing"] == [old] and bitten["added"] == [sample],
                 f"missing={bitten['missing']} added={bitten['added']}")


def _check_offline(report: Report, config: Dict) -> None:
    """铁律「全链路本地/离线」的静态门槛：交付链路上不许有任何走出本机的入口。

    只读源码、不跑流水线、不写 `out/`。四类检查：三类判据各自为空、以及**反向**证明
    这套扫描真会咬人（最弱的环节是「扫描器哪天静默返回空列表也照样全绿」）。
    """
    import tempfile
    from .offlineaudit import DEFAULTS, audit as _offline_audit

    result = _offline_audit(ROOT, config)
    literals, calls, refs = (result["remote_literals"], result["network_call_sites"],
                             result["remote_asset_refs"])
    report.check("offline.no_remote_url_literals", not literals,
                 f"scanned={result['scanned']} " +
                 ("clean" if not literals else
                  ",".join(f"{item['file']}:{item['line']}{item['url']}" for item in literals[:3])))
    report.check("offline.no_unallowlisted_network_calls", not calls,
                 "clean" if not calls else
                 ",".join(f"{item['file']}:{item['line']}{item['call']}" for item in calls[:3]))
    report.check("offline.delivery_pages_are_self_contained", not refs,
                 "clean" if not refs else
                 ",".join(f"{item['file']}:{item['ref'][:40]}" for item in refs[:3]))
    # 扫描真跑到了东西：docpipe 下十几个模块 + 界面文件，数量掉下这个门就是 glob 形态变了
    # 或者目录被搬走——那会让上面三条「0 违规」变成「0 文件」的假绿。
    report.check("offline.audit_covers_the_delivery_surface", result["scanned"] >= 20,
                 f"scanned={result['scanned']}")
    # 白名单本身也是判决：允许联网的模块一旦被人塞进流水线文件，前三条照样全绿。
    allowed = {str(item) for item in
               ((config.get("security") or {}).get("offline") or {})
               .get("network_allowed_modules", [])}
    pipeline = {"parse.py", "extract.py", "trust.py", "report.py", "run.py", "serve.py"}
    report.check("offline.network_allowlist_excludes_pipeline",
                 allowed and not (allowed & pipeline), f"allowed={sorted(allowed)}")

    with tempfile.TemporaryDirectory() as tmp:
        # 扫描器要能咬人，但它自己不许留任何豁免口子（没有文件级/行级白名单）——所以这里
        # 的远程地址是运行时拼出来的，源码里没有任何一个「协议://主机」字面量。
        # 这条约束本身就是被这次自查逼出来的：写死字面量时，第 1 类判据立刻把验收自己的
        # 夹具报成违规（scanned=26 里 acceptance.py 也在扫）。
        root = Path(tmp)
        (root / "docpipe" / "web").mkdir(parents=True)
        sep = "://"
        plant_css = "https" + sep + "cdn.example.com/font.css"
        plant_link = "http" + sep + "cdn.example.com/a.css"
        plant_socket = "10.0.0.1"
        (root / "docpipe" / "plant.py").write_text(
            f'FONT = "{plant_css}"\n'
            'import socket\n'
            'def probe():\n'
            f'    return socket.create_connection(("{plant_socket}", 9))\n',
            encoding="utf-8")
        (root / "docpipe" / "web" / "index.html").write_text(
            f'<link href="{plant_link}">'
            '<img src="//evil' + '.example.com/b.png">'
            '<svg xmlns="http' + sep + 'www.w3.org/2000/svg"></svg>',
            encoding="utf-8")
        planted = _offline_audit(root, config)
        # 命名空间声明（xmlns）必须被豁免，否则真实仓库里 calibrate.py 画 SVG 就过不了；
        # 但同一段里真会取远程资源的 href 仍要报——所以豁免只能收窄到属性名。
        report.check("offline.audit_bites_on_planted_violations",
                     any(item["url"] == f"https{sep}cdn.example.com"
                         for item in planted["remote_literals"])
                     and any(item["call"].startswith("import socket")
                             for item in planted["network_call_sites"])
                     and any(item["url"] == f"http{sep}cdn.example.com"
                             for item in planted["remote_asset_refs"])
                     and any(item["url"].startswith("//evil.example.com")
                             for item in planted["remote_asset_refs"])
                     and not any("www.w3.org" in item["url"]
                                 for item in planted["remote_asset_refs"])
                     and not any("www.w3.org" in item["url"]
                                 for item in planted["remote_literals"]),
                     f"literals={len(planted['remote_literals'])} "
                     f"calls={len(planted['network_call_sites'])} "
                     f"refs={[item['url'] for item in planted['remote_asset_refs']]}")

    # 代码兜底默认值与出厂配置必须逐字一致（铁律审计 IV 的同一条规矩，用在新的段上）。
    shipped = ((config.get("security") or {}).get("offline") or {})
    diffs = {key: (DEFAULTS[key], shipped.get(key))
             for key in DEFAULTS
             if isinstance(DEFAULTS[key], (str, int, float))
             and shipped.get(key) != DEFAULTS[key]}
    missing = [key for key in DEFAULTS if key not in shipped]
    report.check("offline.defaults_match_shipped_config",
                 not diffs and not missing,
                 f"diffs={list(diffs.items())[:3]} missing={missing}")


def _check_review_reconcile(report: Report, root: Path, config: Dict) -> None:
    """铁律审计 X：`review/` 台账与 `final/` 成品必须逐格对得上，判据全在 `validation.delivery`。

    这里的树是复核台 API 写完台账、`export()` 应用过修正的真产物：写台账的和改取值的是两套
    代码，两边对上才算对上。每个反向用例只种一个破口，一条 FAIL 只指向一个判决。
    """
    from .deliveryaudit import final_documents as _final_documents
    from .deliveryaudit import options as _delivery_options
    from .deliveryaudit import reconcile as _reconcile
    from .deliveryaudit import review_ledger as _review_ledger

    opts = _delivery_options(config)
    clean = _reconcile(root, opts)
    report.check("delivery.reviewed_trail_reconciles_the_delivered_values",
                 not clean["problems"] and not clean["mismatches"]
                 and clean["documents"] >= 1 and clean["record_count"] >= 1,
                 f"documents={clean['documents']} records={clean['record_count']} "
                 f"cells={clean['cells']} problems={clean['problems'][:3]} "
                 f"mismatches={clean['mismatches'][:3]}")
    # 状态格与修正格的样本：后面每个反向用例都只动其中一处。
    status_docs = [(doc, [path for path, field in doc["fields"].items()
                          if isinstance(field, dict) and field.get("review_status")])
                   for doc in _final_documents(root)]
    status_docs = [(doc, paths) for doc, paths in status_docs if paths]
    corrected = [(doc, path) for doc, _ in status_docs
                 for path, field in doc["fields"].items()
                 if isinstance(field, dict) and field.get("review_status") == "reviewed_corrected"
                 and field.get("effective_value") != field.get("value")]
    if not report.check("delivery.reconcile_has_a_trail_to_bite",
                        bool(status_docs) and bool(corrected),
                        "status_docs=" + str([(doc["document_id"], len(paths))
                                               for doc, paths in status_docs[:3]])
                        + " corrected=" + str([(doc["document_id"], path)
                                               for doc, path in corrected[:3]])):
        return
    doc_id, wrong_path = status_docs[0][0]["document_id"], status_docs[0][1][0]
    fixed_doc, fixed_path = corrected[0]

    with tempfile.TemporaryDirectory() as tmp:
        def scratch(tag, skip=("render",)):
            target = Path(tmp) / tag
            shutil.copytree(root, target, ignore=shutil.ignore_patterns(*skip))
            return target

        def rewrite_final(tree, document_id):
            file = tree / "final" / document_id / f"{document_id}.json"
            return file, json.loads(file.read_text(encoding="utf-8"))

        def append_records(tree, document_id, extra):
            ledger = tree / "review" / f"{document_id}.review.jsonl"
            with ledger.open("a", encoding="utf-8") as handle:
                for record in extra:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        def record(document_id, path, action, **rest):
            return dict({"document_id": document_id, "field_path": path,
                         "original_value": "种下去的原文", "corrected_value": None,
                         "action": action, "reviewer": "acceptance",
                         "timestamp": now_iso()}, **rest)

        # 1) 成品宣称已复核，台账里却查不到这条：签字丢了。
        tree = scratch("unbacked")
        (tree / "review" / f"{doc_id}.review.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in _review_ledger(tree, doc_id)
                    if item.get("field_path") != wrong_path), encoding="utf-8")
        bitten = _reconcile(tree, opts)
        report.check("delivery.reconcile_bites_on_an_unbacked_status",
                     [item["problem"] for item in bitten["problems"]] == ["unbacked_status"]
                     and bitten["problems"][0]["path"] == wrong_path
                     and not bitten["mismatches"],
                     f"path={wrong_path} problems={bitten['problems'][:2]} "
                     f"mismatches={bitten['mismatches'][:2]}")

        # 2) 台账写了修正值，成品却还发原值：这一格等于没改。
        tree = scratch("unapplied")
        file, payload = rewrite_final(tree, fixed_doc["document_id"])
        payload["fields"][fixed_path]["effective_value"] = \
            payload["fields"][fixed_path]["value"]
        file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        bitten = _reconcile(tree, opts)
        report.check("delivery.reconcile_bites_on_a_correction_that_never_landed",
                     [item["problem"] for item in bitten["problems"]] == ["unapplied_correction"]
                     and bitten["problems"][0]["path"] == fixed_path,
                     f"path={fixed_path} problems={bitten['problems'][:2]}")

        # 3) 同一格复核了两次：成品必须跟最后一条，跟的是更早那条就是过期值。
        tree = scratch("stale")
        append_records(tree, fixed_doc["document_id"],
                       [record(fixed_doc["document_id"], fixed_path, "correct",
                               corrected_value="__更晚一条修正__")])
        bitten = _reconcile(tree, opts)
        report.check("delivery.reconcile_follows_the_last_ledger_record",
                     [item["problem"] for item in bitten["problems"]] == ["stale_winner"]
                     and "更早一条" in str(bitten["problems"][0].get("note")),
                     f"problems={bitten['problems'][:2]}")

        # 4) 台账里签了一格成品中不存在的路径：字段被改名或删掉了，账却没跟着清。
        tree = scratch("ghost")
        append_records(tree, doc_id, [record(doc_id, "zz.成品里没有这格", "accept")])
        bitten = _reconcile(tree, opts)
        report.check("delivery.reconcile_bites_on_a_record_without_a_cell",
                     [item["problem"] for item in bitten["problems"]] == ["record_without_cell"]
                     and "字段不在成品里" in str(bitten["problems"][0]["actual"]),
                     f"problems={bitten['problems'][:2]}")

        # 5) 动作与状态一对一：词汇表换掉，成品里的状态必须全体变成「越表」。
        renamed = _reconcile(root, dict(opts, review_state_pairs={
            "accept": "approved", "correct": "amended"}))
        status_cells = sum(len(paths) for _, paths in status_docs)
        report.check("delivery.reconcile_vocabulary_follows_the_config",
                     all(item["problem"] == "unknown_state" for item in renamed["problems"])
                     and len(renamed["problems"]) == status_cells
                     and not clean["problems"] and status_cells > 0,
                     f"status_cells={status_cells} surfaced={len(renamed['problems'])} "
                     f"first={renamed['problems'][:1]}")

        # 6) 整套词汇表改名（配对、改值动作、计数键一起翻）后重新导出：成品与审计必须跟着
        #    一起翻面。配置是唯一定义词——代码里再留一套出厂词，就会拿旧词去指控合法的人工修正。
        tree = scratch("renamed")
        renamed_cfg = copy.deepcopy(config)
        renamed_cfg.setdefault("paths", {})["work"] = str(tree / "work")
        delivery = renamed_cfg["validation"]["delivery"]
        delivery["review_state_pairs"] = {"accept": "reviewed_ok", "correct": "reviewed_edit"}
        delivery["value_changing_actions"] = ["correct"]
        delivery["review_count_keys"] = {"reviewed_count": None,
                                         "reviewed_corrected_count": "reviewed_edit"}
        renamed_overall = export(tree, tree / "final", renamed_cfg)["overall"]
        renamed = _reconcile(tree, _delivery_options(renamed_cfg))
        moved = next((doc["fields"].get(fixed_path) for doc in _final_documents(tree)
                      if doc["document_id"] == fixed_doc["document_id"]), {})
        shipped_corrected = int((json.loads(
            (root / "final" / "final_summary.json").read_text(encoding="utf-8"))
            .get("overall") or {}).get("reviewed_corrected_count") or 0)
        report.check("delivery.renaming_the_whole_vocab_moves_export_and_audit_together",
                     (moved or {}).get("review_status") == "reviewed_edit"
                     and moved.get("effective_value") != moved.get("value")
                     and renamed_overall["reviewed_corrected_count"] == shipped_corrected
                     and not renamed["problems"] and not renamed["mismatches"],
                     f"status={(moved or {}).get('review_status')} "
                     f"corrected={renamed_overall['reviewed_corrected_count']}/{shipped_corrected} "
                     f"problems={renamed['problems'][:2]}")

        # 7) 只改一半（value_changing 写了动作表里没有的词）：打架的是词汇表自己，审计必须
        #    先把配置点名，而不是让签字的人独自背「越权改值」。
        bitten = _reconcile(root, dict(opts, value_changing_actions=["amend"]))
        report.check("delivery.reconcile_blames_a_self_contradictory_vocab_first",
                     "value_changing_action_unpaired" in
                     {item["problem"] for item in bitten["problems"]}
                     and bitten["problems"][0]["actual"] == ["amend"],
                     f"first={bitten['problems'][:1]}")

        # 8) 词汇表只有一个主人：出厂 YAML、流水线兜底、审计自带的那份必须逐字相同。
        from .reviews import ACTION_DEFAULTS
        shipped = (config.get("validation") or {}).get("delivery") or {}
        report.check("delivery.review_vocab_is_not_written_twice",
                     shipped.get("review_state_pairs") == ACTION_DEFAULTS["review_state_pairs"]
                     and shipped.get("value_changing_actions")
                     == ACTION_DEFAULTS["value_changing_actions"]
                     and opts["review_state_pairs"] == ACTION_DEFAULTS["review_state_pairs"]
                     and opts["value_changing_actions"] == ACTION_DEFAULTS["value_changing_actions"],
                     f"yaml={shipped.get('review_state_pairs')} code={ACTION_DEFAULTS}")

        # 9) 只动汇总、一格都不动：三层计数必须自己发现，且不去乱点某格。
        tree = scratch("split")
        summary_file = tree / "final" / "final_summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        summary["overall"]["reviewed_count"] = int(summary["overall"].get("reviewed_count") or 0) + 3
        summary_file.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        bitten = _reconcile(tree, opts)
        report.check("delivery.reconcile_bites_on_a_count_split",
                     not bitten["problems"]
                     and {"final_summary.overall", "documents_vs_overall"}
                     <= {item["layer"] for item in bitten["mismatches"]}
                     and {item["problem"] for item in bitten["mismatches"]} == {"reviewed_count"},
                     f"problems={bitten['problems'][:2]} mismatches={bitten['mismatches'][:3]}")

        # 10) 从没导出过的树：0 条记录不等于「对得上」，得单独点名「这一问没内容」。
        empty = scratch("no_final", skip=("final", "review", "render"))
        untouched = _reconcile(empty, opts)
        report.check("delivery.no_trail_is_not_a_green_reconciliation",
                     untouched["no_trail"] and untouched["documents"] == 0
                     and not untouched["problems"] and not clean["no_trail"],
                     f"documents={untouched['documents']} no_trail={untouched['no_trail']} "
                     f"cells={untouched['cells']} clean_no_trail={clean['no_trail']}")


def _check_coordinates(report: Report, out_dir: Path, config: Dict) -> None:
    """铁律审计 XI：有值的格子，前台那两层真的画得出方框吗？判据全在 `validation.delivery`。

    `output.accepted_grounding` 查的是原始框 `bbox` 四元、面积为正；复核台与单文件网页版摆的
    却是 `bbox_norm`（缺失时那格被静默过滤，看不见证据也不报错）。这里补的就是这一层依赖。
    每个反向用例在临时副本上只种一个破口，`out/dev`、`out/holdout` 本身一个字节都不动。
    """
    from .deliveryaudit import coordinates as _coordinates
    from .deliveryaudit import document_files as _document_files
    from .deliveryaudit import options as _delivery_options

    opts = _delivery_options(config)
    clean = _coordinates(out_dir, opts)
    report.check("delivery.coordinates_are_drawable",
                 not clean["offenders"] and clean["counts"]["cells"] >= 1
                 and clean["counts"]["audited"] == clean["counts"]["cells"],
                 f"cells={clean['counts']['cells']} audited={clean['counts']['audited']} "
                 f"accepted={clean['counts']['accepted']} "
                 f"offenders={clean['offenders'][:3]}")

    def planted(tag):
        target = Path(temp_dir) / tag
        shutil.copytree(out_dir, target,
                        ignore=shutil.ignore_patterns("final", "review", "render"))
        return target

    def rewrite(tree, mutate):
        victim = _document_files(tree)[0]
        payload = json.loads(victim.read_text(encoding="utf-8"))
        path = next(iter(payload["fields"]))
        mutate(payload["fields"][path])
        victim.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(payload.get("document_id")), path

    temp_dir = tempfile.mkdtemp()
    try:
        # 1) 前台那套坐标没写出来：格子有值、有判决、有原始框，界面却什么框都不摆。
        tree = planted("absent")
        doc_id, path = rewrite(tree, lambda field: field["grounding"].__setitem__(
            "bbox_norm", None))
        bitten = _coordinates(tree, opts)
        report.check("delivery.coordinates_bite_on_a_missing_norm",
                     [item["problem"] for item in bitten["offenders"]] == ["norm_absent"]
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} offenders={bitten['offenders'][:2]}")

        # 2) 归一化框落到页面外：百分比方框画到视口外面，鼠标怎么点都点不到那一格。
        #    把原始框左上角也贴到 0，这一格才只暴露「画到页面外」一条——贴 0 时推不出页面
        #    尺度，坐标一致性那条按「看不见证据不下结论」的规矩本来就不开口。
        tree = planted("bounds")
        doc_id, path = rewrite(tree, lambda field: field["grounding"].__setitem__(
            "bbox", [0.0, 0.0, float(field["grounding"]["bbox"][2]),
                     float(field["grounding"]["bbox"][3])]))
        rewrite(tree, lambda field: field["grounding"].__setitem__(
            "bbox_norm", [0.0, 0.0, 1.6, 2.0]))
        bitten = _coordinates(tree, opts)
        report.check("delivery.coordinates_bite_on_an_off_page_frame",
                     [item["problem"] for item in bitten["offenders"]] == ["norm_out_of_bounds"]
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} offenders={bitten['offenders'][:2]}")

        # 3) 两套坐标打架：只改了 `bbox` 没重算 `bbox_norm`，纵边推不出同一个页面尺度。
        tree = planted("scale")
        doc_id, path = rewrite(tree, lambda field: field["grounding"].__setitem__(
            "bbox", [float(field["grounding"]["bbox"][0]),
                     float(field["grounding"]["bbox"][1]) * 3.0,
                     float(field["grounding"]["bbox"][2]),
                     float(field["grounding"]["bbox"][3]) * 4.0]))
        bitten = _coordinates(tree, opts)
        report.check("delivery.coordinates_bite_on_a_scale_conflict",
                     [item["problem"] for item in bitten["offenders"]] == ["scale_conflict"]
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} offenders={bitten['offenders'][:2]}")

        # 4) 页码越界：前台按 `grounding.page === 当前页` 过滤，越界的格子在任何一页都不出现。
        tree = planted("page")
        doc_id, path = rewrite(tree, lambda field: field["grounding"].__setitem__("page", 9999))
        bitten = _coordinates(tree, opts)
        report.check("delivery.coordinates_bite_on_a_page_out_of_range",
                     [item["problem"] for item in bitten["offenders"]] == ["page_out_of_range"]
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} offenders={bitten['offenders'][:2]}")

        # 5) 画哪套坐标是配置说的：换个键名，全部有值格都该以「没写出来」露头。
        renamed = _coordinates(out_dir, dict(opts, drawn_coordinate_keys=["bbox_display"]))
        report.check("delivery.coordinates_drawn_keys_follow_the_config",
                     {item["problem"] for item in renamed["offenders"]} == {"norm_absent"}
                     and len(renamed["offenders"]) == clean["counts"]["cells"],
                     f"cells={clean['counts']['cells']} surfaced={len(renamed['offenders'])}")

        # 6) 容差是配置说的：出厂那半个刻度就是「四舍五入写进产物」的唯一误差来源，
        #    抽干它，自洽的格子会全体判成打架——否则「0 违规」其实是判据没牙。
        strict = _coordinates(out_dir, dict(opts, coordinate_tolerance=0.0))
        report.check("delivery.coordinate_tolerance_follows_the_config",
                     strict["verdicts"].get("scale_conflict") == clean["counts"]["cells"]
                     and not clean["offenders"] and clean["counts"]["cells"] > 0,
                     f"cells={clean['counts']['cells']} strict={strict['verdicts']} "
                     f"tolerance={opts['coordinate_tolerance']}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _check_delivery(report: Report, out_dir: Path, config: Dict) -> None:
    """铁律审计 VIII：交付的每一格必须带着判决层的痕迹，且三层计数互相对得上。"""
    from .deliveryaudit import DEFAULTS as DELIVERY_DEFAULTS
    from .deliveryaudit import (audit as _delivery_audit, document_files,
                                options as _delivery_options,
                                package_report as _package_report)

    opts = ((config.get("validation") or {}).get("delivery") or {})
    result = _delivery_audit(out_dir, config)
    # 审计自己先要有牙：扫到空目录时「零违规」和「全干净」输出一样，必须算失败。
    report.check("delivery.audit_covers_the_delivery",
                 result["documents"] >= int(opts.get("min_delivered_documents",
                                                     DELIVERY_DEFAULTS["min_delivered_documents"]))
                 and result["fields"] >= int(opts.get("min_delivered_fields",
                                                      DELIVERY_DEFAULTS["min_delivered_fields"])),
                 f"documents={result['documents']} fields={result['fields']} "
                 f"floors={opts.get('min_delivered_documents')},"
                 f"{opts.get('min_delivered_fields')}")
    trace_detail = [(item["document_id"], item["offender_count"], item["offenders"][:1])
                    for item in result["undecided"][:3]]
    report.check("delivery.every_field_carries_a_decision",
                 not result["undecided"],
                 f"documents={result['documents']} fields={result['fields']} "
                 f"offenders={trace_detail}")
    report.check("delivery.counts_agree_across_layers",
                 not result["mismatches"],
                 f"mismatches={result['mismatches'][:5]}")

    # CSV 出口不许比 JSON 少一层：行数与状态分布必须和逐格审计同一份账。
    csv_rows = list(csv.DictReader((out_dir / "fields.csv").open(encoding="utf-8-sig")))
    states = [str(state) for state in opts.get("decided_states",
                                               DELIVERY_DEFAULTS["decided_states"])]
    from_csv = {state: sum(1 for row in csv_rows if row.get("decision") == state)
                for state in states}
    report.check("delivery.csv_outlet_carries_the_same_decision",
                 len(csv_rows) == result["fields"]
                 and all(from_csv[state] == result["by_state"][state] for state in states)
                 and sum(from_csv.values()) == len(csv_rows),
                 f"rows={len(csv_rows)} by_state={from_csv} json={result['by_state']}")

    # 反向用例：交付面被动过一手，审计必须当场看见（两处分开种，才知道是哪条判据在咬）。
    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp) / "out"
        shutil.copytree(out_dir, planted,
                        ignore=shutil.ignore_patterns("final", "review", "render"))
        victim = document_files(planted)[0]
        payload = json.loads(victim.read_text(encoding="utf-8"))
        path = next(iter(payload["fields"]))
        del payload["fields"][path]["decision"]
        victim.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        bitten = _delivery_audit(planted, config)
        report.check("delivery.audit_bites_on_a_field_without_a_decision",
                     any(item["document_id"] == payload.get("document_id")
                         and any(off.get("problem") == "missing_keys"
                                 and "decision" in off.get("missing", [])
                                 for off in item["offenders"])
                         for item in bitten["undecided"])
                     and any(item["layer"] == "summary.documents" and item["problem"] == "accept"
                             for item in bitten["mismatches"]),
                     f"undecided={bitten['undecided'][:1]} mismatches={bitten['mismatches'][:2]}")

        shutil.rmtree(planted)
        shutil.copytree(out_dir, planted,
                        ignore=shutil.ignore_patterns("final", "review", "render"))
        summary_path = planted / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # 只动汇总、一格明细都不动：这是「明细一份、汇总另一份」那种分裂的最小可复现版本。
        summary["documents"][0]["accept_count"] += 5
        summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        bumped = _delivery_audit(planted, config)
        report.check("delivery.audit_bites_on_summary_only_counts",
                     not bumped["undecided"]
                     and any(item["document_id"] == summary["documents"][0]["document_id"]
                             and item["layer"] == "summary.documents"
                             and item["problem"] == "accept"
                             and item["expected"] - item["actual"] == 5
                             for item in bumped["mismatches"])
                     and any(item["layer"] == "documents_vs_overall"
                             for item in bumped["mismatches"]),
                     f"undecided={len(bumped['undecided'])} "
                     f"mismatches={bumped['mismatches'][:3]}")

    # 判决词汇表跟着配置走：只承认 accept 时，被复核的那几格必须以「陌生状态」露头，
    # 说明这条判据读的是 decided_states，不是代码里写死的两个词。
    narrowed = copy.deepcopy(config)
    narrowed.setdefault("validation", {})["delivery"] = dict(opts, decided_states=["accept"])
    strict = _delivery_audit(out_dir, narrowed)
    review_count = int((json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
                        .get("overall") or {}).get("review_count", 0))
    surfaced = sum(item["offender_count"] for item in strict["undecided"])
    sampled = [off for item in strict["undecided"] for off in item["offenders"]]
    report.check("delivery.state_vocabulary_follows_the_config",
                 review_count > 0 and surfaced == review_count
                 and all(off.get("problem") == "unknown_state"
                         and off.get("decision") == "review" for off in sampled)
                 and not result["undecided"]
                 and not strict["mismatches"],
                 f"review_count={review_count} surfaced={surfaced} "
                 f"mismatches={strict['mismatches'][:2]} "
                 f"decided={narrowed['validation']['delivery']['decided_states']}")

    shipped_config = opts
    # 铁律审计 IX：交付包自己必须说得出装了什么。前三条问的是 out/，这四条问的是
    # 真正发给别人的那个 ZIP——final/ 是照着磁盘扫的，遗留文件与原始材料都会混进去。
    # pack_opts 走 deliveryaudit.options：出厂 YAML 少了哪个键就退回代码默认值，
    # 和 deliveryaudit 命令行看到的是同一套判据。
    pack_opts = _delivery_options(config)
    with tempfile.TemporaryDirectory() as tmp:
        packed_root = Path(tmp) / "ws"
        shutil.copytree(out_dir, packed_root,
                        ignore=shutil.ignore_patterns("final", "render"))
        zip_config = copy.deepcopy(config)
        zip_config.setdefault("paths", {})["work"] = str(Path(tmp) / "work")
        shipped = _package_report(Path(export_zip(packed_root, config=zip_config)["path"]),
                                  pack_opts)
        report.check("delivery.package_manifest_lists_every_member",
                     not shipped["missing_keys"] and not shipped["unlisted"]
                     and not shipped["phantom"] and not shipped["bad_digest"]
                     and not shipped["skipped"] and not shipped["thin"]
                     and shipped["member_count"] == shipped["declared_count"],
                     f"members={shipped['member_count']} declared={shipped['declared_count']} "
                     f"floor={pack_opts.get('min_package_members')} "
                     f"missing={shipped['missing_keys']} unlisted={shipped['unlisted'][:3]} "
                     f"phantom={shipped['phantom'][:3]} bad_digest={shipped['bad_digest'][:3]}")

        # 反向：交付目录里混进不属于本轮导出的东西（上一轮的调试表、别人的原始 PDF、
        # 一份不在本次导出清单里的文档台账），打包必须**既不带、又点名**。
        (packed_root / "final" / "上一轮的调试表.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (packed_root / "final" / "原始材料.pdf").write_bytes(b"%PDF-1.4 planted\n%%EOF\n")
        (packed_root / "review" / "ZZ99.review.jsonl").write_text('{"planted":1}\n', encoding="utf-8")
        refused = export_zip(packed_root, config=zip_config)
        with zipfile.ZipFile(refused["path"]) as archive:
            member_names = set(archive.namelist())
        reasons = {str(item.get("name")): str(item.get("reason"))
                   for item in refused["skipped_members"]}
        report.check("delivery.package_refuses_members_not_exported_this_run",
                     not any(name.endswith(("上一轮的调试表.csv", "原始材料.pdf",
                                             "ZZ99.review.jsonl")) for name in member_names)
                     and len(reasons) == 3
                     and reasons.get("final/原始材料.pdf") == "not_exported_this_run"
                     and reasons.get("review/ZZ99.review.jsonl") == "document_not_in_export"
                     and not _package_report(Path(refused["path"]), pack_opts)["missing_keys"],
                     f"members={len(member_names)} skipped={sorted(reasons.items())}")

        # 清单撒谎：把第一个成员从清单里抹掉，包内文件一个没动——必须报「包里有而清单没记」。
        lying = Path(tmp) / "lying.zip"
        with zipfile.ZipFile(refused["path"]) as source, zipfile.ZipFile(lying, "w") as sink:
            for item in source.infolist():
                blob = source.read(item.filename)
                if item.filename == "export_manifest.json":
                    record = json.loads(blob.decode("utf-8"))
                    dropped = record["members"].pop(0)
                    record["member_count"] -= 1
                    blob = json.dumps(record, ensure_ascii=False).encode("utf-8")
                sink.writestr(item, blob)
        bitten = _package_report(lying, pack_opts)
        report.check("delivery.package_bites_on_a_manifest_that_lies",
                     bitten["unlisted"] == [dropped["name"]] and not bitten["phantom"]
                     and not bitten["bad_digest"] and not bitten["missing_keys"]
                     and bitten["member_count"] == bitten["declared_count"],
                     f"dropped={dropped['name']} unlisted={bitten['unlisted'][:3]} "
                     f"phantom={bitten['phantom'][:3]}")

        # 下限必须是配置读的：抬到一万条，同一个包就要判「这个包基本没被审到」，
        # 而清单一致性本身仍然干净——否则「0 违规」其实是「没审到东西」。
        strict = copy.deepcopy(config)
        strict.setdefault("validation", {})["delivery"] = dict(pack_opts, min_package_members=10000)
        tightened = _package_report(Path(refused["path"]), _delivery_options(strict))
        report.check("delivery.package_member_floor_follows_the_config",
                     tightened["thin"] and not tightened["missing_keys"]
                     and not tightened["unlisted"] and not tightened["phantom"],
                     f"members={tightened['member_count']} floor=10000 "
                     f"thin={tightened['thin']} consistent={not tightened['unlisted']}")

    diffs = {key: (DELIVERY_DEFAULTS[key], shipped_config.get(key))
             for key in DELIVERY_DEFAULTS
             if shipped_config.get(key) != DELIVERY_DEFAULTS[key]}
    missing = [key for key in DELIVERY_DEFAULTS if key not in shipped_config]
    report.check("delivery.defaults_match_shipped_config",
                 not diffs and not missing,
                 f"diffs={list(diffs.items())[:3]} missing={missing}")


def _check_gate_explainability(report: Report, out_dir: Path, config: Dict) -> None:
    """铁律审计 XII：交付的每一格「自动接受」，出厂门槛解释得通吗？判据在 `validation.delivery`。

    名字里带上 explainability 是有原因的：`_check_gates` 已经被 `gate.*` 那 25 项占了，
    同名定义会在模块加载时把老的悄悄盖掉——验收项凭空少一截，而这条新腿照样全绿。

    `thresholds` 只在判决层读一次，产物里剩的是 `confidence` 与 `decision` 两个字段。这里拿
    产物自己声明的 `path`/`risk`/`confidence` 独立查一遍配置——不叫 `config.risk_threshold`，
    因为那个函数改了口径这条判据会跟着一起改，两边就永远对得上了。每个反向用例在临时副本上
    只种一个破口，`out/dev`、`out/holdout` 本身一个字节都不动。
    """
    from .deliveryaudit import audit as _delivery_audit
    from .deliveryaudit import document_files as _document_files
    from .deliveryaudit import gates as _gates
    from .deliveryaudit import options as _delivery_options

    opts = _delivery_options(config)
    tiers = {str(key): float(value) for key, value in
             (config["thresholds"].get("risk_tiers") or {}).items()}
    floor = float(config["thresholds"]["global_min_confidence"])
    clean = _gates(out_dir, opts, config)
    by_state = _delivery_audit(out_dir, config)["by_state"]
    report.check("delivery.gates_explain_every_release",
                 not clean["offenders"] and not clean["no_gate"]
                 and clean["counts"]["gated"] == by_state["accept"]
                 and clean["counts"]["cells"] == clean["counts"]["gated"],
                 f"gated={clean['counts']['gated']} cells={clean['counts']['cells']} "
                 f"accept={by_state['accept']} offenders={clean['offenders'][:3]}")

    def planted(tag):
        target = Path(temp_dir) / tag
        shutil.copytree(out_dir, target,
                        ignore=shutil.ignore_patterns("final", "review", "render"))
        return target

    def rewrite(tree, mutate):
        victim = _document_files(tree)[0]
        payload = json.loads(victim.read_text(encoding="utf-8"))
        gated = set(opts["gate_states"])
        path = next(item for item, field in payload["fields"].items()
                    if isinstance(field, dict) and field.get("decision") in gated)
        mutate(payload["fields"][path])
        victim.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(payload.get("document_id")), path

    temp_dir = tempfile.mkdtemp()
    try:
        # 1) 高于全局下限、低于自己那一档：这一格凭什么放行，配置里说不出话。
        tree = planted("tier")
        doc_id, path = rewrite(tree, lambda field: field.__setitem__(
            "confidence", tiers[str(field["risk"])] - 0.02))
        bitten = _gates(tree, opts, config)
        report.check("delivery.gates_bite_on_a_release_below_its_tier",
                     [item["problem"] for item in bitten["offenders"]] == ["below_risk_tier"]
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} offenders={bitten['offenders'][:2]}")

        # 2) 低于全局下限：只该由全局下限那条开口，不许再叠一条档位违规。
        tree = planted("floor")
        doc_id, path = rewrite(tree, lambda field: field.__setitem__("confidence", floor - 0.4))
        bitten = _gates(tree, opts, config)
        report.check("delivery.gates_bite_on_a_release_below_the_global_floor",
                     bitten["verdicts"] == {"below_global_min": 1}
                     and bitten["offenders"][0]["path"] == path,
                     f"document={doc_id} path={path} verdicts={bitten['verdicts']}")

        # 3) 风险档没配门槛：等于按一个配置里根本不存在的档放行。
        tree = planted("class")
        doc_id, path = rewrite(tree, lambda field: field.__setitem__("risk", "secret_sauce"))
        bitten = _gates(tree, opts, config)
        report.check("delivery.gates_bite_on_an_unconfigured_risk_class",
                     bitten["verdicts"] == {"unknown_risk_class": 1}
                     and "secret_sauce" in str(bitten["offenders"]),
                     f"document={doc_id} path={path} verdicts={bitten['verdicts']}")

        # 4) 判决说放行、复核理由却还挂着：两个字段不是同一次写的（accept 分支会清空理由）。
        tree = planted("reasons")
        doc_id, path = rewrite(tree, lambda field: field.__setitem__(
            str(opts["gate_reason_key"]), ["LOW_CONFIDENCE"]))
        bitten = _gates(tree, opts, config)
        report.check("delivery.gates_bite_on_a_release_still_carrying_reasons",
                     bitten["verdicts"] == {"accept_with_review_reasons": 1},
                     f"document={doc_id} path={path} verdicts={bitten['verdicts']}")

        # 5) 字典键与格子里的 `path` 分家：门槛按哪个查说不清，覆盖表就再也落不到那一格上。
        tree = planted("identity")
        doc_id, path = rewrite(tree, lambda field: field.__setitem__("path", "bl.somebody_else"))
        bitten = _gates(tree, opts, config)
        report.check("delivery.gates_bite_on_a_split_cell_identity",
                     bitten["verdicts"] == {"path_identity_split": 1}
                     and bitten["offenders"][0]["path"] == "bl.somebody_else",
                     f"document={doc_id} key={path} verdicts={bitten['verdicts']}")

        # 6) 谁有解释义务是配置说的：换成 `review`，出厂那批复核格子反过来全部参加核对，
        #    放行的格子一条不报——否则 `gate_states` 只是句装饰。同时「必须闭嘴」那一半
        #    不该跟着翻：复核格子天生带着理由，那是它进队列的原因，不是矛盾。第一版把两件
        #    事塞进一个键，翻表时这 3 格（dev）反而全被判成违规，正是要防的那种误报。
        flipped = _gates(out_dir, dict(opts, gate_states=["review"]), config)
        report.check("delivery.gate_states_follow_the_config",
                     flipped["counts"]["gated"] == by_state["review"]
                     and flipped["states"] == ["review"]
                     and "accept_with_review_reasons" not in flipped["verdicts"]
                     and clean["counts"]["gated"] != flipped["counts"]["gated"],
                     f"flipped_gated={flipped['counts']['gated']} "
                     f"review={by_state['review']} verdicts={flipped['verdicts']}")

        # 7) 谁必须不挂理由也是配置说的：把静音表换成 `review`，出厂那批复核格子（每格都带
        #    一条真实理由：SIGN_INFERRED / LOCAL_RECROP）全体露头，而解释义务那一半照旧只管
        #    accept——两张表确实各管各的。
        muted = _gates(out_dir, dict(opts, reason_free_states=["review"]), config)
        report.check("delivery.reason_free_states_follow_the_config",
                     muted["verdicts"] == {"accept_with_review_reasons": by_state["review"]}
                     and sorted(muted["states"]) == ["accept"]
                     and not clean["verdicts"],
                     f"reason_free=review flagged={sum(muted['verdicts'].values())} "
                     f"review={by_state['review']} clean={clean['verdicts']}")

        # 8) 门槛整段读不到时这一问弃权：只数格子、一条不报，退出码不许被它冒充成失败。
        blank = copy.deepcopy(config)
        blank["thresholds"] = {}
        abstained = _gates(out_dir, opts, blank)
        report.check("delivery.gate_leg_abstains_without_thresholds",
                     abstained["no_gate"] and not abstained["offenders"]
                     and abstained["counts"]["gated"] == by_state["accept"]
                     and abstained["counts"]["cells"] == 0,
                     f"no_gate={abstained['no_gate']} gated={abstained['counts']['gated']} "
                     f"offenders={len(abstained['offenders'])}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _check_ledger_write(report: Report, out_dir: Path, config: Dict) -> None:
    """铁律审计 XIII：复核台说「已保存」之前，那条修正必须真的能从台账读回来。

    台账是逐行追加的明文文件。上一行缺换行结尾时（有人手改过、或上一次写被截断），下一次追加会把
    两条记录**粘成一行**，而三个读取侧对坏行的处理是「跳过」或「记账」——于是两条已经签过的人工修正
    一起消失，界面当时却已经说过「已保存」。这里用真实 HTTP 接口走四件事：写完读得回来、缺换行时不
    吞上一条、粘成一行时数得出来、合法 JSON 但不是对象的一行不许交给界面。

    「写完读不回来必须报错」那一半只有单测证明（要把读法换成空只能靠 mock，没有不改代码就能造出来的
    自然形态）——验收不冒充跑过它。
    """
    from .reviews import read_review_ledger

    app_source = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    serve_source = (ROOT / "docpipe" / "serve.py").read_text(encoding="utf-8")
    knobs = [f"{section}.{key}" for section in ("validation", "ui", "ingest", "thresholds")
             for key in (config.get(section) or {})
             if "ledger" in str(key) or "read_back" in str(key)]
    # 这道门刻意没有配置出口：能关掉的「已保存」检查等于没检查（同离线审计不留白名单的做法）。
    report.check("review_write.the_read_back_gate_has_no_switch", not knobs, str(knobs))
    report.check("review_write.the_server_keeps_one_ledger_parser",
                 "read_review_ledger" in serve_source and "json.loads(line)" not in serve_source,
                 f"delegated={'read_review_ledger' in serve_source}")
    report.check("review_write.unread_lines_reach_the_review_ui",
                 "function unreadNote" in app_source
                 and app_source.count("ledger_unread") >= 3
                 and "unreadNote(data.ledger_unread)" in app_source,
                 f"mentions={app_source.count('ledger_unread')}")

    fixture_root = resolve_path(config["paths"]["work"], config) / "eval" / "ledger_write"
    shutil.rmtree(fixture_root, ignore_errors=True)
    stored = _stored_documents(out_dir)
    workspace = fixture_root / "out"
    if not report.check("review_write.stored_document_available", bool(stored), str(out_dir)):
        return
    document = {}
    doc_id = ""
    candidates: List[str] = []
    for identifier in sorted(stored):
        payload = stored[identifier]
        pending = [path for path, field in sorted((payload.get("fields") or {}).items())
                   if field.get("decision") == "review"]
        if len(pending) >= 2:
            document, doc_id, candidates = payload, identifier, pending
            break
        if len(payload.get("fields") or {}) >= 2 and not document:
            # 复核台也接受「这一格我确认没问题」，所以出厂产物里没有待复核格时，
            # 用同一份文档里任意两格走同一条写入路径。
            document = payload
            doc_id = identifier
            candidates = sorted(payload.get("fields") or {})
    if not report.check("review_write.two_actionable_fields", len(candidates) >= 2,
                        f"document={doc_id} found={len(candidates)}"):
        return
    first, second = candidates[0], candidates[1]

    def save(body: Dict):
        """保存接口在「写完读不回来」时给的是 500，不是 ok——那正是这条腿要看的失败形态。"""
        try:
            return _post(port, f"/api/review/{doc_id}", body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            try:
                return int(exc.code), json.loads(detail)
            except (json.JSONDecodeError, ValueError):
                return int(exc.code), {"error": detail[:120]}
    (workspace / doc_id).mkdir(parents=True)
    (workspace / doc_id / f"{doc_id}.json").write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8")
    (workspace / "summary.json").write_text(json.dumps({"documents": [
        {"document_id": doc_id, "path": f"{doc_id}/{doc_id}.json"}]}), encoding="utf-8")
    (workspace / "review").mkdir()
    ledger = workspace / "review" / f"{doc_id}.review.jsonl"

    server = make_server(workspace, "127.0.0.1", 0, config)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not report.check("review_write.server_ready", _wait_ready(port), f"port={port}"):
            return
        marker = "__LEDGER_WRITE_PROBE__"
        status, saved = save({"field_path": first, "action": "correct",
                              "corrected_value": marker, "reviewer": "acceptance"})
        parsed = read_review_ledger(workspace, doc_id)
        report.check("review_write.saved_record_reads_back",
                     status == 200 and saved.get("ok") and saved.get("ledger_unread") == 0
                     and parsed["records"].get(first, {}).get("corrected_value") == marker
                     and not parsed["unread"],
                     f"status={status} unread={len(parsed['unread'])} "
                     f"reported={saved.get('ledger_unread')}")

        # 把台账改成「最后一行没有换行结尾」——上一次写被截断、或有人手改过台账的样子。
        ledger.write_text(ledger.read_text(encoding="utf-8").rstrip("\r\n"), encoding="utf-8")
        status, saved = save({"field_path": second, "action": "accept",
                              "reviewer": "acceptance"})
        parsed = read_review_ledger(workspace, doc_id)
        report.check("review_write.append_keeps_line_framing",
                     status == 200 and saved.get("ledger_unread") == 0
                     and parsed["lines"] == 2 and sorted(parsed["records"]) == sorted([first, second])
                     and parsed["records"][first]["corrected_value"] == marker,
                     f"lines={parsed['lines']} unread={len(parsed['unread'])} "
                     f"records={sorted(parsed['records'])}")

        # 两条人工修正粘在同一行：这一行读不出来，但必须数得出来，而不是当它不存在。
        glued = (json.dumps({"document_id": doc_id, "field_path": first, "action": "accept",
                             "timestamp": "2026-10-28T09:00:00"}, ensure_ascii=False)
                 + json.dumps({"document_id": doc_id, "field_path": second, "action": "correct",
                               "corrected_value": "另一条", "timestamp": "2026-10-28T09:00:01"},
                              ensure_ascii=False))
        ledger.write_text(glued + "\n", encoding="utf-8")
        status, _, body = _get(port, f"/api/document/{doc_id}")
        loaded = json.loads(body.decode("utf-8"))
        report.check("review_write.a_glued_pair_is_counted_not_swallowed",
                     status == 200 and loaded.get("ledger_unread") == 1
                     and loaded.get("reviews") == [],
                     f"unread={loaded.get('ledger_unread')} "
                     f"reviews={len(loaded.get('reviews') or [])}")

        # 合法 JSON 但不是对象的一行（`null`）以前会原样交给界面：取 field_path 时抛异常、整页打不开。
        good = json.dumps({"document_id": doc_id, "field_path": first, "action": "accept",
                           "timestamp": "2026-10-28T09:00:02"}, ensure_ascii=False)
        ledger.write_text(f"null\n\"裸字符串\"\n{good}\n", encoding="utf-8")
        status, _, body = _get(port, f"/api/document/{doc_id}")
        loaded = json.loads(body.decode("utf-8"))
        report.check("review_write.a_non_object_line_never_reaches_the_browser",
                     status == 200
                     and [item.get("field_path") for item in loaded.get("reviews") or []] == [first]
                     and loaded.get("ledger_unread") == 2,
                     f"reviews={loaded.get('reviews')} unread={loaded.get('ledger_unread')}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _check_doc_type_labels(report: Report, out_dir: Path, config: Dict) -> None:
    """铁律审计 XVI：文档类型的中文名只有配置一个主人，两个网页出口都跟着它翻面。

    复核台拿不到 YAML，靠 `/api/workspaces` 的 `doc_type_labels` 下发；单文件网页版在导出时
    嵌进 payload。界面自带的兜底、服务端载荷、成品网页三处必须逐字等于配置——它们此前已经
    漂开：`CUSTOMS_VAT_PAYMENT` 在配置里是「专用缴款书」，界面写成「缴款书」；配置里没有的
    类型在网页版上直接印英文码。只读产物，`build_report` 写到临时目录，`out/` 一个字节不动。
    """
    from .report import UNKNOWN_DOC_LABEL, build_report, doc_type_labels
    from .serve import ReviewStore

    labels = doc_type_labels(config)
    script = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const SHIPPED_DOC_LABELS = \{(.*?)\};", script, re.S)
    fallback = dict(re.findall(r'([A-Z_]+): "([^"]*)"', block.group(1))) if block else {}
    report.check("report.doc_type_labels_have_one_master",
                 bool(fallback) and fallback == labels
                 and labels.get("CUSTOMS_VAT_PAYMENT") == "海关进口增值税专用缴款书"
                 and labels.get("UNKNOWN") == UNKNOWN_DOC_LABEL
                 and "payload.doc_type_labels" in script
                 and "state.docLabels || {}" in script,
                 f"fallback={fallback} labels={labels}")

    shipped = ReviewStore(out_dir, config).workspace_payload().get("doc_type_labels") or {}
    report.check("report.doc_type_labels_travel_with_the_workspace_payload",
                 shipped == labels, f"payload={shipped} labels={labels}")

    def _payload(path: Path) -> Dict:
        html = path.read_text(encoding="utf-8")
        embedded = re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                             html, re.S)
        return json.loads(embedded.group(1).replace("<\\/", "</")) if embedded else {}

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    present = sorted({item["doc_type"] for item in summary.get("documents", [])})
    target_type = next((name for name in present if name in labels and name != "UNKNOWN"), None)
    with tempfile.TemporaryDirectory() as temp:
        clean = _payload(Path(build_report(out_dir, dest=Path(temp) / "clean",
                                           config=config)["path"]))
        moved = {}
        moved_store = {}
        if target_type:
            renamed = copy.deepcopy(config)
            renamed["doc_types"][target_type]["label"] = "改名后的类型名"
            renamed_path = Path(build_report(out_dir, dest=Path(temp) / "renamed",
                                             config=renamed)["path"])
            moved = {doc["doc_type"]: doc["doc_type_label"]
                     for doc in _payload(renamed_path).get("documents", [])}
            moved_store = doc_type_labels(renamed)
    shown = {doc["doc_type"]: doc["doc_type_label"] for doc in clean.get("documents", [])}
    group_labels = {item["type"]: item["label"] for item in clean.get("groups", [])}
    report.check("report.doc_type_labels_in_the_report_are_the_config_ones",
                 bool(shown) and all(shown[name] == labels.get(name) for name in shown)
                 and all(group_labels[name] == labels.get(name) for name in group_labels)
                 and all(shown[name] not in (None, "", name) for name in shown),
                 f"shown={shown} groups={group_labels}")
    report.check("report.renaming_a_doc_type_label_moves_the_report",
                 target_type is not None and target_type in moved
                 and moved[target_type] == "改名后的类型名"
                 and shown.get(target_type) not in moved.values()
                 and moved_store.get(target_type) == "改名后的类型名",
                 f"target={target_type} moved={moved.get(target_type)} "
                 f"store={moved_store.get(target_type)} shown={shown}")


def _check_non_cny_amounts(report: Report, out_dir: Path, config: Dict) -> None:
    """§4.4.1：第一版只自动接受 CNY，其他币种「全部金额复核」。

    以前只有币种那一格带 NON_CNY_CURRENCY：一份美元流水的余额、合计、交易金额
    照样自动接受，交付出去就是美元数字顶着人民币的名义。这里拿真实产物的字段表
    改掉币种、在内存里重跑判决（out/ 不动），把三件事一起钉住：非人民币时每个
    monetary 字段都被拦；人民币（含别名）时一个都不拦；已产出的流水里币种与金额
    的判决自洽。第三项在 dev/holdout 上是 0 条非人民币，detail 里必须写明覆盖到
    了几份流水，避免变成空转。
    """
    from .trust import assess, statement_currency

    docs = []
    for path in sorted(out_dir.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("doc_type") == "BANK_STATEMENT" and record.get("fields"):
            docs.append(record)
    target = docs[0] if docs else None
    monetary = [path for path, field in (target or {}).get("fields", {}).items()
                if field.get("risk") == "monetary"]

    def verdict(currency: str) -> Dict:
        doc = copy.deepcopy(target)
        doc["fields"]["bs.currency"]["value"] = currency
        page_count = max([int((field.get("grounding") or {}).get("page") or 1)
                          for field in doc["fields"].values()] + [1])
        parsed = {"page_count": page_count, "quality": {"parse_score": 0.99},
                  "pages": [{"page": number, "width": 10 ** 6, "height": 10 ** 6,
                             "quality_flags": [], "raster": None}
                            for number in range(1, page_count + 1)]}
        return assess([doc], {doc["document_id"]: parsed}, copy.deepcopy(config),
                      cross_document=False)[0]

    held, loose = [], []
    if target:
        foreign, local = verdict("美元"), verdict("人民币")
        held = [path for path in monetary
                if "NON_CNY_CURRENCY" not in (foreign["fields"][path]["review_reasons"] or [])]
        loose = [path for path in monetary
                 if "NON_CNY_CURRENCY" in (local["fields"][path]["review_reasons"] or [])]
    report.check("currency.non_cny_statement_reviews_every_amount",
                 bool(monetary) and not held,
                 f"doc={target['document_id'] if target else None} monetary={len(monetary)} "
                 f"unflagged={held[:5]}")
    report.check("currency.cny_statement_leaves_amounts_alone",
                 bool(monetary) and not loose, f"flagged={loose[:5]}")

    bad, non_cny = [], 0
    for record in docs:
        if statement_currency(record, config) == "CNY":
            continue
        non_cny += 1
        for path, field in record["fields"].items():
            if field.get("risk") == "monetary" \
                    and "NON_CNY_CURRENCY" not in (field.get("review_reasons") or []):
                bad.append(f"{record['document_id']}:{path}")
    report.check("currency.produced_statements_agree_with_their_currency_verdict",
                 bool(docs) and not bad,
                 f"statements={len(docs)} non_cny={non_cny} bad={bad[:5]}")


def _check_parse_grounding(report: Report, config: Dict) -> None:
    """SCENARIO §7.1（line 377）与 §3.1：解析阶段的块/单元格必须有页码与页内 bbox。

    过去这条铁律只落在字段层（trust._grounding_ok），解析产物里没有任何地方执行它：
    一个没坐标的块会原样进 lines/tables，等着后面某处抽出字段才被发现，而「丢弃并记
    GROUNDING_MISSING」这句在解析产物里查无此事。这里问三件事：栅门真的会丢弃并留痕；
    容差是配置里的那一个数（两层共读，改配置就同时移动两道门）；现有缓存里没有存活的
    无坐标块，且每页都带着丢弃清单。
    """
    import copy as _copy

    from . import parse as _parse

    parts = [{"text": "正常块", "bbox": [10, 10, 50, 20]},
             {"text": "没有框"},
             {"text": "零宽", "bbox": [10, 10, 10, 20]},
             {"text": "出页", "bbox": [0, 0, 999, 999]},
             {"text": "形状不对", "bbox": [1, 2, 3]}]
    kept, dropped = _parse.enforce_part_grounding(1, 595, 842, _copy.deepcopy(parts), config)
    reasons = [item["reason"] for item in dropped]
    report.check("parse.grounding_gate_drops_and_records",
                 [item["text"] for item in kept] == ["正常块"]
                 and reasons == ["bbox_missing", "bbox_degenerate", "bbox_outside", "bbox_missing"]
                 and all(item["code"] == _parse.GROUNDING_CODE for item in dropped)
                 and all(item["reason"] in _parse.GROUNDING_REASONS for item in dropped),
                 f"kept={[item['text'] for item in kept]} reasons={reasons}")

    edge = [{"text": "压边", "bbox": [100, 100, 596.5, 120]}]
    tight = _copy.deepcopy(config)
    tight["parsing"]["grounding"]["tolerance_pt"] = 0.0
    loose = _copy.deepcopy(config)
    loose["parsing"]["grounding"]["tolerance_pt"] = 2.0
    kept_tight, dropped_tight = _parse.enforce_part_grounding(
        1, 595, 842, _copy.deepcopy(edge), tight)
    kept_loose, dropped_loose = _parse.enforce_part_grounding(
        1, 595, 842, _copy.deepcopy(edge), loose)
    from . import trust as _trust
    field = {"path": "bl.company_name", "value": "示例", "required": True, "risk": "party_name",
             "grounding": {"page": 1, "bbox": [100, 100, 596.5, 120]}}
    pages = {1: {"page": 1, "width": 595, "height": 842}}
    ok_tight, _ = _trust._grounding_ok(field, 1, pages, _trust.grounding_tolerance(tight))
    ok_loose, _ = _trust._grounding_ok(field, 1, pages, _trust.grounding_tolerance(loose))
    report.check("parse.grounding_tolerance_is_one_config_number",
                 not kept_tight and [i["reason"] for i in dropped_tight] == ["bbox_outside"]
                 and kept_loose and not dropped_loose
                 and ok_tight is False and ok_loose is True
                 and _parse.grounding_tolerance(config) == _trust.grounding_tolerance(config),
                 f"tight_kept={len(kept_tight)} loose_kept={len(kept_loose)} "
                 f"field_tight={ok_tight} field_loose={ok_loose} "
                 f"tol={_parse.grounding_tolerance(config)}")

    parsed_dir = Path(resolve_path(config["paths"]["work"], config)) / "parsed"
    files = sorted(parsed_dir.glob("*.json")) if parsed_dir.exists() else []
    current = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("parser_version") == _parse.PARSER_VERSION:
            current.append((path.name, record))
    survivors, unrecorded, stale_count, bad_records = [], [], [], []
    for name, record in current:
        total = 0
        for page in record.get("pages") or []:
            width, height = float(page.get("width") or 0.0), float(page.get("height") or 0.0)
            if not isinstance(page.get("grounding_dropped"), list):
                unrecorded.append(name)
                continue
            for item in page["grounding_dropped"]:
                total += 1
                if item.get("code") != _parse.GROUNDING_CODE \
                        or item.get("reason") not in _parse.GROUNDING_REASONS:
                    bad_records.append(f"{name}:{item}")
            for line in page.get("lines") or []:
                bbox = line.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4
                        and bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0
                        and bbox[0] >= 0 and bbox[1] >= 0
                        and bbox[2] <= width and bbox[3] <= height):
                    survivors.append(f"{name}:line:{line.get('text')!r}")
            for table in page.get("tables") or []:
                for row in table.get("rows") or []:
                    for header, cell in (row.get("cells") or {}).items():
                        bbox = cell.get("bbox")
                        if not (isinstance(bbox, list) and len(bbox) == 4
                                and bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0):
                            survivors.append(f"{name}:{table.get('name')}:{header}")
        stale_count.append(int((record.get("quality") or {}).get("grounding_dropped", -1)) - total)
    report.check("parse.no_ungrounded_block_survives_in_the_cache",
                 bool(current) and not survivors and not unrecorded and not bad_records
                 and not any(stale_count),
                 f"documents={len(current)}/{len(files)} survivors={survivors[:3]} "
                 f"unrecorded={unrecorded[:3]} bad_records={bad_records[:2]} "
                 f"count_drift={[d for d in stale_count if d][:3]}")


def _check_doc_contract(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §8.1/§6.4：原因码三张表同源 + 单文档 JSON 形状。

    三张表以前只在文档里「应该一致」：`LOW_CONFIDENCE` 会被 trust 发出来，却没有
    `review_actions` 文案（界面上只剩裸码），也不在强制表里；`RAW_UNVERIFIED` 反过来
    ——有文案却双重惰性（开关默认关、码不在强制表），验收还得手改配置把它叫醒。
    这里问三件事：三张表彼此对得上；两个出口的产物真的符合 §8.1 的形状与约束；
    两条自弃权/编造路径在出厂配置下真的走到复核，而且文案一路跟到复核员手里。
    """
    import copy as _copy

    from . import doccontract as _dc
    from . import trust as _trust
    from .extract import _raw_text_audit, extract_document
    from .parse import parse_document
    from .review_actions import field_actions, review_action_table

    tables = _dc.check_reason_tables(config)
    report.check("contract.reason_tables_are_one_truth", not tables,
                 f"problems={tables[:3]} 发射/强制/文案="
                 f"{len(_trust.REVIEW_REASON_CODES)}/"
                 f"{len(config.get('hard_abstain_reasons') or [])}/"
                 f"{len(_dc.guidance_codes(config))}")

    tree = _dc.check_tree(out_dir, config)
    report.check("contract.documents_match_scenario_8_1",
                 tree["documents"] >= 5 and tree["fields"] >= 100
                 and not tree["problems"],
                 f"docs={tree['documents']} fields={tree['fields']} "
                 f"problems={tree['problems'][:3]}")

    work = resolve_path(config["paths"]["work"], config)
    temp_root = Path(tempfile.mkdtemp(prefix="docpipe_contract_", dir=str(work)))
    try:
        source = resolve_path(config["paths"]["samples"], config) / "dev" / "D007.pdf"
        if not report.check("contract.sample_available", source.exists(), str(source)):
            return
        parsed = parse_document(source, "CTR1", force=True, config=config,
                                out_dir=temp_root / "parsed")
        extracted = extract_document(parsed, config)
        table = review_action_table(config)
        target = next((path for path, field in extracted["fields"].items()
                       if field.get("raw_text")), "")
        if not report.check("contract.field_with_raw_text", bool(target),
                            "样张里没有带原文的字段"):
            return
        # 编造腿：原文换成页面上没有的一串字，出厂配置必须直接判复核并带文案。
        hurt = _copy.deepcopy(extracted)
        hurt["fields"][target]["raw_text"] = "9,999,999.00"
        (hurt["fields"][target].get("meta") or {}).pop("raw_grounded", None)
        _raw_text_audit(hurt["fields"], parsed, config)
        invented = assess([hurt], {"CTR1": parsed}, config,
                          cross_document=False)[0]["fields"][target]
        invented_actions = field_actions(invented, table)
        # 置信度腿：把全局下限抬到任何实测置信度之上，LOW_CONFIDENCE 就必须出现——
        # 这个码以前既没文案也不在强制表里，复核员只会看到一串英文。
        strict = _copy.deepcopy(config)
        strict["thresholds"]["global_min_confidence"] = 0.995
        low = assess([_copy.deepcopy(extracted)], {"CTR1": parsed}, strict,
                     cross_document=False)[0]["fields"][target]
        low_actions = field_actions(low, table)
        report.check("contract.abstain_codes_reach_the_reviewer_with_guidance",
                     invented.get("decision") == "review"
                     and "RAW_UNVERIFIED" in (invented.get("review_reasons") or [])
                     and bool(invented_actions)
                     and low.get("decision") == "review"
                     and "LOW_CONFIDENCE" in (low.get("review_reasons") or [])
                     and bool(low_actions),
                     f"invented={invented.get('review_reasons')} actions={len(invented_actions)} "
                     f"low={low.get('review_reasons')} actions={len(low_actions)}")
    except Exception as exc:
        report.check("contract.flow_exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _check_placeholder(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §4.1 规则 2：「必填字段为空、为 -- 、*** 、无 等占位符，判错」。

    这条以前完全没有机制：占位符就是普通字符串，必填字段印 `--` 会以原文交付；
    数字类型靠解析失败偶然落进复核，字符串/长文本/代码类则可能被自动接受。
    现在整格等于占位符一律折成空值（`extract.placeholders`），空这件事必须有人看
    （PLACEHOLDER_EMPTY）。这里问四件事：交付物里一个占位符都没有；任何值类型都
    留不住符号；清单是配置不是代码；被拦下的格子带着文案走到复核员手里。
    """
    from .extract import _normalize, placeholders
    from .textutil import normalize_text
    from .trust import Calibrator, decide_field

    tokens = list(placeholders(config))
    documents = _stored_documents(out_dir)
    leaked = []
    for doc in documents.values():
        for path, field in (doc.get("fields") or {}).items():
            if isinstance(field, dict) and normalize_text(field.get("value")) in tokens:
                leaked.append(f"{doc.get('document_id')}:{path}={field.get('value')!r}")
    report.check("placeholder.none_delivered_as_value",
                 bool(documents) and not leaked,
                 f"documents={len(documents)} tokens={tokens} leaked={leaked[:3]}")

    kinds = (("string", {}), ("date", {}), ("decimal", {}), ("integer", {}),
             ("enum", {"enum": "unit"}))
    wrong = []
    for kind, extra in kinds:
        for token in tokens:
            value, status, meta = _normalize(token, {"type": kind, **extra}, config, {})
            if value is not None or status != "placeholder" \
                    or meta.get("placeholder") != token:
                wrong.append(f"{kind}:{token!r}->{value!r}/{status}")
    report.check("placeholder.no_value_type_keeps_the_symbol", not wrong,
                 f"wrong={wrong[:3]}")

    custom = copy.deepcopy(config)
    custom["validation"]["placeholders"] = ["NIL"]
    driven = (_normalize("NIL", {"type": "string"}, custom, {})[1] == "placeholder"
              and _normalize("--", {"type": "string"}, custom, {})[1] == "ok")
    report.check("placeholder.list_is_config_not_code", driven,
                 "把清单换成 ['NIL'] 之后 '--' 仍被当占位符（说明代码里写了字面量）")

    parsed = {"page_count": 1,
              "pages": [{"page": 1, "width": 595.0, "height": 842.0}]}
    field = {
        "path": "is.revenue", "label": "营业收入", "value_type": "decimal",
        "required": True, "risk": "monetary", "value": None, "raw_text": "--",
        "status": "placeholder", "confidence": 0.9, "extractor_id": "table_row:income",
        "grounding": {"page": 1, "bbox": [260.0, 180.0, 340.0, 192.0]},
        "signals": {"ocr_conf": 0.99, "rule_exact": 0.0, "format_valid": 1.0,
                    "table_score": 1.0, "cross_field_score": 1.0},
        "meta": {"placeholder": "--"},
    }
    decided = decide_field(field, {"doc_type": "INCOME_STATEMENT", "fields": {}},
                           parsed, config, [], Calibrator(config))
    actions = field_actions(decided, review_action_table(config))
    report.check("placeholder.code_reaches_the_reviewer_with_guidance",
                 decided["decision"] == "review"
                 and "PLACEHOLDER_EMPTY" in (decided.get("review_reasons") or [])
                 and bool(actions),
                 f"decision={decided['decision']} "
                 f"reasons={decided.get('review_reasons')} actions={len(actions)}")


def _check_normalization(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §4.2：「只接受配置中的枚举；同义写法按配置映射，映射不唯一则复核」。

    这两句以前都没有机械装置：不在词表里的原文带着 status=unmapped 原样自动接受
    （判决层从不读 status），同一别名挂在两个规范值下时按字典顺序静默挑第一个。
    这里问四件事：出厂词表自身没有撞名；交付出去的每个枚举值都在词表里（带分母）；
    撞名走复核；词表外的值走复核。
    """
    from .config import field_spec
    from .extract import _normalize, enum_matches, normalize_enum
    from .trust import (Calibrator, decide_field, normalize_path, validate_document_fields)

    enums = config.get("enums") or {}
    collisions = []
    alias_count = 0
    for name, table in enums.items():
        owners: Dict[str, set] = {}
        for canonical, aliases in table.items():
            for alias in [str(canonical)] + [str(item) for item in (aliases or [])]:
                owners.setdefault(alias.strip().upper(), set()).add(str(canonical))
                alias_count += 1
        for alias, group in owners.items():
            if len(group) > 1:
                collisions.append(f"{name}:{alias}->{'/'.join(sorted(group))}")
    report.check("normalization.enum_aliases_are_unambiguous",
                 bool(enums) and not collisions,
                 f"enums={len(enums)} aliases={alias_count} collisions={collisions[:3]}")

    documents = _stored_documents(out_dir)
    checked = 0
    offenders = []
    for doc_id, doc in documents.items():
        for path, field in (doc.get("fields") or {}).items():
            if not isinstance(field, dict):
                continue
            spec = field_spec(doc.get("doc_type"), normalize_path(path), config) or {}
            if spec.get("type") != "enum":
                continue
            value = field.get("value")
            if value in (None, ""):
                continue
            checked += 1
            if normalize_enum(str(value), spec.get("enum"), config) != value:
                offenders.append(f"{doc_id}:{path}={value!r}")
    report.check("normalization.delivered_enum_values_are_configured",
                 bool(documents) and checked > 0 and not offenders,
                 f"documents={len(documents)} enum_fields={checked} offenders={offenders[:3]}")

    def decider(raw, cfg, path="bs.currency", doc_type="BANK_STATEMENT", enum="currency"):
        value, status, meta = _normalize(raw, {"type": "enum", "enum": enum}, cfg, {})
        field = {
            "path": path, "label": "字段", "value_type": "enum", "required": False,
            "risk": "optional_low_risk", "value": value, "raw_text": raw,
            "status": status, "confidence": 0.99, "extractor_id": "label_value",
            "grounding": {"page": 1, "bbox": [10.0, 10.0, 60.0, 24.0],
                          "bbox_norm": [0.01, 0.01, 0.1, 0.03]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "table_score": 1.0},
            "meta": meta,
        }
        parsed = {"page_count": 1,
                  "pages": [{"page": 1, "width": 595.0, "height": 842.0}],
                  "quality": {"parse_score": 0.99}}
        doc = {"doc_id": "N001", "doc_type": doc_type, "fields": {path: field}}
        validators: Dict = {}
        validate_document_fields(doc, parsed, cfg, validators)
        decided = decide_field(field, doc, parsed, cfg, validators.get(path, []),
                               Calibrator(cfg))
        return field, decided, validators.get(path, [])

    colliding = copy.deepcopy(config)
    colliding["enums"]["currency"] = {"CNY": ["人民币", "元"], "USD": ["美元", "元"]}
    field, decided, validators = decider("元", colliding)
    report.check("normalization.alias_collision_reaches_review",
                 field["status"] == "unmapped"
                 and (field.get("meta") or {}).get("enum_ambiguous") == ["CNY", "USD"]
                 and decided["decision"] == "review"
                 and "FORMAT_FAIL" in (decided.get("review_reasons") or [])
                 and any("CNY" in str(item.get("detail")) and "USD" in str(item.get("detail"))
                         for item in validators),
                 f"status={field['status']} meta={field.get('meta')} "
                 f"decision={decided['decision']} reasons={decided.get('review_reasons')}")

    field, decided, validators = decider("美元", config)
    report.check("normalization.unmapped_enum_is_never_auto_accepted",
                 decided["decision"] == "review"
                 and "FORMAT_FAIL" in (decided.get("review_reasons") or [])
                 and decided.get("value") == "美元"
                 and [item.get("detail") for item in validators]
                 == ["value not in the configured enum"],
                 f"decision={decided['decision']} reasons={decided.get('review_reasons')} "
                 f"value={decided.get('value')!r}")


def run_acceptance(out_dir: Path, labels_dir: Path, config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    global HTTP_TIMEOUT_SECONDS
    HTTP_TIMEOUT_SECONDS = float((config.get("acceptance") or {})
                                 .get("http_timeout_seconds", HTTP_TIMEOUT_SECONDS))
    report = Report()
    out_dir = out_dir.resolve()
    labels_dir = labels_dir.resolve()
    started_at = now_iso()
    report.check("input.out_dir", out_dir.exists(), str(out_dir))
    report.check("input.labels_dir", labels_dir.exists(), str(labels_dir))
    summary = _check_contract(report, out_dir, labels_dir)
    evaluation = None
    if out_dir.exists() and labels_dir.exists():
        evaluation = evaluate_predictions(out_dir, labels_dir, config)
        _check_gates(report, evaluation, config)
    export_summary = _check_review_flow(report, out_dir, config)
    _check_image_ingest(report, config)
    _check_photo_preprocess(report, config)
    _check_table_anomaly(report, config)
    _check_credit_report_matrix(report, config)
    _check_ocr_engine(report, config)
    _check_ocr_recrop(report, config)
    _check_extractor_crosscheck(report, config)
    _check_raw_audit(report, config)
    _check_reason_ui_labels(report, config)
    _check_review_contract(report, out_dir, config)
    _check_value_rewrite_disclosure(report, out_dir, config)
    _check_chain_disclosure(report, out_dir, config)
    _check_config_thresholds(report, config)
    _check_schema_paths(report, config)
    _check_split_isolation(report, config)
    _check_sample_freeze(report, config)
    _check_corpus_shape(report, labels_dir, config)
    _check_statement_rows(report, config)
    _check_evaluation_provenance(report, evaluation, labels_dir, config)
    _check_evaluation_report(report, evaluation)
    _check_section65_metrics(report, evaluation, out_dir, labels_dir, config)
    _check_section65_accept_rate(report, evaluation, out_dir, labels_dir, config)
    _check_matching_contract(report, config)
    _check_cross_document(report, config)
    _check_statement_recency(report, config)
    _check_income_columns(report, config)
    _check_income_amounts(report, config)
    _check_parse_grounding(report, config)
    _check_doc_contract(report, out_dir, config)
    _check_placeholder(report, out_dir, config)
    _check_normalization(report, out_dir, config)
    _check_offline(report, config)
    _check_delivery(report, out_dir, config)
    _check_coordinates(report, out_dir, config)
    _check_gate_explainability(report, out_dir, config)
    _check_ledger_write(report, out_dir, config)
    _check_review_loop(report, out_dir, config)
    table_structure = _check_table_structure(report, out_dir, labels_dir, config)
    _check_page_quality_gate(report, config)
    _check_ingest_flow(report, config)
    _check_doc_type_labels(report, out_dir, config)
    _check_non_cny_amounts(report, out_dir, config)
    _check_string_lengths(report, config)
    _check_enum_review_values(report, config)

    _check_scenario_contract(report, config)
    _check_label_contract(report, config)
    _check_csv_contract(report, out_dir, config)
    _check_summary_contract(report, out_dir, config)
    _check_document_contract(report, out_dir, config)
    _check_signal_contract(report, out_dir, config)
    _check_cli_surface(report, config)
    _check_table_crosspath(report, labels_dir, config)
    _check_doc_type_contract(report, config)
    _check_parse_contract(report, config)
    _check_document_refusal(report, config)
    _check_model_contract(report, out_dir, config)
    _check_threshold_contract(report, config)
    _check_gate_contract(report, config)
    payload = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "out_dir": str(out_dir),
        "labels_dir": str(labels_dir),
        "ok": report.ok,
        "checks": report.checks,
        "metrics": evaluation["summary"] if evaluation else None,
        "documents": (evaluation or {}).get("documents"),
        "provenance": (evaluation or {}).get("provenance"),
        "claim_scope": (evaluation or {}).get("claim_scope"),
        "limitations": (evaluation or {}).get("limitations"),
        "production_ready": (evaluation or {}).get("production_ready"),
        "export": export_summary,
        "table_structure": table_structure,
        "summary": summary,
    }
    out_path = resolve_path(config["paths"]["work"], config) / "eval" / \
        f"acceptance_{labels_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["report_path"] = str(out_path)
    return payload


def _check_evaluation_report(report: Report, evaluation: Optional[Dict]) -> None:
    """SCENARIO §10.2：评估报告要输出七类内容。

    第 1 项的「平均置信度」与第 2 项的「按字段路径」此前根本没算：报告只按文档类型与
    风险档分组，换一个抽取器或换一套阈值后，看不出是哪一类字段在拖后腿；平均值也不带
    覆盖率，三个格子撑起来的均值会冒充一千五百个格子。这一项把七类内容逐条点名，并要求
    按字段路径分组后的字段总数等于总字段数（分组不许把格子弄丢）。
    """
    summary = (evaluation or {}).get("summary") or {}
    overall = ("field_count", "auto_accept_rate", "review_rate", "accepted_error_rate",
               "monetary_accepted_error_rate", "grounding_hit_rate", "avg_confidence")
    confusion = ("correct_accept", "wrong_accept", "over_review", "unresolved_review",
                 "reviewed_corrected")
    paths = summary.get("by_field_path") or []
    grouped_total = sum(int(row.get("field_count") or 0) for row in paths)
    latency = summary.get("latency") or {}
    claim = summary.get("claim") or {}
    report.check("evaluation.section102_items",
                 bool(evaluation)
                 and all(key in summary for key in overall)
                 and all(key in (summary.get("confusion") or {}) for key in confusion)
                 and bool(summary.get("by_doc_type")) and bool(paths)
                 and grouped_total == int(summary.get("field_count") or 0)
                 and summary.get("error_categories") is not None
                 and isinstance(latency.get("avg_page_seconds"), (int, float))
                 and isinstance(latency.get("avg_document_seconds"), (int, float))
                 and isinstance(summary.get("accepted_error_ci"), list)
                 and claim.get("min_accept_for_claim") is not None
                 and bool((evaluation or {}).get("limitations")),
                 f"keys={len(summary)} paths={len(paths)} grouped={grouped_total}"
                 f"/{summary.get('field_count')} avg_conf={summary.get('avg_confidence')}"
                 f" coverage={summary.get('confidence_coverage')}"
                 f" page_avg={latency.get('avg_page_seconds')}"
                 f" limitations={len((evaluation or {}).get('limitations') or [])}")

def _check_evaluation_provenance(report: Report, evaluation: Optional[Dict],
                                 labels_dir: Path, config: Dict) -> None:
    """SCENARIO §9.3/§9.4：报告要带代码版本、配置哈希、样本清单哈希，并写明合成集局限。

    这三样此前一个都没记：`decisions_*.json` 里只有指标，换一版代码或改一个阈值重跑，
    两份报告分不清谁是谁；§9.4 的三条局限只活在 SCENARIO 里，报告上看不到，合成集的
    数字就很容易被当成真实业务准确率。
    """
    from .evaluate import report_provenance

    shipped = report_provenance(labels_dir.name, config)
    prov = (evaluation or {}).get("provenance") or {}
    hexes = [prov.get("code_sha256"), prov.get("config_sha256"),
             prov.get("sample_manifest_sha256")]
    report.check("evaluation.provenance_is_recorded",
                 bool(evaluation)
                 and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hexes)
                 and prov.get("split") == labels_dir.name,
                 f"split={prov.get('split')} code={str(hexes[0])[:12]} "
                 f"config={str(hexes[1])[:12]} samples={str(hexes[2])[:12]}")

    # 反向：门槛少一份就只能说 synthetic_only，够了才允许说 real_calibrated。
    floor = int(shipped["provenance"]["min_real_samples"])
    below = report_provenance(labels_dir.name, config, real_samples=max(0, floor - 1))
    at = report_provenance(labels_dir.name, config, real_samples=floor)
    report.check("evaluation.scope_follows_the_real_sample_floor",
                 below["claim_scope"] == "synthetic_only"
                 and below["production_ready"] is False
                 and at["claim_scope"] == "real_calibrated"
                 and at["production_ready"] is True
                 and (evaluation or {}).get("claim_scope") == shipped["claim_scope"],
                 f"floor={floor} below={below['claim_scope']} at={at['claim_scope']} "
                 f"reported={ (evaluation or {}).get('claim_scope') }")

    # 反向：局限文案跟着配置走——换成哨兵要逐字带出，清空就不许自己补一段。
    mutated = copy.deepcopy(config)
    mutated.setdefault("evaluation", {}).setdefault("provenance", {})["limitations"] = ["哨兵局限"]
    sentinel = report_provenance(labels_dir.name, mutated)["limitations"]
    emptied = copy.deepcopy(config)
    emptied.setdefault("evaluation", {}).setdefault("provenance", {})["limitations"] = []
    none = report_provenance(labels_dir.name, emptied)["limitations"]
    report.check("evaluation.limitations_come_from_config",
                 sentinel == ["哨兵局限"] and none == []
                 and bool(shipped["limitations"])
                 and (evaluation or {}).get("limitations") == shipped["limitations"],
                 f"shipped={len(shipped['limitations'])} sentinel={sentinel} none={none} "
                 f"reported={len((evaluation or {}).get('limitations') or [])}")



def _section65_counts(preds: Dict[str, Dict], labels: Dict[str, Dict],
                      config: Dict) -> Dict:
    """按 SCENARIO §6.5 的两行定义重算，不复用评估层的实现——同源就不算核对。"""
    from .config import field_spec

    missing = total = fully = 0
    per_doc: Dict[str, bool] = {}
    for doc_id, label in labels.items():
        fields = (preds.get(doc_id) or {}).get("fields") or {}
        required_ok = True
        required_seen = False
        for path in label["fields"]:
            total += 1
            field = fields.get(path) or {}
            grounding = field.get("grounding") or {}
            bbox = grounding.get("bbox")
            present = False
            if grounding.get("page") not in (None, "") and isinstance(bbox, list) \
                    and len(bbox) == 4:
                try:
                    x0, y0, x1, y1 = (float(value) for value in bbox)
                    present = x1 > x0 and y1 > y0
                except (TypeError, ValueError):
                    present = False
            if not present:
                missing += 1
            spec = field_spec(label["doc_type"], re.sub(r"\[\d+\]", "[i]", path),
                              config) or {}
            if spec.get("required"):
                required_seen = True
                if field.get("decision") != "accept":
                    required_ok = False
        per_doc[doc_id] = bool(required_seen and required_ok)
        fully += int(per_doc[doc_id])
    return {"fields": total, "missing": missing, "documents": len(labels),
            "fully_auto": fully, "per_doc": per_doc}


def _check_section65_metrics(report: Report, evaluation: Optional[Dict],
                             out_dir: Path, labels_dir: Path, config: Dict) -> None:
    """SCENARIO §6.5 指标表的最后两行：溯源缺失率与文档全自动率。

    这两行此前一个都没算：报告里只有「已接受字段的溯源缺失数」，一份坐标全丢、
    整份进复核的文档在这个口径下也是 0；「一份文档还要不要人从头看一遍」更是从未
    记录过——按字段数平摊会把「整份文档不可用」平均成一个小数点。
    """
    labels: Dict[str, Dict] = {}
    preds: Dict[str, Dict] = {}
    if labels_dir.exists() and out_dir.exists():
        for label_path in sorted(labels_dir.glob("*.json")):
            doc_id = label_path.stem
            pred_path = out_dir / doc_id / f"{doc_id}.json"
            if not pred_path.exists():
                continue
            labels[doc_id] = json.loads(label_path.read_text(encoding="utf-8"))
            preds[doc_id] = json.loads(pred_path.read_text(encoding="utf-8"))
    recounted = _section65_counts(preds, labels, config)
    summary = (evaluation or {}).get("summary") or {}
    documents = (evaluation or {}).get("documents") or []
    expected_rate = (round(recounted["missing"] / recounted["fields"], 4)
                     if recounted["fields"] else None)
    report.check("evaluation.section65_grounding_missing_rate",
                 bool(evaluation) and recounted["fields"] > 0
                 and summary.get("field_count") == recounted["fields"]
                 and summary.get("grounding_missing_count") == recounted["missing"]
                 and summary.get("grounding_missing_rate") == expected_rate,
                 f"fields={recounted['fields']} missing={recounted['missing']} "
                 f"reported={summary.get('grounding_missing_count')}"
                 f"/{summary.get('grounding_missing_rate')} expected={expected_rate}")

    expected_docs = (round(recounted["fully_auto"] / recounted["documents"], 4)
                     if recounted["documents"] else None)
    report.check("evaluation.section65_document_auto_rate",
                 bool(evaluation) and recounted["documents"] > 0
                 and len(documents) == recounted["documents"]
                 and summary.get("document_fully_auto_count") == recounted["fully_auto"]
                 and summary.get("document_auto_rate") == expected_docs,
                 f"documents={recounted['documents']} fully_auto={recounted['fully_auto']} "
                 f"reported={summary.get('document_fully_auto_count')}"
                 f"/{summary.get('document_auto_rate')} expected={expected_docs}")

    # 反向：把一份「全自动」文档的某个必填格改成 review，分子必须正好掉一份。
    from .config import field_spec

    mutated = copy.deepcopy(preds)
    flipped = None
    for doc_id, pred in sorted(mutated.items()):
        if not recounted["per_doc"].get(doc_id):
            continue
        label = labels.get(doc_id) or {}
        for path, field in (pred.get("fields") or {}).items():
            spec = field_spec(label.get("doc_type", ""),
                              re.sub(r"\[\d+\]", "[i]", path), config) or {}
            if spec.get("required") and field.get("decision") == "accept":
                field["decision"] = "review"
                flipped = f"{doc_id}:{path}"
                break
        if flipped:
            break
    after = _section65_counts(mutated, labels, config) if flipped else recounted
    report.check("evaluation.section65_document_auto_rate_bites",
                 bool(flipped)
                 and after["fully_auto"] == recounted["fully_auto"] - 1,
                 f"flipped={flipped} before={recounted['fully_auto']} "
                 f"after={after['fully_auto']}")


def _required_accept_counts(preds: Dict[str, Dict], labels: Dict[str, Dict],
                            config: Dict) -> Dict:
    """按 §6.5 第一行的定义重数一遍：分母只有必填格，可选格单独记账。"""
    from .config import field_spec

    required = required_accept = optional = optional_accept = 0
    for doc_id, label in labels.items():
        fields = (preds.get(doc_id) or {}).get("fields") or {}
        for path in label["fields"]:
            field = fields.get(path) or {}
            accepted = field.get("decision") == "accept"
            spec = field_spec(label["doc_type"], re.sub(r"\[\d+\]", "[i]", path),
                              config) or {}
            if spec.get("required"):
                required += 1
                required_accept += int(accepted)
            else:
                optional += 1
                optional_accept += int(accepted)
    return {"required": required, "required_accept": required_accept,
            "optional": optional, "optional_accept": optional_accept}


def _accept_rate_problem(summary: Dict, counts: Dict) -> str:
    """报告里的四个数与独立重数的四个数对不上就说出哪一对：空字符串才算对。"""
    if not counts["required"]:
        return "独立重数里没有必填字段，这条指标无从核对"
    expected = round(counts["required_accept"] / counts["required"], 4)
    pairs = (("required_field_count", summary.get("required_field_count"), counts["required"]),
             ("required_accept_count", summary.get("required_accept_count"),
              counts["required_accept"]),
             ("auto_accept_rate", summary.get("auto_accept_rate"), expected))
    wrong = [f"{name}={got} expected={want}" for name, got, want in pairs if got != want]
    total = counts["required"] + counts["optional"]
    all_rate = round((counts["required_accept"] + counts["optional_accept"]) / total, 4) \
        if total else 0.0
    if summary.get("field_accept_rate") != all_rate:
        wrong.append(f"field_accept_rate={summary.get('field_accept_rate')} "
                     f"expected={all_rate}")
    return "; ".join(wrong)


def _check_section65_accept_rate(report: Report, evaluation: Optional[Dict],
                                 out_dir: Path, labels_dir: Path, config: Dict) -> None:
    """SCENARIO §6.5 第一行：自动接受率 = 必填 accept / 必填。

    这条指标以前用**全部字段**当分母（dev 1495 格里有 561 格是可选格，且全部自动接受），
    可选格越多、分母越虚胖，门槛看着越低——「阶段 4 ≥ 60%」本该问的是必填格。
    """
    labels: Dict[str, Dict] = {}
    preds: Dict[str, Dict] = {}
    if labels_dir.exists() and out_dir.exists():
        for label_path in sorted(labels_dir.glob("*.json")):
            doc_id = label_path.stem
            pred_path = out_dir / doc_id / f"{doc_id}.json"
            if not pred_path.exists():
                continue
            labels[doc_id] = json.loads(label_path.read_text(encoding="utf-8"))
            preds[doc_id] = json.loads(pred_path.read_text(encoding="utf-8"))
    counts = _required_accept_counts(preds, labels, config)
    summary = (evaluation or {}).get("summary") or {}
    problem = _accept_rate_problem(summary, counts)
    report.check("evaluation.section65_accept_rate_denominator",
                 bool(evaluation) and counts["required"] > 0 and not problem,
                 f"required={counts['required_accept']}/{counts['required']} "
                 f"optional={counts['optional_accept']}/{counts['optional']} "
                 f"reported={summary.get('required_accept_count')}"
                 f"/{summary.get('required_field_count')} rate={summary.get('auto_accept_rate')}"
                 + (f" problem={problem}" if problem else ""))

    # 反向：把两个最像的错法各造一次，检查必须当场说话——
    # (a) 按老口径（全部字段）算的比例；(b) 分母少算一格（指针漏掉一个必填格）。
    total = counts["required"] + counts["optional"]
    legacy = (round((counts["required_accept"] + counts["optional_accept"]) / total, 4)
              if total else 0.0)
    drifted = dict(summary, auto_accept_rate=legacy)
    shrunk = dict(summary, required_field_count=max(0, counts["required"] - 1))
    legacy_problem = _accept_rate_problem(drifted, counts)
    shrunk_problem = _accept_rate_problem(shrunk, counts)
    report.check("evaluation.section65_accept_rate_bites",
                 bool(evaluation) and legacy != summary.get("auto_accept_rate")
                 and bool(legacy_problem) and bool(shrunk_problem),
                 f"legacy_rate={legacy} caught={legacy_problem[:60]!r} "
                 f"denominator_drift_caught={shrunk_problem[:60]!r}")


def _check_matching_contract(report: Report, config: Dict) -> None:
    """SCENARIO §10.1 的三条匹配规则：值完全相等、溯源「中心在框内 或 IoU≥0.5」、
    交易行内容键只含日期/金额/余额。

    过去这三条都只活在文档里：`grounding_ok` 把「任何正重叠」算命中，decimal 判等带
    ±0.01 容差，内容键多一个 direction。合成语料上两套口径恰好同分（dev 有 495 格、
    holdout 有 401 格靠「中心在框内」救回，逐格看过），所以改对了没人发现，改坏了
    同样不会——这里把三条各自钉住，每条都由反向夹具单独点亮。
    """
    import copy as _copy

    from . import evaluate as _evaluate

    shipped = _evaluate.match_options(config)
    contract_keys = ["txn_date", "amount", "balance"]
    declared = {key: shipped.get(key) for key in
                ("decimal_tolerance", "min_iou", "center_inside", "transaction_keys")}
    report.check("evaluation.match_matches_the_contract",
                 float(shipped.get("decimal_tolerance", -1)) == 0.0
                 and float(shipped.get("min_iou", -1)) == 0.5
                 and shipped.get("center_inside") is True
                 and list(shipped.get("transaction_keys") or []) == contract_keys,
                 f"shipped={declared}")

    gt = {"page": 1, "bbox": [100.0, 100.0, 200.0, 200.0]}
    # 中心 (150,150) 在标注框内，但预测框是整页大小的框：IoU 只有 0.111。
    centre_only = {"page": 1, "bbox": [0.0, 0.0, 300.0, 300.0]}
    # 平移 10 点的紧框：IoU 0.68，中心也在框内——用来单独验证 IoU 门槛本身。
    tight = {"page": 1, "bbox": [110.0, 110.0, 210.0, 210.0]}
    # 正重叠 100 平方点、中心 (210,210) 在框外：旧口径的假命中就是这种格子。
    sliver = {"page": 1, "bbox": [190.0, 190.0, 230.0, 230.0]}
    no_centre = dict(shipped, center_inside=False)
    strict_iou = dict(no_centre, min_iou=0.8)
    report.check("evaluation.grounding_centre_rule",
                 _evaluate._iou(gt["bbox"], centre_only["bbox"]) < 0.5
                 and _evaluate.grounding_ok(gt, centre_only, shipped)
                 and not _evaluate.grounding_ok(gt, centre_only, no_centre),
                 f"iou={round(_evaluate._iou(gt['bbox'], centre_only['bbox']), 4)} "
                 f"default={_evaluate.grounding_ok(gt, centre_only, shipped)} "
                 f"centre_off={_evaluate.grounding_ok(gt, centre_only, no_centre)}")
    report.check("evaluation.grounding_iou_rule",
                 _evaluate._iou(gt["bbox"], tight["bbox"]) >= 0.5
                 and _evaluate.grounding_ok(gt, tight, no_centre)
                 and not _evaluate.grounding_ok(gt, tight, strict_iou),
                 f"iou={round(_evaluate._iou(gt['bbox'], tight['bbox']), 4)} "
                 f"centre_off={_evaluate.grounding_ok(gt, tight, no_centre)} "
                 f"min_iou=0.8={_evaluate.grounding_ok(gt, tight, strict_iou)}")
    report.check("evaluation.grounding_tiny_overlap_is_a_miss",
                 _evaluate._iou(gt["bbox"], sliver["bbox"]) > 0.0
                 and not _evaluate.grounding_ok(gt, sliver, shipped),
                 f"iou={round(_evaluate._iou(gt['bbox'], sliver['bbox']), 4)} "
                 f"hit={_evaluate.grounding_ok(gt, sliver, shipped)}")

    decimal_spec = {"type": "decimal", "risk": "monetary"}
    loose_decimal = dict(shipped, decimal_tolerance=0.01)
    report.check("evaluation.value_exactness",
                 _evaluate.values_match(decimal_spec, "100.00", "100", shipped)
                 and not _evaluate.values_match(decimal_spec, "100.00", "100.01", shipped)
                 and _evaluate.values_match(decimal_spec, "100.00", "100.01", loose_decimal),
                 f"trailing_zero={_evaluate.values_match(decimal_spec, '100.00', '100', shipped)} "
                 f"one_cent={_evaluate.values_match(decimal_spec, '100.00', '100.01', shipped)} "
                 f"tolerance_0.01={_evaluate.values_match(decimal_spec, '100.00', '100.01', loose_decimal)}")

    labels = {"doc_type": "BANK_STATEMENT", "fields": {}}
    pred_fields = {}
    row = {"txn_date": "2025-07-01", "amount": "100.00", "balance": "1100.00",
           "direction": "income"}
    for column, value in row.items():
        path = f"bs.transactions[0].{column}"
        labels["fields"][path] = {"value": value}
        pred_fields[path] = {"value": "expense" if column == "direction" else value}
    default = _evaluate.transaction_comparison(labels, pred_fields, "BANK_STATEMENT", config)
    with_direction = _copy.deepcopy(config)
    with_direction.setdefault("evaluation", {}).setdefault("match", {})[
        "transaction_keys"] = contract_keys + ["direction"]
    strict = _evaluate.transaction_comparison(labels, pred_fields, "BANK_STATEMENT",
                                              with_direction)
    report.check("evaluation.transaction_keys_follow_config",
                 default["content_matched"] == 1
                 and default["keys"] == contract_keys
                 and strict["content_matched"] == 0
                 and strict["keys"] == contract_keys + ["direction"],
                 f"default={default['content_matched']}/{default['keys']} "
                 f"with_direction={strict['content_matched']}/{strict['keys']}")


def _check_cross_document(report: Report, config: Dict) -> None:
    """SCENARIO §5.4 跨文档一致性：契约表格是规则表，代码只能照它执行。

    过去两条会发出去的判决把严重度词写死在 trust.py 里（V-X-01 契约是 warn、代码发 fail），
    而 V-X-03/04/05 在代码里根本没有对应物——「表里有、代码没有」这件事以前只能靠人读文档发现。
    这里问四件事：规则表与 §5.4 逐条对得上；代码真会发的判决等于表里 `enforced: true` 的那些；
    失败打在契约点名的字段上；契约要求复核的规则不能因为词写成 warn 就放行。
    """
    import copy as _copy
    import re as _re

    from . import trust as _trust

    scenario = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    rows = _re.findall(r"^\|\s*(V-X-\d+)\s*\|[^|]*\|\s*([^|]*?)\s*\|\s*$",
                       scenario, _re.M)
    contract = {}
    for rule_id, action in rows:
        word = _re.split(r"[；（/，,]", action)[0].strip()
        contract[rule_id] = {"severity": word if word in ("warn", "fail") else "none",
                             "review": "复核" in action}
    registry = _trust.cross_document_rules(config)
    drift = []
    for rule_id in sorted(contract):
        entry = registry.get(rule_id)
        if entry is None:
            drift.append(f"{rule_id}:缺登记")
            continue
        if str(entry["severity"]) != contract[rule_id]["severity"]:
            drift.append(f"{rule_id}:severity=配置{entry['severity']}/契约{contract[rule_id]['severity']}")
        if bool(entry["review"]) != contract[rule_id]["review"]:
            drift.append(f"{rule_id}:review=配置{entry['review']}/契约{contract[rule_id]['review']}")
    extra = sorted(set(registry) - set(contract))
    report.check("cross_document.registry_matches_the_scenario_table",
                 bool(contract) and not drift and not extra,
                 f"scenario={len(contract)} registry={len(registry)} "
                 f"drift={drift[:3]} extra={extra}")

    enforced = sorted(rid for rid, entry in registry.items() if entry.get("enforced"))
    code_paths = {"V-X-01": sorted(set(_trust.CROSS_DOC_NAME_PATHS.values())),
                  "V-X-02": sorted(_trust.CROSS_DOC_USCC_PATHS),
                  "V-X-04": sorted(_trust.STATEMENT_RECENCY_PATHS)}
    path_drift = [f"{rid}:registry={sorted(registry[rid].get('paths') or [])} "
                  f"code={code_paths.get(rid)}"
                  for rid in enforced
                  if sorted(registry[rid].get("paths") or []) != code_paths.get(rid)]
    report.check("cross_document.emitted_rules_are_declared",
                 enforced == sorted(_trust.CROSS_DOC_EMITTED) and not path_drift,
                 f"enforced={enforced} emitted={sorted(_trust.CROSS_DOC_EMITTED)} "
                 f"not_enforced={sorted(set(registry) - set(enforced))} path_drift={path_drift[:2]}")

    def _doc(doc_id, doc_type, values):
        return {"document_id": doc_id, "doc_type": doc_type,
                "fields": {path: {"path": path, "value": value} for path, value in values.items()}}

    uscc = "91310000MA1FL0AB3C"
    single = [_doc("X1", "BUSINESS_LICENSE",
                   {"bl.company_name": "示例信息技术有限公司", "bl.unified_social_credit_code": uscc}),
              _doc("X2", "BANK_STATEMENT", {"bs.account_name": "示例信息技术公司"}),
              _doc("X3", "INCOME_STATEMENT", {"is.company_name": "示例信息技术公司"})]
    validators = {doc["document_id"]: {} for doc in single}
    _trust.validate_cross_document(single, config, validators)
    hit = {(doc_id, path) for doc_id, by_path in validators.items()
           for path, items in by_path.items() if any(i["id"] == "V-X-01" for i in items)}
    report.check("cross_document.name_rule_covers_all_four_subject_fields",
                 hit == {("X1", "bl.company_name"), ("X2", "bs.account_name"),
                         ("X3", "is.company_name")},
                 f"hit={sorted(hit)}")

    pair = [_doc("Y1", "BUSINESS_LICENSE",
                 {"bl.company_name": "同一名称有限公司", "bl.unified_social_credit_code": uscc}),
            _doc("Y2", "CREDIT_REPORT_ENTERPRISE",
                 {"cr.company_name": "同一名称有限公司",
                  "cr.unified_social_credit_code": "91110000MA01XY2Z8P"})]
    validators = {doc["document_id"]: {} for doc in pair}
    _trust.validate_cross_document(pair, config, validators)
    hit = {(doc_id, path) for doc_id, by_path in validators.items()
           for path, items in by_path.items() if any(i["id"] == "V-X-02" for i in items)}
    report.check("cross_document.uscc_rule_fails_on_the_uscc_fields",
                 hit == {("Y1", "bl.unified_social_credit_code"),
                         ("Y2", "cr.unified_social_credit_code")},
                 f"hit={sorted(hit)}")

    mixed = [_doc("Z1", "BUSINESS_LICENSE",
                  {"bl.company_name": "甲公司", "bl.unified_social_credit_code": uscc}),
             _doc("Z2", "BANK_STATEMENT", {"bs.account_name": "乙公司"}),
             _doc("Z3", "CREDIT_REPORT_ENTERPRISE",
                  {"cr.company_name": "丙公司",
                   "cr.unified_social_credit_code": "91110000MA01XY2Z8P"})]
    validators = {doc["document_id"]: {} for doc in mixed}
    _trust.validate_cross_document(mixed, config, validators)
    quiet = all(not by_path for by_path in validators.values())
    report.check("cross_document.multi_subject_batches_are_not_compared",
                 quiet, f"validators={ {k: v for k, v in validators.items() if v} }")

    field = {"path": "bl.company_name", "label": "企业名称", "value": "示例信息技术有限公司",
             "value_type": "string", "required": True, "risk": "party_name",
             "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
             "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                         "table_score": 1.0, "cross_field_score": 1.0, "grounding_present": 1},
             "meta": {}, "decision": "review", "validators": [], "review_reasons": []}
    doc = _doc("X1", "BUSINESS_LICENSE",
               {"bl.company_name": "示例信息技术有限公司", "bl.unified_social_credit_code": uscc})
    parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
              "pages": [{"page": 1, "width": 595, "height": 842, "quality_flags": [],
                         "raster": None}]}
    items = [{"id": "V-X-01", "status": str(registry["V-X-01"]["severity"]),
              "detail": "same USCC but different company name"}]
    armed = _trust.decide_field(_copy.deepcopy(field), doc, parsed, config,
                                _copy.deepcopy(items), _trust.Calibrator(config))
    loose_config = _copy.deepcopy(config)
    loose_config["cross_document"]["rules"]["V-X-01"]["review"] = False
    relaxed = _trust.decide_field(_copy.deepcopy(field), doc, parsed, loose_config,
                                 _copy.deepcopy(items), _trust.Calibrator(loose_config))
    report.check("cross_document.contract_review_flag_moves_the_gate",
                 "CROSS_FIELD_FAIL" in armed["review_reasons"]
                 and armed["decision"] == "review"
                 and "CROSS_FIELD_FAIL" not in relaxed["review_reasons"],
                 f"armed={armed['decision']}/{armed['review_reasons']} "
                 f"relaxed={relaxed['decision']}/{relaxed['review_reasons']}")

def _check_sample_freeze(report: Report, config: Dict) -> None:
    """SCENARIO 第 109 行：样本清单冻结后不得原地修改；需要修改就新建版本号。

    清单里的 sha256 一直记着，只是从没人拿它和磁盘对过。原地改一份 PDF、忘了改标注，
    指标会假装变好/变坏而流水线一个字都不报；`version` 没升就是「改了没新建版本」。
    """
    if not ((config.get("acceptance") or {}).get("require_sample_freeze", True)):
        return

    import copy as _copy
    import json as _json

    from . import samplefreeze

    result = samplefreeze.audit(config)
    problems = list(result["problems"])
    report.check("samples.manifest_matches_disk_and_version", not problems,
                 f"splits={ {k: v.get('documents') for k, v in result['splits'].items()} } "
                 f"problems={problems[:3]}")

    # 反向：把清单里第一条 sha256 改掉，审计必须点名到那份文档。
    path = samplefreeze.manifest_path("dev", config)
    payload = _json.loads(path.read_text(encoding="utf-8"))
    first = next((item for item in payload.get("documents") or []
                  if item.get("split") in (None, "dev")), None)
    bitten = []
    if first:
        tampered = _copy.deepcopy(payload)
        for item in tampered.get("documents") or []:
            if item.get("document_id") == first.get("document_id"):
                item["sha256"] = "0" * 64
        bitten = samplefreeze.audit_split("dev", config, manifest=tampered)["hash_mismatches"]
    report.check("samples.tampered_hash_is_named",
                 bool(first) and [first.get("document_id")] == [row[0] for row in bitten],
                 f"doc={first.get('document_id') if first else None} bitten={bitten[:2]}")


def _check_split_isolation(report: Report, config: Dict) -> None:
    """SCENARIO 第 108 行「留出集标签只许评估脚本读」的静态+动态门槛。

    调参侧唯一的取数入口是 `calibrate.build_samples`：它按 `labels/<split>` 读标注、
    再从样本拟合校准器与阈值。以前 `--split holdout` 照跑不误，拟合完的产物会拿去
    评估 holdout，指标当然好看——而没有任何一处会报错。现在名单来自配置
    `paths.evaluation_only_splits`，命中就当场拒训。

    三条：名单里必须有留出集、名单上的 split 必须被拒、名单挪到开发集后开发集也必须
    被拒（证明它读配置而不是写死名字）。再附一条静态检查：`calibrate.py` 里不许出现
    硬编码的 `"holdout"`。
    """
    from .calibrate import build_samples as _build
    from .config import evaluation_only_splits as _guarded_splits
    import copy as _copy

    guarded = _guarded_splits(config)
    refused = ""
    try:
        _build("holdout", config)
    except SystemExit as error:
        refused = str(error)
    report.check("split.evaluation_labels_refuse_to_be_fitted",
                 "holdout" in guarded and "留出集" in refused,
                 f"guarded={sorted(guarded)} refused={refused[:80]}")

    moved = _copy.deepcopy(config)
    moved["paths"]["evaluation_only_splits"] = ["dev"]
    moved_refused = ""
    try:
        _build("dev", moved)
    except SystemExit as error:
        moved_refused = str(error)
    report.check("split.the_refusal_follows_the_config",
                 "holdout" not in _guarded_splits(moved)
                 and "留出集" in moved_refused,
                 f"refused={moved_refused[:80]}")

    source = (ROOT / "docpipe" / "calibrate.py").read_text(encoding="utf-8")
    literal = '"holdout"' in source
    report.check("split.no_hardcoded_evaluation_split", not literal,
                 f"literal={literal} labels={'labels' in source}")


def _check_corpus_shape(report: Report, labels_dir: Path, config: Dict) -> None:
    """SCENARIO §9.1 的语料形状，是全部指标的**共同分母**。

    §9.1 把语料写死了：每个 split 10 份、四类按 3/3/2/2、5 份文本层 + 5 份扫描件、
    每类至少 1 份扫描件、每份流水 45～60 笔且 3～4 页、每个 split 至少 120 笔交易。
    此前这道形状只活在 SCENARIO 里：少两格、整个 split 全用文本层、流水砍到 20 笔，
    指标照样在更小的分母上算完并打印。验收一侧只有「交付了多少格」的下限
    （`validation.delivery.min_delivered_fields`）与「别宣称达标」
    （`gate.small_group_not_claimed`），两条都管不到语料设计本身。

    `docpipe.corpusaudit` 只读 `samples/manifest.json` 与标注里的行号（不读值、
    不参与判决、不写文件）；这里守它的三条边界：形状不合 §9.1 要能点名是哪一份
    哪一类、判据全读 `evaluation.corpus`（改配置就能移动门槛）、兜底默认值与出厂
    配置逐字一致（否则删掉一个键会让产品悄悄换成另一套语料形状）。两条隔离证明
    都种在内存里的配置上、不碰磁盘：压紧流水笔数只许 `statement_rows` 红，把成交
    下限抬到「本 split 实际笔数 + 1」只许 `transaction_floor` 红。
    """
    import copy as _copy

    from . import corpusaudit

    shape = corpusaudit.audit_split(labels_dir.name, config)
    codes = [item["code"] if not item["document_id"]
             else f"{item['code']}({item['document_id']})"
             for item in shape["problems"]]
    report.check("corpus.shape_matches_the_contract", not shape["problems"],
                 f"docs={shape['documents']} types={shape['by_doc_type']} "
                 f"formats={shape['by_format']} scans={shape['scans_by_type']} "
                 f"transactions={shape['transactions']} "
                 f"problems={','.join(codes) or 'none'}")

    section = ((config.get("evaluation") or {}).get("corpus")) or {}
    diffs = [key for key, want in corpusaudit.DEFAULTS.items()
             if section.get(key) != want]
    report.check("corpus.defaults_match_shipped_config", not diffs,
                 "diffs=" + (",".join(diffs) or "none"))

    documents = corpusaudit.split_documents(labels_dir.name, config)
    thin = corpusaudit.audit_split(labels_dir.name, config,
                                   documents[:max(1, len(documents) // 2)])
    thin_codes = {item["code"] for item in thin["problems"]}
    report.check("corpus.thin_corpus_is_named",
                 {"document_count", "doc_type_count"} <= thin_codes,
                 f"documents={len(documents)} codes={sorted(thin_codes)}")

    rows = [item["rows"] for item in shape["statements"]]
    tight = _copy.deepcopy(config)
    tight_section = tight.setdefault("evaluation", {}).setdefault("corpus", {})
    tight_section["statement_rows"] = {"min": 0,
                                       "max": (min(rows) - 1) if rows else 0}
    tight_shape = corpusaudit.audit_split(labels_dir.name, tight)
    tight_codes = {item["code"] for item in tight_shape["problems"]}
    report.check("corpus.statement_bounds_follow_the_config",
                 bool(rows) and tight_codes == {"statement_rows"},
                 f"rows={rows} codes={sorted(tight_codes)}")

    floor = shape["transactions"] + 1
    raised = _copy.deepcopy(config)
    raised_section = raised.setdefault("evaluation", {}).setdefault("corpus", {})
    raised_section["min_transactions_per_split"] = floor
    raised_shape = corpusaudit.audit_split(labels_dir.name, raised)
    raised_codes = {item["code"] for item in raised_shape["problems"]}
    report.check("corpus.transaction_floor_follows_the_config",
                 raised_codes == {"transaction_floor"},
                 f"total={shape['transactions']} floor={floor} "
                 f"codes={sorted(raised_codes)}")



def _check_statement_rows(report: Report, config: Dict) -> None:
    """SCENARIO §5.2 V-BS-07：交易笔数与页面数据行一致，重复表头/页脚不得计入。

    解析层过去把「重复表头」「页脚」「首列日期读坏」的行一起静默丢掉，产物里 55 笔
    交易和 48 笔交易长得一模一样。现在每一条丢弃行都带原因与原文，这里守四件事：
    归类真的按行内容分（页脚是稀疏合并行，整行都填满还带「合计」二字的真交易不许
    被放行）；良性原因只认配置；认不出的丢弃让该页交易字段复核；保留下来的行数与
    抽出的笔数对不上，同样点名到页。最后一条查现有解析缓存：每一张流水表都带着
    丢弃清单，且清单里的原因都不在「认不出」那一类。
    """
    import copy as _copy

    from . import parse as _parse
    from . import trust as _trust

    def cell(text):
        return {"text": text, "bbox": [10.0, 10.0, 60.0, 20.0], "confidence": 0.99}

    def row(**values):
        return {"cells": {label: cell(value) for label, value in values.items()}}

    spec = {"headers": ["交易日期", "摘要", "对方户名", "收支方向", "交易金额", "余额"],
            "row_required": ["交易日期", "交易金额"]}

    def keep(rows):
        skipped = []
        kept = _parse._rows_from_cells(rows, spec, "BANK_STATEMENT", 1, "transactions",
                                       skipped=skipped,
                                       footer_options=_parse.footer_rule(config))
        return kept, skipped

    header = row(交易日期="交易日期", 摘要="摘要", 对方户名="对方户名",
                 收支方向="收支方向", 交易金额="交易金额", 余额="余额")
    footer = row(交易日期="第1页共2页", 交易金额="—")
    real = row(交易日期="2026-03-01", 摘要="货款", 对方户名="甲公司",
               收支方向="收入", 交易金额="100.00", 余额="1100.00")
    kept, skipped = keep([header, real, footer])
    report.check("statement_rows.dropped_rows_are_named",
                 len(kept) == 1
                 and [item["reason"] for item in skipped] == ["header", "footer"]
                 and "第1页共2页" in skipped[1]["text"]
                 and all(item["reason"] in _parse.ROW_SKIP_REASONS for item in skipped),
                 f"kept={len(kept)} skipped={skipped}")

    full = row(交易日期="2O26-03-01", 摘要="货款合计", 对方户名="甲公司",
               收支方向="收入", 交易金额="100.00", 余额="1100.00")
    kept_full, skipped_full = keep([full])
    report.check("statement_rows.a_full_row_is_never_a_footer",
                 not kept_full
                 and [item["reason"] for item in skipped_full] == ["unaccounted"],
                 f"kept={len(kept_full)} skipped={skipped_full}")

    def statement(kept_pages, extracted_pages, skips=None):
        fields = {}
        index = 0
        pages = []
        for page_no, count in kept_pages:
            pages.append({"page": page_no, "width": 595.0, "height": 842.0,
                          "quality_flags": [], "lines": [],
                          "tables": [{"table_id": f"p{page_no}-transactions",
                                      "name": "transactions", "page": page_no,
                                      "rows": [{"row_index": item} for item in range(count)],
                                      "skipped_rows": list((skips or {}).get(page_no) or [])}]})
        for page_no, count in extracted_pages:
            for _ in range(count):
                for column, value in (("txn_date", "2026-03-01"), ("amount", "1.00"),
                                      ("balance", "1.00")):
                    path = f"bs.transactions[{index}].{column}"
                    fields[path] = {"path": path, "label": column, "value": value,
                                    "required": True, "risk": "monetary",
                                    "grounding": {"page": page_no}, "signals": {},
                                    "meta": {}, "validators": [], "review_reasons": []}
                index += 1
        doc = {"document_id": "ACC1", "doc_type": "BANK_STATEMENT", "fields": fields}
        return doc, {"document_id": "ACC1", "doc_type": "BANK_STATEMENT", "pages": pages}

    def flagged(doc, parsed):
        validators = {}
        _trust.validate_bank_statement(doc, parsed, config, validators)
        return sorted(path for path, items in validators.items()
                      for item in items if item["id"] == "V-BS-07")

    doc, parsed = statement([(1, 2), (2, 1)], [(1, 2), (2, 1)],
                            {1: [{"row_index": 2, "reason": "footer", "text": "第1页共2页"}]})
    report.check("statement_rows.a_benign_drop_is_left_alone",
                 flagged(doc, parsed) == [], f"flagged={flagged(doc, parsed)}")

    doc, parsed = statement([(1, 2), (2, 1)], [(1, 2), (2, 1)],
                            {2: [{"row_index": 1, "reason": "unaccounted",
                                  "text": "2O26-03-01 货款"}]})
    hits = flagged(doc, parsed)
    report.check("statement_rows.an_unknown_drop_flags_its_page",
                 hits == [f"bs.transactions[{item}].{column}" for item in (2,)
                          for column in ("amount", "balance", "txn_date")],
                 f"flagged={hits}")

    silent = _copy.deepcopy(config)
    silent["validation"]["row_accounting"]["benign_skip_reasons"].append("unaccounted")
    saved = config
    try:
        config = silent
        quiet = flagged(*statement([(1, 2), (2, 1)], [(1, 2), (2, 1)],
                                   {2: [{"row_index": 1, "reason": "unaccounted", "text": "x"}]}))
    finally:
        config = saved
    report.check("statement_rows.the_benign_list_comes_from_the_config",
                 quiet == [], f"flagged={quiet}")

    doc, parsed = statement([(1, 2)], [(1, 1)])
    hits = flagged(doc, parsed)
    report.check("statement_rows.kept_rows_must_equal_extracted_rows",
                 len(hits) == 3 and all(item.startswith("bs.transactions[0].") for item in hits),
                 f"flagged={hits}")

    doc, parsed = statement([(1, 1), (2, 2)], [(1, 1)])
    hits = flagged(doc, parsed)
    report.check("statement_rows.a_lost_page_flags_every_transaction",
                 len(hits) == 3, f"flagged={hits}")

    parsed_dir = Path(resolve_path(config["paths"]["work"], config)) / "parsed"
    files = sorted(parsed_dir.glob("*.json")) if parsed_dir.exists() else []
    benign = {str(item) for item in
              ((config["validation"].get("row_accounting") or {}).get("benign_skip_reasons") or [])}
    missing, unknown, tables = [], [], 0
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if record.get("parser_version") != _parse.PARSER_VERSION:
            continue
        for page in record.get("pages") or []:
            for table in page.get("tables") or []:
                if table.get("name") != "transactions":
                    continue
                tables += 1
                if not isinstance(table.get("skipped_rows"), list):
                    missing.append(path.name)
                    continue
                for skip in table["skipped_rows"]:
                    if str(skip.get("reason")) not in benign:
                        unknown.append(f"{path.name}:{skip.get('reason')}")
    report.check("statement_rows.cached_tables_carry_the_drop_record",
                 bool(tables) and not missing and not unknown,
                 f"tables={tables} missing={sorted(set(missing))[:3]} "
                 f"unknown={sorted(set(unknown))[:5]}")


def _check_statement_recency(report: Report, config: Dict) -> None:
    """SCENARIO §5.3 V-IS-05 与 §5.4 V-X-04：两条规则比的是同一对日期，落地时一个词都不许另挑。

    V-IS-05 在代码里从来没有对应物；V-X-04 曾经登记成 `enforced: false`，理由写的是「批次里
    有多个借款主体时无法归到同一主体」——可归组口径本来就是 V-X-01 那套名称归一化。这里问
    四件事：契约表格里的月份数、严重度词与降级对象等于配置；新配对沉默；期末晚于查询日打在两
    个契约点名的日期上；期间太旧只降级利润表的三个对象；把阈值挪开一句判决都不发。
    """
    from . import trust as _trust

    scenario = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    is_row = re.search(r"^\|\s*V-IS-05\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$",
                       scenario, re.M)
    x_row = re.search(r"^\|\s*V-X-04\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$",
                      scenario, re.M)
    months = int(re.search(r"(\d+)\s*个月", is_row.group(1)).group(1)) if is_row else 0
    is_word = re.split(r"[；（/，,→]", is_row.group(2))[0].strip() if is_row else ""
    x_word = re.split(r"[；（/，,]", x_row.group(2))[0].strip() if x_row else ""
    options = _trust.statement_recency_options(config) or {}
    x04 = _trust.cross_document_rules(config).get("V-X-04") or {}
    degraded = ["is.statement_period_start", "is.statement_period_end", "is.net_profit"]
    report.check("statement_recency.config_matches_the_scenario_row",
                 bool(months) and bool(options)
                 and options.get("severity") == is_word
                 and int(options.get("stale_months", -1)) == months
                 and int(options.get("future_tolerance_days", -1)) == 0
                 and list(options.get("degraded_paths") or []) == degraded
                 and x04.get("severity") == x_word
                 and sorted(x04.get("paths") or []) == sorted(_trust.STATEMENT_RECENCY_PATHS),
                 f"scenario={is_word}/{months}months code={options.get('severity')}/"
                 f"{options.get('stale_months')}months degraded={options.get('degraded_paths')} "
                 f"x04={x04.get('severity')}/{x04.get('paths')}")

    def _doc(doc_id, doc_type, values):
        return {"document_id": doc_id, "doc_type": doc_type,
                "fields": {path: {"path": path, "value": value} for path, value in values.items()}}

    def _pair(period_end, query, period_start="2023-01-01", statement_name="示例科技有限公司",
              report_name="示例科技有限公司", drop_query=False):
        statement = _doc("S1", "INCOME_STATEMENT",
                         {"is.company_name": statement_name,
                          "is.statement_period_start": period_start,
                          "is.statement_period_end": period_end,
                          "is.net_profit": "100.00"})
        values = {"cr.company_name": report_name}
        if not drop_query:
            values["cr.query_date"] = query
        return [statement, _doc("R1", "CREDIT_REPORT_ENTERPRISE", values)]

    def _hits(docs, cfg=None):
        validators = {doc["document_id"]: {} for doc in docs}
        _trust.validate_cross_document(docs, cfg or config, validators)
        return {(doc_id, path): [f"{item['id']}/{item['status']}" for item in items]
                for doc_id, by_path in validators.items()
                for path, items in by_path.items()}

    report.check("statement_recency.a_fresh_pair_stays_silent",
                 _hits(_pair("2025-12-31", "2026-01-05")) == {},
                 f"hits={_hits(_pair('2025-12-31', '2026-01-05'))}")

    future = _hits(_pair("2026-03-31", "2025-12-13", period_start="2025-01-01"))
    report.check("statement_recency.future_period_warns_both_contract_paths",
                 future == {("S1", "is.statement_period_end"): ["V-X-04/warn"],
                            ("R1", "cr.query_date"): ["V-X-04/warn"]},
                 f"hits={future}")

    stale = _hits(_pair("2023-12-31", "2026-01-05"))
    report.check("statement_recency.stale_period_degrades_the_three_contract_paths",
                 stale == {("S1", path): ["V-IS-05/warn"] for path in degraded},
                 f"hits={stale}")

    loose = copy.deepcopy(config)
    loose["validation"]["statement_recency"]["stale_months"] = 30
    report.check("statement_recency.threshold_moves_the_gate",
                 _hits(_pair("2023-12-31", "2026-01-05"), loose) == {},
                 f"hits={_hits(_pair('2023-12-31', '2026-01-05'), loose)}")

    unpaired = (_hits(_pair("2023-12-31", "2026-01-05", report_name="另一家公司"))
                | _hits(_pair("2023-12-31", "2026-01-05", drop_query=True)))
    report.check("statement_recency.unpaired_or_undated_documents_stay_silent",
                 unpaired == {}, f"hits={unpaired}")

    def _field(body):
        return {"path": "is.net_profit", "label": "净利润", "value": "100.00",
                "value_type": "decimal", "required": True, "risk": "monetary",
                "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
                "signals": dict({"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                                 "table_score": 1.0, "cross_field_score": 1.0,
                                 "grounding_present": 1}, **body.get("signals", {})),
                "meta": {}, "decision": "review", "validators": [], "review_reasons": []}

    doc = _doc("S1", "INCOME_STATEMENT",
               {"is.company_name": "示例科技有限公司", "is.statement_period_start": "2023-01-01",
                "is.statement_period_end": "2023-12-31", "is.net_profit": "100.00"})
    parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
              "pages": [{"page": 1, "width": 595, "height": 842, "quality_flags": [],
                         "raster": None}]}
    items = [{"id": "V-IS-05", "status": str(options.get("severity")), "detail": "25 months"}]
    armed = _trust.decide_field(_field({}), doc, parsed, config, copy.deepcopy(items),
                                _trust.Calibrator(config))
    quiet = _trust.decide_field(_field({}), doc, parsed, config, [],
                                _trust.Calibrator(config))
    report.check("statement_recency.warn_lowers_the_cross_field_signal",
                 armed["signals"]["cross_field_score"] == 0.3
                 and quiet["signals"]["cross_field_score"] == 1.0
                 and "CROSS_FIELD_FAIL" not in armed["review_reasons"],
                 f"armed={armed['decision']}/{armed['signals']['cross_field_score']} "
                 f"quiet={quiet['decision']}/{quiet['signals']['cross_field_score']}")


def _check_income_columns(report: Report, config: Dict) -> None:
    """SCENARIO §5.3 V-IS-04 后半句：本期列与上期列不得混列。

    这条规则过去只有前半句（币种确认不了就全部金额复核）；「混列」在代码里没有落点，
    而版面唯一能自证列身份的证据是金额框的横向位置。这里问六件事：契约点名的两列与
    「全部金额复核」这个去处等于配置；分得开的版式沉默；落在另一列带里的字段把全表
    金额置 V-IS-04 fail；两列近到分不开也算无法确认；只有一列时条款自然满足；关掉
    开关就整段沉默。
    """
    from . import trust as _trust

    scenario = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    row = re.search(r"^\|\s*V-IS-04\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$",
                    scenario, re.M)
    body, action = (row.group(1), row.group(2)) if row else ("", "")
    options = _trust.income_column_options(config) or {}
    columns = list(options.get("period_columns") or [])
    income = (((config.get("doc_types") or {}).get("INCOME_STATEMENT") or {})
              .get("tables") or {}).get("income") or {}
    declared = [str(header) for header in (income.get("headers") or []) if header != "项目"]
    report.check("income_columns.config_matches_the_scenario_row",
                 bool(row) and "全部金额复核" in action
                 and bool(options.get("enabled"))
                 and columns == declared
                 and all(name.replace("金额", "") + "列" in body for name in columns),
                 f"scenario={action!r} config={columns} declared={declared}")

    def _doc(boxes):
        fields = {path: {"path": path, "value": "100.00", "value_type": "decimal",
                         "grounding": {"page": 1, "bbox": bbox}}
                  for path, bbox in boxes.items()}
        return {"document_id": "S1", "doc_type": "INCOME_STATEMENT",
                "currency_unit": "CNY", "fields": fields}

    def _hits(doc, cfg=None):
        validators = {}
        _trust.validate_income_statement(doc, cfg or config, validators)
        return {path: [f"{item['id']}/{item['status']}" for item in items]
                for path, items in validators.items()}

    clean = {"is.revenue": [320, 10, 450, 20], "is.cost_of_sales": [320, 20, 450, 30],
             "is.prior_revenue": [450, 10, 580, 20]}
    report.check("income_columns.a_separated_layout_stays_silent",
                 _hits(_doc(clean)) == {}, f"hits={_hits(_doc(clean))}")

    crossed = dict(clean, **{"is.cost_of_sales": [450, 20, 580, 30]})
    hits = _hits(_doc(crossed))
    report.check("income_columns.a_field_in_the_other_column_fails_every_amount",
                 set(hits) == set(crossed)
                 and all(items == ["V-IS-04/fail"] for items in hits.values()),
                 f"hits={hits}")

    close = {"is.revenue": [320, 10, 450, 20], "is.prior_revenue": [330, 10, 460, 20]}
    report.check("income_columns.indistinguishable_columns_are_unconfirmed",
                 set(_hits(_doc(close))) == set(close)
                 and all(items == ["V-IS-04/fail"]
                         for items in _hits(_doc(close)).values()),
                 f"hits={_hits(_doc(close))}")

    single = {"is.revenue": [320, 10, 450, 20], "is.net_profit": [320, 20, 450, 30]}
    report.check("income_columns.one_amount_column_has_nothing_to_mix",
                 _hits(_doc(single)) == {}, f"hits={_hits(_doc(single))}")

    off = copy.deepcopy(config)
    off["validation"]["income_columns"]["enabled"] = False
    report.check("income_columns.the_switch_comes_from_the_config",
                 _hits(_doc(crossed), off) == {}, f"hits={_hits(_doc(crossed), off)}")

    field = {"path": "is.revenue", "label": "营业收入", "value": "100.00",
             "value_type": "decimal", "required": True, "risk": "monetary",
             "grounding": {"page": 1, "bbox": [320, 10, 450, 20]},
             "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                         "table_score": 1.0, "cross_field_score": 1.0,
                         "grounding_present": 1},
             "meta": {}, "decision": "review", "validators": [], "review_reasons": []}
    parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
              "pages": [{"page": 1, "width": 595, "height": 842, "quality_flags": [],
                         "raster": None}]}
    statement = {"document_id": "S1", "doc_type": "INCOME_STATEMENT",
                 "currency_unit": "CNY", "fields": {"is.revenue": field}}
    items = [{"id": "V-IS-04", "status": "fail", "detail": "period columns not separable"}]
    decided = _trust.decide_field(field, statement, parsed, config, items,
                                  _trust.Calibrator(config))
    report.check("income_columns.mixing_forces_review_with_the_cross_field_code",
                 decided["decision"] == "review"
                 and "CROSS_FIELD_FAIL" in decided["review_reasons"],
                 f"decision={decided['decision']} reasons={decided['review_reasons']}")


def _check_income_amounts(report: Report, config: Dict) -> None:
    """SCENARIO §4.5 利润表字段域：非负金额、期间顺序、所得税软上限。

    这三条硬校验过去只写在契约字段表里，代码只核对三条恒等式——而恒等式只证明「加减
    没算错」。把整张表的符号一起翻过来（营业收入 -1000、营业成本 -600、营业利润 -400），
    V-IS-01/02/03 依然全过，每个金额照样自动接受。这里问七件事：字段表里写着「≥ 0」的
    行与配置里挂了 NON_NEGATIVE 的字段逐条对得上；负数金额按 V-ID-05 拒收；符号整体
    翻转时恒等式全过而字段域把它拦下；契约写「可为负」的格子沉默；期间倒置走 V-ID-04；
    所得税高过利润总额只 warn 不阻断；以及「哪些字段不许为负」真的读字段表。
    """
    from . import trust as _trust

    scenario = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    section = scenario.split("### 4.5 利润表", 1)[-1].split("### 4.6", 1)[0]
    declared = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].startswith("`is."):
            declared[cells[0].strip("`")] = cells[6]
    expected_nonneg = {path for path, hard in declared.items() if "≥ 0" in hard}
    specs = (((config.get("doc_types") or {}).get("INCOME_STATEMENT") or {})
             .get("fields") or [])
    configured_nonneg = {spec["path"] for spec in specs
                         if "NON_NEGATIVE" in (spec.get("rules") or [])}
    ceiling = (config.get("validation") or {}).get("income_tax_ceiling") or {}
    report.check("income_amounts.config_matches_the_scenario_row",
                 bool(declared) and expected_nonneg == configured_nonneg
                 and "≤ 利润总额（软）" in declared.get("is.income_tax", "")
                 and bool(ceiling.get("enabled")) and str(ceiling.get("severity")) == "warn",
                 f"scenario={sorted(expected_nonneg)} config={sorted(configured_nonneg)} "
                 f"ceiling={ceiling}")

    def _doc(values, boxes=None):
        boxes = boxes or {}
        fields = {}
        for index, (path, value) in enumerate(values.items()):
            fields[path] = {
                "path": path, "value": value,
                "value_type": "date" if path.endswith(("_start", "_end")) else "decimal",
                "grounding": {"page": 1, "bbox": boxes.get(
                    path, [320, 10 + index * 10, 450, 20 + index * 10])}}
        return {"document_id": "S1", "doc_type": "INCOME_STATEMENT",
                "currency_unit": "CNY", "fields": fields}

    def _hits(doc, cfg=None):
        validators = {}
        _trust.validate_document_fields(doc, {"pages": []}, cfg or config, validators)
        _trust.validate_income_statement(doc, cfg or config, validators)
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in validators.items() if items}

    negative = _hits(_doc({"is.revenue": "-100.00"}))
    report.check("income_amounts.a_negative_amount_fails_the_contract_rule",
                 negative.get("is.revenue") == ["V-ID-05/fail"], f"hits={negative}")

    zero = _hits(_doc({"is.revenue": "0.00"}))
    report.check("income_amounts.zero_is_a_legal_amount", zero == {}, f"hits={zero}")

    flipped = _hits(_doc({"is.revenue": "-1000.00", "is.cost_of_sales": "-600.00",
                          "is.taxes_surcharges": "0.00", "is.selling_expenses": "0.00",
                          "is.admin_expenses": "0.00", "is.finance_expenses": "0.00",
                          "is.operating_profit": "-400.00",
                          "is.non_operating_income": "0.00",
                          "is.non_operating_expense": "0.00",
                          "is.total_profit": "-400.00", "is.income_tax": "0.00",
                          "is.net_profit": "-400.00"}))
    identities = [item for items in flipped.values() for item in items
                  if item.startswith("V-IS-0")]
    report.check("income_amounts.a_sign_flip_passes_identities_but_fails_the_domain",
                 bool(identities)
                 and not any(item.endswith("/fail") for item in identities)
                 and "V-IS-01/pass" in (flipped.get("is.revenue") or [])
                 and "V-ID-05/fail" in (flipped.get("is.revenue") or [])
                 and "V-ID-05/fail" in (flipped.get("is.cost_of_sales") or [])
                 and "V-IS-03/pass" in (flipped.get("is.net_profit") or []),
                 f"hits={flipped}")

    allowed = _hits(_doc({"is.finance_expenses": "-12.50", "is.other_income": "-3.00",
                          "is.investment_income": "-1.00", "is.net_profit": "-900.00",
                          "is.prior_net_profit": "-900.00"},
                         {"is.prior_net_profit": [580, 10, 710, 20]}))
    report.check("income_amounts.fields_the_contract_allows_negative_stay_silent",
                 allowed == {}, f"hits={allowed}")

    reversed_period = _hits(_doc({"is.statement_period_start": "2024-12-31",
                                  "is.statement_period_end": "2024-01-01"}))
    report.check("income_amounts.a_reversed_period_fails_the_date_rule",
                 reversed_period.get("is.statement_period_start") == ["V-ID-04/fail"]
                 and reversed_period.get("is.statement_period_end") == ["V-ID-04/fail"],
                 f"hits={reversed_period}")

    taxed = _hits(_doc({"is.total_profit": "100.00", "is.income_tax": "150.00",
                        "is.net_profit": "-50.00"}))
    loss = _hits(_doc({"is.total_profit": "-100.00", "is.income_tax": "0.00",
                       "is.net_profit": "-100.00"}))
    report.check("income_amounts.income_tax_over_profit_only_warns",
                 "V-IS-06/warn" in (taxed.get("is.income_tax") or [])
                 and "V-IS-03/pass" in (taxed.get("is.total_profit") or [])
                 and "V-IS-06/warn" not in
                 [item for items in loss.values() for item in items],
                 f"hits={taxed} loss={loss}")

    relaxed = copy.deepcopy(config)
    for spec in relaxed["doc_types"]["INCOME_STATEMENT"]["fields"]:
        if spec["path"] == "is.revenue":
            spec["rules"] = ["MONEY"]
    relaxed_hits = _hits(_doc({"is.revenue": "-100.00"}), relaxed)
    report.check("income_amounts.the_non_negative_list_comes_from_the_field_table",
                 relaxed_hits == {}, f"hits={relaxed_hits}")


def _check_string_lengths(report: Report, config: Dict) -> None:
    """SCENARIO §4.3/§4.4 字段表里写死的中文长度域（「长度 2～80」「非空时长度 ≥ 4」）。

    这些数字过去只躺在契约表格里：一个字的「企业名称」（标签被切走半截）非空、有坐标、
    过得了所有格式规则，于是自动接受。这里问五件事：字段表声明的上下限与配置逐条对得上、
    过短的串按 V-ID-09 拒收、恰好落在边界上的串放行、契约没写长度的字段沉默、以及
    上下限真的读配置（收紧或删掉键，判决跟着翻面）。
    """
    from . import trust as _trust

    def _declared(section: str) -> Dict[str, tuple]:
        text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
        body = text.split(section, 1)[-1].split("\n### ", 1)[0]
        found: Dict[str, tuple] = {}
        for line in body.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 7 or not cells[0].startswith("`"):
                continue
            path = cells[0].strip("`")
            if not path.startswith(("bl.", "bs.")):
                continue
            hard = cells[6]
            span = re.search(r"长度\s*(\d+)\s*[～~-]\s*(\d+)", hard)
            floor = re.search(r"长度\s*≥\s*(\d+)", hard)
            if span:
                found[re.sub(r"\[\d+\]", "[i]", path)] = (int(span.group(1)), int(span.group(2)))
            elif floor:
                found[re.sub(r"\[\d+\]", "[i]", path)] = (int(floor.group(1)), None)
        return found

    declared = {}
    declared.update(_declared("### 4.3 营业执照"))
    declared.update(_declared("### 4.4 企业对公银行流水"))
    configured: Dict[str, tuple] = {}
    for doc_type in ("BUSINESS_LICENSE", "BANK_STATEMENT"):
        specs = (((config.get("doc_types") or {}).get(doc_type) or {}).get("fields")) or []
        for spec in specs:
            if spec.get("min_len") is None and spec.get("max_len") is None:
                continue
            configured[re.sub(r"\[\d+\]", "[i]", spec["path"])] = (
                spec.get("min_len"), spec.get("max_len"))
    report.check("string_lengths.registry_matches_the_scenario_tables",
                 bool(declared) and declared == configured,
                 f"scenario={sorted(declared.items())} config={sorted(configured.items())}")

    def _doc(values, doc_type="BUSINESS_LICENSE"):
        fields = {}
        for index, (path, value) in enumerate(values.items()):
            fields[path] = {
                "path": path, "value": value, "value_type": "string",
                "grounding": {"page": 1, "bbox": [320, 10 + index * 10, 450, 20 + index * 10]}}
        return {"document_id": "S1", "doc_type": doc_type, "fields": fields}

    def _hits(doc, cfg=None):
        validators = {}
        _trust.validate_document_fields(doc, {"pages": []}, cfg or config, validators)
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in validators.items() if items}

    short = _hits(_doc({"bl.company_name": "司"}))
    report.check("string_lengths.a_too_short_name_fails_the_domain",
                 short == {"bl.company_name": ["V-ID-09/fail"]}, f"hits={short}")

    boundary = _hits(_doc({"bl.company_name": "阿里", "bl.legal_representative": "张三"}))
    report.check("string_lengths.a_name_at_the_boundary_passes", boundary == {},
                 f"hits={boundary}")

    silent = _hits(_doc({"is.company_name": "司"}, doc_type="INCOME_STATEMENT"))
    report.check("string_lengths.fields_without_declared_bounds_stay_silent",
                 silent == {}, f"hits={silent}")

    tight = copy.deepcopy(config)
    for spec in tight["doc_types"]["BUSINESS_LICENSE"]["fields"]:
        if spec["path"] == "bl.company_name":
            spec["max_len"] = 2
    tightened = _hits(_doc({"bl.company_name": "阿里巴"}), tight)
    report.check("string_lengths.the_bounds_come_from_the_config",
                 tightened == {"bl.company_name": ["V-ID-09/fail"]}, f"hits={tightened}")


def _check_enum_review_values(report: Report, config: Dict) -> None:
    """SCENARIO §4.6 的「未知则复核」：unknown 是合法枚举值，但不是可用结论。

    §4.6 把呆账 / 资产处置 / 对外担保三面标志的硬校验写成「未知则复核」。抽取侧把
    「未知 / 不详 / -」归一成枚举值 unknown，而 V-ID-07 只问「是不是配置里认的词」——
    于是文件自己说「不知道」的标志可以自动接受，在审查里那和读成「无」是两件事。
    这里问四件事：字段表的「未知则复核」行与配置点名逐条对得上、unknown 走 V-ID-10、
    是/否正常沉默、点名删掉就翻面。
    """
    from . import trust as _trust

    text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
    body = text.split("### 4.6 ", 1)[-1].split("\n### ", 1)[0]
    declared = []
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7 or not cells[0].startswith("`"):
            continue
        if "未知则复核" in cells[6]:
            declared.append(cells[0].strip("`"))
    specs = {}
    for spec in (((config.get("doc_types") or {}).get("CREDIT_REPORT_ENTERPRISE")
                  or {}).get("fields")) or []:
        specs[spec["path"]] = spec
    hold = config["validation"].get("enum_review_values") or {}
    covered = {path: (specs.get(path) or {}).get("enum") for path in declared}
    report.check("enum_unknown.registry_matches_the_scenario_tables",
                 bool(declared) and all(enum and hold.get(enum) for enum in covered.values()),
                 f"declared={declared} enums={covered} config={hold}")

    def _doc(values, doc_type="CREDIT_REPORT_ENTERPRISE"):
        fields = {}
        for index, (path, value) in enumerate(values.items()):
            fields[path] = {
                "path": path, "value": value, "value_type": "string",
                "grounding": {"page": 1, "bbox": [120, 10 + index * 12, 360, 22 + index * 12]}}
        return {"document_id": "E1", "doc_type": doc_type, "fields": fields}

    def _hits(doc, cfg=None):
        validators = {}
        _trust.validate_document_fields(doc, {"pages": []}, cfg or config, validators)
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in validators.items() if items}

    unknown = _hits(_doc({"cr.bad_debt_flag": "unknown"}))
    report.check("enum_unknown.the_unknown_value_fails_the_domain",
                 unknown == {"cr.bad_debt_flag": ["V-ID-10/fail"]}, f"hits={unknown}")

    ordinary = _hits(_doc({"cr.asset_disposal_flag": "no", "cr.guarantee_flag": "yes"}))
    report.check("enum_unknown.ordinary_values_stay_silent", ordinary == {},
                 f"hits={ordinary}")

    relaxed = copy.deepcopy(config)
    relaxed["validation"]["enum_review_values"] = {}
    silent = _hits(_doc({"cr.bad_debt_flag": "unknown"}), relaxed)
    report.check("enum_unknown.the_list_comes_from_the_config", silent == {},
                 f"hits={silent}")


def _check_scenario_contract(report: Report, config: Dict) -> None:
    """SCENARIO §13 是契约的一部分：契约列出的配置键必须存在、值一致、有人读。

    与 configaudit 方向相反：那边问出厂 YAML 里的键有没有人读，这里问契约那一头。
    三问各用一棵反向树证明会咬：一条 FAIL 只指向一类问题。`output.include_parsed_debug`
    是已批准偏差点，它必须真的在契约里找得到，否则打错的豁免名会静默放行一条不存在的键。
    """
    from . import calibrate as _calibrate
    from . import configcontract as _contract
    from . import parse as _parse

    spec = _contract.section13_spec((ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
    source = _contract.source_text()
    real = _contract.audit(spec, config, source)
    report.check(
        "config_keys.scenario_section13_holds",
        bool(real["leaves"] >= 25 and not real["missing"] and not real["mismatched"]
             and not real["unread"] and not real["stale_deviations"]),
        _contract.render(real).replace("\n", " | "))

    doctored = copy.deepcopy(config)
    del doctored["parsing"]["table_engine"]
    missing = _contract.audit(spec, doctored, source)["missing"]
    report.check("config_keys.a_contract_key_missing_from_yaml_bites",
                 missing == ["parsing.table_engine"], f"missing={missing}")

    doctored = copy.deepcopy(config)
    doctored["thresholds"]["global_min_confidence"] = 0.5
    mismatched = _contract.audit(spec, doctored, source)["mismatched"]
    report.check("config_keys.a_contract_value_drift_bites",
                 len(mismatched) == 1
                 and mismatched[0].startswith("thresholds.global_min_confidence"),
                 f"mismatched={mismatched}")

    spec_bad = copy.deepcopy(spec)
    doctored = copy.deepcopy(config)
    spec_bad["parsing"]["ghost_engine"] = "unregistered"
    doctored["parsing"]["ghost_engine"] = "unregistered"
    unread = _contract.audit(spec_bad, doctored, source)["unread"]
    report.check("config_keys.a_key_nobody_reads_bites",
                 unread == ["parsing.ghost_engine"], f"unread={unread}")

    guards = []
    bad = copy.deepcopy(config)
    bad["parsing"]["pdf_text_engine"] = "docling"
    try:
        _parse.text_engine(bad)
        guards.append("text_engine")
    except ValueError:
        pass
    bad = copy.deepcopy(config)
    bad["parsing"]["table_engine"] = "camelot"
    try:
        _parse.table_engine(bad)
        guards.append("table_engine")
    except ValueError:
        pass
    bad = copy.deepcopy(config)
    bad["calibration"]["group_by_document"] = False
    try:
        _calibrate.require_document_grouping(bad)
        guards.append("group_by_document")
    except ValueError:
        pass
    bad = copy.deepcopy(config)
    bad["calibration"]["method"] = "svm"
    try:
        _calibrate.calibration_method(bad)
        guards.append("method")
    except ValueError:
        pass
    report.check("config_keys.declared_engines_and_methods_reject_unknown",
                 guards == [], f"没有报错的守卫={guards}")


def _check_label_contract(report: Report, config: Dict) -> None:
    """SCENARIO §9.2 的标注格式要拿得出：出厂标注干净，篡改一条就咬。

    `evaluate.grounding_ok` 直接取 `gt["bbox"]`，`page` 对不上即判未溯源——标注少
    `bbox` 是当场 KeyError，少 `page` 是静默把整份记成未溯源。这里先核出厂标注，
    再用真实标注逐项只改一处，每处只许翻出一个理由码。
    """
    from . import labelaudit as _labels

    real = _labels.audit(config)
    splits = real["splits"]
    total_fields = sum(shape["fields"] for shape in splits.values())
    unchecked = sum(shape["fields"] - shape["bounds_checked"] for shape in splits.values())
    report.check(
        "labels.section9_2_contract_holds",
        bool(splits and total_fields and not real["problems"] and not unchecked),
        f"分片={ {name: (shape['documents'], shape['fields']) for name, shape in splits.items()} }"
        f" 字段={total_fields} 越界未核={unchecked} 问题={real['problems']}")

    doc = label = sizes = None
    for candidate in _labels.split_documents("dev", config):
        found, error = _labels.label_of(candidate, config)
        if error or not (found or {}).get("fields"):
            continue
        doc, label = candidate, found
        sizes = _labels.page_sizes(candidate.get("document_id"), config)
        if sizes:
            break
    if doc is None or not sizes:
        report.check("labels.a_label_without_bbox_bites", False, "没有可用于篡改的 dev 标注")
        return

    def codes(mutate) -> List[str]:
        doctored = copy.deepcopy(label)
        mutate(doctored)
        shape = _labels.audit_label(doc, doctored, config, sizes)
        return [problem["code"] for problem in shape["problems"]]

    path = next(iter(label["fields"]))

    def bbox_missing(item):
        item["fields"][path].pop("bbox")

    report.check("labels.a_label_without_bbox_bites",
                 codes(bbox_missing) == ["field_bbox_missing"],
                 f"{doc['document_id']}:{path} 去掉 bbox → {codes(bbox_missing)}")

    def page_beyond(item):
        item["fields"][path]["page"] = int(doc["page_count"]) + 1

    report.check("labels.a_page_beyond_page_count_bites",
                 codes(page_beyond) == ["field_page_out_of_range"],
                 f"{doc['document_id']}:{path} page={doc['page_count'] + 1} → {codes(page_beyond)}")

    def collapsed(item):
        item["fields"][path]["bbox"] = [0, 0, 0, 0]

    report.check("labels.a_collapsed_bbox_bites",
                 codes(collapsed) == ["field_bbox_degenerate"],
                 f"{doc['document_id']}:{path} bbox 塌成零面积 → {codes(collapsed)}")

    def off_page(item):
        item["fields"][path]["bbox"] = [10000, 10, 10100, 20]

    report.check("labels.a_bbox_off_the_page_bites",
                 codes(off_page) == ["field_bbox_outside"],
                 f"{doc['document_id']}:{path} bbox 越出页宽 → {codes(off_page)}")

    def ghost_path(item):
        item["fields"]["bl.ghost_field"] = copy.deepcopy(item["fields"][path])

    report.check("labels.an_unregistered_field_path_bites",
                 codes(ghost_path) == ["field_path_unknown"],
                 f"{doc['document_id']} 加一条 schema 外路径 → {codes(ghost_path)}")

    def ghost_type(item):
        item["doc_type"] = "GHOST_DOC_TYPE"

    report.check("labels.an_unknown_doc_type_bites",
                 codes(ghost_type) == ["doc_type_unknown"],
                 f"{doc['document_id']} doc_type 改成未知类型 → {codes(ghost_type)}")


def _check_csv_contract(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §8.4 的 `fields.csv` 是给人/给 Excel 的交付物：列序与编码写死了。

    契约 11 列写在 `output.csv_columns`，已登记的三列扩展写在 `output.csv_appendix`；
    这里核配置前缀是否逐字等于 §8.4、实际表头是否等于配置拼接、BOM 在不在、行是否逐字段
    覆盖且取值合法。篡改在内存里做（`csv_bytes`），不写临时目录。
    """
    from . import csvaudit as _csv

    spec = _csv.section84_columns((ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
    columns = _csv.expected_columns(config)
    real = _csv.audit(out_dir, config)
    report.check(
        "output.section84_columns_hold",
        bool(spec and columns[:len(spec)] == spec and not real["problems"]),
        f"契约列={spec} 配置={columns} 行={real['rows']} 字段={real.get('fields', 0)}"
        f" 问题={[item['code'] for item in real['problems']]}")

    raw = (out_dir / "fields.csv").read_bytes()
    body = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(body)))
    header, data = rows[0], rows[1:]

    def rewrote(new_header=None, new_rows=None, bom=True) -> List[str]:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows([new_header or header] + (new_rows if new_rows is not None else data))
        payload = buffer.getvalue().encode("utf-8")
        payload = (b"\xef\xbb\xbf" if bom else b"") + payload
        return [item["code"] for item in _csv.audit(out_dir, config, csv_bytes=payload)["problems"]]

    swapped = list(header)
    swapped[5], swapped[6] = swapped[6], swapped[5]
    reordered, dropped = rewrote(new_header=swapped), rewrote(new_header=header[:-1])
    report.check("output.csv_header_drift_bites",
                 {"header_order"} <= set(reordered) and "header_missing" in dropped,
                 f"对调 conf/decision → {sorted(set(reordered))} 删末列 → {sorted(set(dropped))}")

    stripped = rewrote(bom=False)
    report.check("output.csv_encoding_bites",
                 "encoding_mismatch" in stripped and "encoding_mismatch" not in rewrote(),
                 f"去掉 BOM → {sorted(set(stripped))}")

    lost = rewrote(new_rows=data[1:])
    doubled = rewrote(new_rows=data + data[:1])
    report.check("output.csv_rows_cover_every_field",
                 "row_missing" in lost and "row_duplicate" in doubled
                 and real["rows"] == real.get("fields", -1),
                 f"删一行 → {sorted(set(lost))} 重复一行 → {sorted(set(doubled))}"
                 f" 行={real['rows']} 字段={real.get('fields', 0)}")

    decided = [list(row) for row in data]
    index = next(index for index, row in enumerate(decided) if row[6] == "accept")
    decided[index][6] = "maybe"
    typed = [list(row) for row in data]
    typed[index][4] = ""
    boxed = [list(row) for row in data]
    boxed[index][8] = "[1, 2]"
    decision_codes = set(rewrote(new_rows=decided))
    type_codes = set(rewrote(new_rows=typed))
    bbox_codes = set(rewrote(new_rows=boxed))
    report.check("output.csv_row_values_bite",
                 {"decision_invalid"} <= decision_codes and "value_type_missing" in type_codes
                 and "bbox_shape" in bbox_codes,
                 f"decision=maybe → {sorted(decision_codes)} value_type 空 → {sorted(type_codes)}"
                 f" bbox 两个数 → {sorted(bbox_codes)}")


def _check_summary_contract(report: Report, out_dir: Path, config: Dict) -> None:
    from . import summaryaudit as _summary

    real = _summary.audit(out_dir, config)
    shipped = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    report.check("output.section83_keys_hold",
                 real["keys"] >= 8 and not real["problems"],
                 f"键={real['keys']} 文档={real['documents']} 行={real.get('rows')}"
                 f" 问题={real['problems'][:3]}")

    def doctored(mutate) -> List[str]:
        payload = copy.deepcopy(shipped)
        mutate(payload)
        shape = _summary.audit(out_dir, config, summary=payload)
        return [item["code"] for item in shape["problems"]]

    dropped = doctored(lambda it: it["overall"].pop("review_count"))
    broken = doctored(lambda it: it["overall"].__setitem__(
        "accept_count", it["overall"]["accept_count"] + 1))
    report.check("output.summary_key_drop_bites",
                 "key_missing" in dropped and "count_mismatch" in dropped
                 and "count_mismatch" in broken and "rate_mismatch" in broken,
                 f"删 review_count -> {dropped} / 改 accept_count -> {broken[:3]}")

    def legacy_rate(it):
        """把老口径（全部字段）冒充 auto_accept_rate；两个数相等时换一个能区分的值。"""
        overall = it["overall"]
        legacy = round(float(overall.get("field_accept_rate") or 0.0), 4)
        current = overall.get("auto_accept_rate")
        overall["auto_accept_rate"] = legacy if legacy != current else round(legacy + 0.01, 4)

    legacy_codes = doctored(legacy_rate)
    no_required = doctored(lambda it: it["overall"].pop("required_field_count"))
    report.check("output.summary_dual_rate_bites",
                 "rate_mismatch" in legacy_codes and "count_mismatch" in no_required,
                 f"老口径当 auto_accept_rate -> {legacy_codes[:3]}"
                 f" / 删必填分母 -> {no_required[:3]}")

    def ghost(it):
        it["documents"].append({"document_id": "Z999", "doc_type": "BANK_STATEMENT",
                                "field_count": 0, "path": str(out_dir / "Z999" / "Z999.json")})
        it["run"]["document_count"] = it["run"]["document_count"] + 1

    def unlist(it):
        it["documents"] = it["documents"][1:]
        it["run"]["document_count"] = it["run"]["document_count"] - 1

    ghost_codes = doctored(lambda it: (ghost(it), count_fix(it)))
    unlist_codes = doctored(unlist)
    extra_row = _summary.audit(out_dir, config,
                               csv_count=(real.get("rows") or 0) + 1)
    report.check("output.summary_layout_bites",
                 "document_missing" in ghost_codes
                 and "document_unlisted" in unlist_codes
                 and "csv_rows_mismatch" in [item["code"] for item in extra_row["problems"]],
                 f"幽灵文档 -> {ghost_codes[:3]} / 漏列 -> {unlist_codes[:3]}"
                 f" / 多一行 -> {[item['code'] for item in extra_row['problems']]}")


def _check_document_contract(report: Report, out_dir: Path, config: Dict) -> None:
    from . import docaudit as _docs

    real = _docs.audit(out_dir, config)
    report.check("output.section81_shape_holds",
                 real["documents"] >= 10 and real["keys"] >= 15
                 and real["fields"] >= 1000 and not real["problems"],
                 f"文档={real['documents']} 字段={real['fields']} 键={real['keys']}"
                 f" 问题={real['problems'][:3]}")

    base = json.loads(_docs.document_files(out_dir)[0].read_text(encoding="utf-8"))

    def doctored(mutate) -> List[str]:
        payload = copy.deepcopy(base)
        mutate(payload)
        return [item["code"] for item in _docs.audit(out_dir, config, document=payload)["problems"]]

    def first_decimal(doc) -> str:
        for path, field in doc["fields"].items():
            if field.get("value_type") == "decimal":
                return path
        raise AssertionError("语料里没有 decimal 字段")

    def to_float(doc):
        doc["fields"][first_decimal(doc)]["value"] = 12500.0

    def bad_precision(doc):
        doc["fields"][first_decimal(doc)]["confidence"] = 0.12345

    def bad_decision(doc):
        doc["fields"][first_decimal(doc)]["decision"] = "maybe"

    def no_offset(doc):
        doc["extraction_timestamp"] = "2026-10-29T02:39:20"

    def stray_key(doc):
        doc["mystery"] = 1

    def empty_value(doc):
        doc["fields"][first_decimal(doc)]["value"] = ""

    def blank_value(doc):
        doc["fields"][first_decimal(doc)]["value"] = "   "

    types = {
        "decimal": doctored(to_float),
        "precision": doctored(bad_precision),
        "decision": doctored(bad_decision),
        "timestamp": doctored(no_offset),
        "extra": doctored(stray_key),
        "empty": doctored(empty_value),
    }
    report.check("output.section81_type_bites",
                 "decimal_not_string" in types["decimal"]
                 and "confidence_precision" in types["precision"]
                 and "decision_invalid" in types["decision"]
                 and "timestamp_offset" in types["timestamp"]
                 and "key_extra" in types["extra"],
                 f"浮点金额 -> {types['decimal'][:2]} / 四位置信度 -> {types['precision'][:2]}"
                 f" / decision=maybe -> {types['decision'][:2]} / 无时区 -> {types['timestamp'][:2]}"
                 f" / 多一个键 -> {types['extra'][:2]}")

    def clear_required(doc) -> str:
        path = next(item for item, field in doc["fields"].items()
                    if field.get("required") and field.get("value") is not None)
        doc["fields"][path]["value"] = None
        doc["fields"][path]["review_reasons"] = []
        return path

    def required_null(doc):
        clear_required(doc)

    def required_null_with_reason(doc):
        path = clear_required(doc)
        doc["fields"][path]["review_reasons"] = ["REQUIRED_EMPTY"]

    def unknown_reason(doc):
        doc["fields"][first_decimal(doc)]["review_reasons"] = ["NOT_A_REASON"]

    missing = doctored(required_null)
    with_reason = doctored(required_null_with_reason)
    unknown = doctored(unknown_reason)
    stale_config = dict(config)
    stale_config["validation"] = dict(config.get("validation") or {},
                                      review_reason_extensions=["GROUNDING_MISSING"])
    stale = [item["code"] for item in _docs.audit(out_dir, stale_config)["problems"]]
    report.check("output.section81_null_and_reason_bites",
                 "null_reason_missing" in missing
                 and "null_reason_missing" not in with_reason
                 and "reason_unknown" in unknown
                 and "reason_extension_stale" in stale,
                 f"必填空无原因 -> {missing[:2]} / 补 REQUIRED_EMPTY -> {with_reason[:2]}"
                 f" / 未知原因 -> {unknown[:2]} / 过期登记 -> {stale[:2]}")

    blank = doctored(blank_value)
    empties = []
    for path in _docs.document_files(out_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        empties += [f"{payload['document_id']}:{item}" for item, field in payload["fields"].items()
                    if isinstance(field.get("value"), str) and not field["value"].strip()]
    report.check("output.section81_empty_is_null",
                 not empties
                 and "empty_string_value" in types["empty"]
                 and "empty_string_value" in blank,
                 f"产物空串 {empties[:3]} / value='' -> {types['empty'][:2]}"
                 f" / value='   ' -> {blank[:2]}")


def _check_signal_contract(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §6.1：每格的信号键要齐、值要落在表里写的域内、表外信号要登记。"""
    from . import docaudit as _docs

    real = _docs.audit(out_dir, config)
    specs = _docs.signal_spec((Path(__file__).resolve().parent.parent
                               / "SCENARIO.md").read_text(encoding="utf-8"))
    report.check("output.section61_signals_hold",
                 len(specs) >= 7 and real.get("signals") == len(specs)
                 and real["documents"] >= 10 and not real["problems"],
                 f"信号={sorted(specs)} 问题={real['problems'][:3]}")

    base = json.loads(_docs.document_files(out_dir)[0].read_text(encoding="utf-8"))
    first = next(iter(base["fields"]))

    def doctored(mutate) -> List[str]:
        payload = copy.deepcopy(base)
        mutate(payload["fields"][first]["signals"])
        return [item["code"] for item in _docs.audit(out_dir, config, document=payload)["problems"]]

    dropped = doctored(lambda sig: sig.pop("table_score"))
    flipped = doctored(lambda sig: sig.__setitem__("format_valid", 0.7))
    stray = doctored(lambda sig: sig.__setitem__("mystery_signal", 1.0))
    report.check("output.section61_signal_drop_bites",
                 "signal_missing" in dropped and "signal_out_of_domain" in flipped,
                 f"删 table_score -> {dropped[:2]} / format_valid=0.7 -> {flipped[:2]}")
    report.check("output.section61_signal_extra_bites",
                 "signal_extra" in stray
                 and "signal_extra" not in [item["code"] for item in real["problems"]],
                 f"多一个未登记信号 -> {stray[:2]}")

    allowed = doctored(lambda sig: sig.__setitem__("extractor_agreement", None))
    forbidden = doctored(lambda sig: sig.__setitem__("format_valid", None))
    stale_config = dict(config)
    stale_config["validation"] = dict(config.get("validation") or {},
                                      signal_extensions=["ocr_conf"])
    stale = [item["code"] for item in _docs.audit(out_dir, stale_config)["problems"]]
    report.check("output.section61_null_and_extension_bites",
                 "signal_out_of_domain" not in allowed
                 and "signal_out_of_domain" in forbidden
                 and "signal_extension_stale" in stale,
                 f"extractor_agreement=None -> {allowed[:2]} / format_valid=None -> {forbidden[:2]}"
                 f" / 过期登记 -> {stale[:2]}")


def _check_cli_surface(report: Report, config: Dict) -> None:
    """SCENARIO §11：文档写下的五条命令就是交付接口，必须还能按写的跑。

    这条契约过去没有任何执行点：模块改名、`--input` 换成别的名字、带值的选项退化成
    开关，流水线、验收、单测全绿，用户照着文档跑第一条就失败。这里读 §11 取命令，
    导入模块并拦下它自己的 argparse（`main([])` 停在解析处，不执行命令正文），比对
    长选项存在性与「是否吃值」。三个反例证明门会红：多写一个不存在的选项、把开关
    写成带值、写了个不存在的模块。
    """
    from . import clispec as _cli

    text = (Path(__file__).resolve().parent.parent / "SCENARIO.md").read_text(encoding="utf-8")
    real = _cli.audit(text)
    report.check("cli.section11_commands_hold",
                 real["commands"] >= 5 and real["flags"] >= 10 and not real["problems"],
                 f"命令={real['commands']} 选项={real['flags']} "
                 f"模块={real['modules']} 问题={real['problems'][:3]}")

    dropped = _cli.problems([{"module": "docpipe.synth",
                              "flags": [{"name": "--out", "value": True},
                                        {"name": "--no-such-flag", "value": False}],
                              "command": "doctored"}])
    arity = _cli.problems([{"module": "docpipe.run",
                            "flags": [{"name": "--force", "value": True}],
                            "command": "doctored"}])
    ghost = _cli.problems([{"module": "docpipe.nosuch", "flags": [], "command": "doctored"}])
    report.check("cli.documented_flag_drop_bites",
                 any(item["code"] == "flag_missing" for item in dropped),
                 f"多写 --no-such-flag -> {dropped[:2]}")
    report.check("cli.documented_arity_bites",
                 any(item["code"] == "flag_arity" for item in arity),
                 f"--force 写成带值 -> {arity[:2]}")
    report.check("cli.documented_module_bites",
                 any(item["code"] == "module_missing" for item in ghost),
                 f"不存在的模块 -> {ghost[:2]}")


def _check_table_crosspath(report: Report, labels_dir: Path, config: Dict) -> None:
    """SCENARIO §12.2：文本层 PDF 的表格要跑第二条通路，差异必须留账。

    第一条通路是 pdfplumber（生产取数），第二条是 PyMuPDF 的 find_tables，只对表框、
    总行数、列数三件事。过去这条要求只写在文档里：pdfplumber 把一行切漏或把一列并起来，
    下游只看到一张「结构正常」的错表。这里按分片现场对账；分片没有样本清单（难例集等）
    就不适用，不凭空判红。三个反例证明门会红：掉一行、挪表框、抬容差。
    """
    from . import tablecross as _tc

    split = labels_dir.name
    manifest = ROOT / "samples" / "manifest.json"
    samples_dir = ROOT / "samples" / split
    if not manifest.exists() or not samples_dir.exists():
        return
    parsed_dir = resolve_path(config["paths"]["work"], config) / "parsed"
    min_iou = _tc.threshold_from_config(config)
    payload = _tc.measure_split(samples_dir, parsed_dir, manifest, split, min_iou)
    coverage = payload["coverage"]
    report.check("parse.second_table_path_covers_text_layer",
                 coverage["text_layer_docs"] > 0
                 and coverage["documents_scanned"] == coverage["text_layer_docs"]
                 and payload["tables"] >= 1
                 and not payload["undeclared_skips"]
                 and not payload["undeclared_problems"],
                 f"文字层文档={coverage['text_layer_docs']} 已扫={coverage['documents_scanned']} "
                 f"表={payload['tables']} 一致={payload['verdicts']['agree']} "
                 f"跳过={payload['skipped']} 未声明={payload['undeclared_skips']}")

    table = {"table_id": "p1-t1", "name": "transactions", "page": 1,
             "bbox": [35.0, 210.0, 557.0, 628.0], "headers": ["交易日期"],
             "columns": ["交易日期"], "rows": [{"cells": {"交易日期": {}}}]}
    same = {"bbox": [35.0, 210.0, 557.0, 628.0], "rows": 2, "cols": 1}
    dropped = _tc.compare_table(dict(table, rows=[]), same, min_iou)
    report.check("parse.second_table_path_bites_on_a_dropped_row",
                 dropped["problems"] == [_tc.PROBLEM_ROW_COUNT],
                 f"数据行 0 对第二通路 2 行 -> {dropped['problems']}")

    moved = _tc.compare_table(table, dict(same, bbox=[55.0, 230.0, 577.0, 648.0]), min_iou)
    report.check("parse.second_table_path_bites_on_a_moved_bbox",
                 moved["problems"] == [_tc.PROBLEM_BBOX],
                 f"IoU={moved['bbox_iou']} 门={min_iou} -> {moved['problems']}")

    hair = dict(same, bbox=[35.0, 210.0, 558.0, 628.0])
    loose = _tc.compare_table(table, hair, min_iou)
    tight = _tc.compare_table(table, hair, 0.9999)
    report.check("parse.second_table_path_threshold_follows_config",
                 not loose["problems"] and tight["problems"] == [_tc.PROBLEM_BBOX],
                 f"出厂 {min_iou}={loose['problems']} 抬到 0.9999={tight['problems']}")


def _check_parse_contract(report: Report, config: Dict) -> None:
    """SCENARIO §12.1「统一解析输出」的契约审计（docpipe.parseaudit）。

    §12.1 画了一份解析层接口，从来没有机械核对过：没人比对缓存里真有这些键吗，
    也没人比对「文档与实现差在哪、差的地方有没有人认领」。实测三处偏离（块→行、
    扁平格→按行分组、引擎对象→字符串）。这里先让出厂缓存过一遍（零违规才算干净），
    再逐条种破口证明每一码都会咬人——偏差双向可证伪是这一问的重点：
    `deviation_stale` 管「实现追上了文档」，`deviation_broken` 管「清单在说谎」。
    """
    import copy as _copy

    from . import parseaudit as _pa

    spec = _pa.section121_spec((ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
    parsed_dir = resolve_path(config["paths"]["work"], config) / "parsed"
    records = _pa.load_records(parsed_dir)
    payload = _pa.audit(records, spec, config)
    report.check("parse_contract.shipped_cache_is_clean",
                 payload["problems_total"] == 0 and not payload["details"]
                 and payload["contract"]["paths"] >= 25
                 and payload["contract"]["present"] >= 5
                 and payload["contract"]["deviations"] >= 10
                 and payload["totals"]["documents"] >= 5
                 and payload["totals"]["lines"] >= 100,
                 f"契约键={payload['contract']['paths']} 命中={payload['contract']['present']} "
                 f"偏差={payload['contract']['deviations']} 继承={payload['contract']['inherited']} "
                 f"文档={payload['totals']['documents']} 行={payload['totals']['lines']} "
                 f"问题={payload['problems_total']} {payload['problems']} "
                 f"版本={payload['versions']} 现行={payload['current_version']}")

    # 单项夹具：一份带表格的真实解析记录，逐条只种一个破口。
    fixture = next((record for record in records
                    if any(table.get("rows") for page in record.get("pages") or []
                           for table in page.get("tables") or [])), None)
    floor = _copy.deepcopy(config)
    floor["evaluation"]["parse_contract"] = {"min_documents": 1, "min_pages": 1,
                                             "min_lines": 1, "bbox_tolerance": 1.0}
    if fixture is None:
        report.check("parse_contract.bites_on_a_drifted_record", False,
                     "解析缓存里没有带表格的记录，这一问的夹具立不起来")
        report.check("parse_contract.invariants_bite", False, "同上：没有夹具")
        report.check("parse_contract.threshold_follows_config", False, "同上：没有夹具")
        report.check("parse_contract.is_diagnostic_only", False, "同上：没有夹具")
        return

    def codes(payload: Dict) -> List[str]:
        return sorted({item["code"] for item in payload["details"]})

    clean = codes(_pa.audit([fixture], spec, floor))
    blocks = _copy.deepcopy(fixture)
    blocks["pages"][0]["blocks"] = [{"text": "x"}]
    stale = codes(_pa.audit([blocks], spec, floor))
    emptied = _copy.deepcopy(fixture)
    for page in emptied["pages"]:
        page["lines"] = []
    broken = codes(_pa.audit([emptied], spec, floor))
    stripped = _copy.deepcopy(fixture)
    for page in stripped["pages"]:
        for table in page.get("tables") or []:
            table.pop("table_id", None)
    missing = codes(_pa.audit([stripped], spec, floor))
    report.check("parse_contract.bites_on_a_drifted_record",
                 clean == [] and stale == [_pa.DEVIATION_STALE]
                 and _pa.DEVIATION_BROKEN in broken
                 and missing == sorted([_pa.BAD_TABLE, _pa.MISSING_CONTRACT_KEY]),
                 f"干净={clean} 文档形状回来={stale} 替代形状消失={broken} "
                 f"缺键={missing}")

    counted = _copy.deepcopy(fixture)
    counted["page_count"] = 99
    out_of_page = _copy.deepcopy(fixture)
    out_of_page["pages"][0]["lines"][0]["bbox"] = [-50.0, -50.0, 10.0, 10.0]
    renumber = _copy.deepcopy(fixture)
    table = next(table for page in renumber["pages"]
                 for table in page.get("tables") or [] if table.get("rows"))
    table["rows"][0]["row_index"] = 42
    page_count = codes(_pa.audit([counted], spec, floor))
    bbox_out = codes(_pa.audit([out_of_page], spec, floor))
    row_index = codes(_pa.audit([renumber], spec, floor))
    report.check("parse_contract.invariants_bite",
                 page_count == [_pa.BAD_PAGE_COUNT]
                 and bbox_out == [_pa.LINE_BBOX_OUT]
                 and row_index == [_pa.BAD_ROW_INDEX],
                 f"页数={page_count} 坐标越页={bbox_out} 行号={row_index}")

    tight = _copy.deepcopy(config)
    tight["evaluation"]["parse_contract"] = {
        "min_documents": payload["totals"]["documents"] + 1,
        "min_pages": payload["totals"]["pages"] + 1,
        "min_lines": payload["totals"]["lines"] + 1,
        "bbox_tolerance": payload["options"]["bbox_tolerance"]}
    thin = _pa.audit(records, spec, tight)
    report.check("parse_contract.threshold_follows_config",
                 thin["problems"] == {_pa.THIN_CORPUS: 3} and thin["problems_total"] == 3,
                 f"三档下限各抬到语料之上 -> {thin['problems']}")

    decision_path = [ROOT / "docpipe" / "trust.py", ROOT / "docpipe" / "extract.py",
                     ROOT / "docpipe" / "run.py", ROOT / "docpipe" / "evaluate.py",
                     ROOT / "docpipe" / "serve.py"]
    wired = [str(path.relative_to(ROOT)) for path in decision_path
             if path.exists() and "parseaudit" in path.read_text(encoding="utf-8")]
    out_dir = resolve_path(config["paths"]["out"], config)
    before = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(out_dir.rglob("*")) if path.is_file()}         if out_dir.exists() else {}
    _pa.audit(records, spec, config)
    after = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(out_dir.rglob("*")) if path.is_file()}         if out_dir.exists() else {}
    report.check("parse_contract.is_diagnostic_only", not wired and before == after,
                 f"imported_by_decision_path={wired} out_changed={len(set(before) ^ set(after))}")


def _check_model_contract(report: Report, out_dir: Path, config: Dict) -> None:
    """SCENARIO §6.2「融合与校准」的产物侧对账（docpipe.modelaudit）。

    §6.2 后两条（字段有效样本 < 30 就把置信度压到 0.94 以下、禁止把原始 softmax /
    OCR 置信度当最终 `confidence`）约束的是**每一个交付出去的数**，但判决算完就把
    数写进产物，之后没人回看它是不是那个模型给的。换一个出口覆写 `confidence`、
    把交付样本数换成组样本数、拿未校准分数冒充概率，产物上都只留下一个 0～0.99 的
    小数，既有八问全绿。这里先让出厂产物过一遍（零违规才算干净），再逐条种破口证明
    每一码都会咬人：改一格 confidence 只报一格 `confidence_unexplained`，砍一维
    coef 只报 `coef_shape`，换两个特征名只报 `feature_order`，而 `Calibrator` 必须
    当场拒绝后两种——这就是本轮修掉的 `zip()` 静默截断。
    """
    from . import modelaudit as _ma
    from .trust import Calibrator

    source = _ma.artifact_path(config) if config["calibration"].get("artifact") else None
    artifact = _ma.load_artifact(source) if source is not None else None
    uncalibrated = source is None or not source.exists()
    documents = _ma.load_documents(out_dir)
    payload = _ma.audit(documents, artifact, config, allow_uncalibrated=uncalibrated)
    report.check("model.heuristic_scores_are_clean" if uncalibrated else "model.shipped_artifact_is_clean",
                 payload["problems_total"] == 0 and not payload["details"]
                 and payload["totals"]["documents"] >= 5
                 and payload["totals"]["fields"] >= 100
                 and (uncalibrated or (payload["coverage"].get("ratio", 0) > 0
                                      and payload["notes_total"] > 0
                                      and set(payload["notes"]) <= set(_ma.DECLARED_NOTES))),
                 f"文档={payload['totals']['documents']} 字段={payload['totals']['fields']} "
                 f"问题={payload['problems_total']} {payload['problems']} "
                 f"如实报={payload['notes']} 覆盖={payload['coverage']}")

    opts = dict(_ma.options(config))
    opts["min_delivered_fields"] = 1

    def codes(batch: List[Dict], artifact_=artifact) -> List[str]:
        return sorted({item["code"]
                       for item in _ma.audit(batch, artifact_, config, opts,
                                             allow_uncalibrated=uncalibrated)["details"]})

    fixture = next((document for document in documents if document.get("fields")), None)
    if fixture is None:
        report.check("model.confidence_is_recomputable", False, "结果目录缺少字段")
    else:
        tampered = copy.deepcopy(fixture)
        where = sorted(tampered["fields"])[0]
        original = float(tampered["fields"][where]["confidence"])
        tampered["fields"][where]["confidence"] = 0.0 if original > 0.5 else 0.99
        report.check("model.confidence_is_recomputable",
                     codes([fixture]) == []
                     and codes([tampered]) == [_ma.CONFIDENCE_UNEXPLAINED],
                     f"干净={codes([fixture])} 改一格={codes([tampered])}（{where}）")

    if not uncalibrated and (fixture is None or not artifact):
        for name in ("model.feature_contract_bites", "model.cap_and_sample_gap_are_reported"):
            report.check(name, False, "结果目录或校准产物不可用")
    elif not uncalibrated:
        short = copy.deepcopy(artifact)
        short["coef"] = short["coef"][:-1]
        swapped = copy.deepcopy(artifact)
        swapped["features"] = list(swapped["features"])
        swapped["features"][0], swapped["features"][1] = \
            swapped["features"][1], swapped["features"][0]
        narrow = copy.deepcopy(artifact)
        narrow["features"] = narrow["features"][:-1]
        shape = {name: sorted({item["code"] for item in
                               _ma.artifact_problems(broken, config, opts)["details"]})
                 for name, broken in (("coef", short), ("features", swapped),
                                      ("narrow", narrow))}
        refusals = []
        for broken in (short, swapped):
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "calibration.json"
                target.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
                probe = copy.deepcopy(config)
                probe["calibration"]["artifact"] = str(target)
                try:
                    Calibrator(probe)
                except ValueError:
                    refusals.append(True)
                else:
                    refusals.append(False)
        report.check("model.feature_contract_bites",
                     shape["coef"] == [_ma.COEF_SHAPE]
                     and shape["features"] == [_ma.FEATURE_ORDER]
                     and set(shape["narrow"]) == {_ma.FEATURE_ORDER, _ma.COEF_SHAPE}
                     and refusals == [True, True],
                     f"砍 coef={shape['coef']} 换名={shape['features']} "
                     f"砍特征={shape['narrow']} Calibrator 拒收={refusals}")

        field_mode = copy.deepcopy(artifact)
        field_mode["sample_grouping"] = "field"
        field_codes = sorted({item["code"] for document in documents
                              for item in _ma.audit([document], field_mode, config,
                                                    opts)["details"]})
        report.check("model.cap_and_sample_gap_are_reported",
                     payload["notes"].get(_ma.SAMPLE_COUNTING_GAP) == 1
                     and payload["totals"]["low_sample_on_group"] > 0
                     and payload["totals"]["low_sample_paths"] > 0
                     and _ma.BASIS_MISMATCH in field_codes,
                     f"如实报={payload['notes']} 组样本顶替="
                     f"{payload['totals']['low_sample_on_group']} 格 / "
                     f"{payload['totals']['low_sample_paths']} 条路径；"
                     f"按 field 口径重算={field_codes}")

    decision_path = [ROOT / "docpipe" / name for name in
                     ("trust.py", "extract.py", "run.py", "evaluate.py", "serve.py")]
    # 只认真的 import：`trust.py` 的注释里提到过 `docpipe.modelaudit` 这个名字，
    # 那是让人去找这个模块的指路牌，不是判决路径在调用它。
    markers = ("from . import modelaudit", "from .modelaudit import",
               "import modelaudit")
    wired = [str(path.relative_to(ROOT)) for path in decision_path
             if path.exists() and any(marker in path.read_text(encoding="utf-8")
                                      for marker in markers)]
    target_dir = Path(out_dir)
    before = {str(path.relative_to(target_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(target_dir.rglob("*")) if path.is_file()} \
        if target_dir.exists() else {}
    _ma.audit(documents, artifact, config, allow_uncalibrated=uncalibrated)
    after = {str(path.relative_to(target_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(target_dir.rglob("*")) if path.is_file()} \
        if target_dir.exists() else {}
    report.check("model.is_diagnostic_only", not wired and before == after,
                 f"imported_by_decision_path={wired} "
                 f"out_changed={len(set(before) ^ set(after))}")


def _check_threshold_contract(report: Report, config: Dict) -> None:
    """SCENARIO §6.3 风险档阈值契约的两头对账（docpipe.riskcontract）。

    §6.3 的八档阈值与全局下限是 `trust.decide_field` 唯一的输入，可文档那一头从来
    没有人读过：把 `monetary` 从 0.97 改成 0.90、把某个字段从 `party_name` 挪到
    `free_text`，产物里只留下一个「符合门槛」的 accept，第十二问（`deliveryaudit.gates`）
    拿的还是同一份配置照样解释得通。这里让出厂配置与字段 schema 逐条过一遍（零违规
    才算干净），再逐条种破口证明每一码都会咬人。本轮据此修掉的一处真偏航是
    `bl.registration_authority`（登记机关）按 `optional_low_risk` 0.93 放行，
    而契约把它算进 `free_text` 0.95。
    """
    from . import riskcontract as _rc

    names = ("thresholds.shipped_config_matches_section63",
             "thresholds.tier_values_are_checked", "thresholds.demoted_field_bites",
             "thresholds.tier_set_is_checked", "thresholds.floor_is_checked",
             "thresholds.date_universality_bites", "thresholds.stale_exception_bites",
             "thresholds.is_diagnostic_only")
    spec = _rc.section63_spec()
    if not spec.get("tiers"):
        for name in names:
            report.check(name, False, "SCENARIO.md §6.3 读不出来，这一组判据立不起来")
        return

    payload = _rc.audit(config, spec)
    report.check(names[0],
                 payload["problems_total"] == 0 and not payload["details"]
                 and payload["tiers"] >= payload["min_tiers"]
                 and payload["floor"] is not None
                 and float(payload["floor"]) == float(payload["config_floor"])
                 and payload["members"]["resolved"] >= 20
                 and payload["checked_fields"] >= 40,
                 f"风险档={payload['tiers']} 全局下限={payload['floor']}/"
                 f"{payload['config_floor']} 契约字段={payload['members']['declared']} "
                 f"解析到={payload['members']['resolved']} 受检字段={payload['checked_fields']} "
                 f"问题={payload['problems_total']} {payload['problems']}")

    def run(probe: Dict, spec_=None) -> Dict:
        return _rc.audit(probe, spec if spec_ is None else spec_)

    def codes(probe: Dict, spec_=None) -> List[str]:
        return sorted({item["code"] for item in run(probe, spec_)["details"]})

    def where(probe: Dict, spec_=None) -> List[str]:
        return [item["where"] for item in run(probe, spec_)["details"]]

    # 阈值走偏两个方向都算：抬到 0.98 只会咬 threshold_mismatch；压到 0.90 会顺带
    # 让 monetary 的字段全部 member_weaker（同一处缺陷的第二个骨牌，单测里压 date
    # 档证明「压」也会咬）。
    drifted = copy.deepcopy(config)
    drifted["thresholds"]["risk_tiers"]["monetary"] = 0.98
    report.check(names[1],
                 codes(drifted) == [_rc.THRESHOLD_MISMATCH]
                 and where(drifted) == ["thresholds.risk_tiers.monetary"],
                 f"monetary 0.97→0.98：{where(drifted)} {codes(drifted)}")

    demoted = copy.deepcopy(config)
    target = next(item for item in demoted["doc_types"]["BUSINESS_LICENSE"]["fields"]
                  if item["path"] == "bl.company_name")
    target["risk"] = "free_text"
    report.check(names[2],
                 codes(demoted) == [_rc.MEMBER_WEAKER]
                 and where(demoted) == ["BUSINESS_LICENSE.bl.company_name"],
                 f"企业名称 party_name→free_text：{where(demoted)} {codes(demoted)}")

    # 少一档在契约那侧补一行、多一档在契约那侧删一行：直接在配置侧删档会顺带让
    # 该档字段报 unknown_risk_member，那是同一处缺陷的第二个骨牌，不是第二条判据。
    missing = copy.deepcopy(spec)
    missing["tiers"]["bonus_risk"] = {"threshold": 0.90, "members": "币种"}
    unlisted = copy.deepcopy(config)
    unlisted["thresholds"]["risk_tiers"]["bonus_risk"] = 0.90
    report.check(names[3],
                 codes(config, missing) == [_rc.TIER_MISSING]
                 and codes(unlisted) == [_rc.TIER_UNLISTED],
                 f"契约多一档={codes(config, missing)} 配置多一档={codes(unlisted)}")

    floor = copy.deepcopy(config)
    floor["thresholds"]["global_min_confidence"] = 0.85
    report.check(names[4], codes(floor) == [_rc.FLOOR_MISMATCH],
                 f"全局下限 0.90→0.85：{where(floor)} {codes(floor)}")

    shifted = copy.deepcopy(config)
    moved = next(item for item in shifted["doc_types"]["BUSINESS_LICENSE"]["fields"]
                 if item["path"] == "bl.establishment_date")
    moved["risk"] = "free_text"
    report.check(names[5],
                 codes(shifted) == [_rc.DATE_TIER_NOT_UNIVERSAL]
                 and where(shifted) == ["BUSINESS_LICENSE.bl.establishment_date"],
                 f"成立日期 date→free_text：{where(shifted)} {codes(shifted)}")

    original_exceptions = _rc.UNRESOLVED_OK
    stale_map = {tier: dict(items) for tier, items in original_exceptions.items()}
    stale_map.setdefault("identity", {})["统一社会信用代码"] = "临时登记：证明过期登记会咬人"
    _rc.UNRESOLVED_OK = stale_map
    try:
        stale = codes(config)
    finally:
        _rc.UNRESOLVED_OK = original_exceptions
    report.check(names[6], stale == [_rc.MEMBER_STALE_EXCEPTION],
                 f"把能解析的名字登记成「没有对应字段」：{stale}")

    decision_path = [ROOT / "docpipe" / name for name in
                     ("trust.py", "extract.py", "run.py", "evaluate.py", "serve.py",
                      "export.py", "report.py", "deliveryaudit.py")]
    markers = ("from . import riskcontract", "from .riskcontract import",
               "import riskcontract")
    wired = [str(path.relative_to(ROOT)) for path in decision_path
             if path.exists() and any(marker in path.read_text(encoding="utf-8")
                                      for marker in markers)]
    report.check(names[7], not wired, f"imported_by_decision_path={wired}")


def _check_gate_contract(report: Report, config: Dict) -> None:
    """SCENARIO §1.3 成功定义 / §10.3 阶段门槛与 `acceptance.*` 的逐值对账（docpipe.gatecontract）。

    §1.3 的五个数此前只活在文档里：`acceptance` 与 `evaluate` 读的是配置那份，把
    `min_auto_accept_rate` 改成 0.50 照样 PASS、报告照样宣称达标。这里把两节现读出来，
    先要求它们彼此一致，再要求配置逐值相等，最后要求每个门槛真有人读。
    """
    from . import gatecontract as _gc

    names = ("gates.shipped_config_matches_success_definition",
             "gates.a_relaxed_auto_accept_gate_is_named",
             "gates.a_missing_gate_key_is_named",
             "gates.doc_sections_must_agree",
             "gates.a_missing_stage_row_is_named",
             "gates.a_thin_stage_section_is_named",
             "gates.a_gate_nobody_reads_is_named",
             "gates.is_diagnostic_only")
    success = _gc.success_spec()
    stages = _gc.stage_spec()
    if not success.get("numbers") or not stages.get("stages"):
        for name in names:
            report.check(name, False, "SCENARIO §1.3 / §10.3 读不出来，这一组判据立不起来")
        return

    payload = _gc.audit(config, success, stages)
    report.check(names[0],
                 payload["problems_total"] == 0 and not payload["details"]
                 and payload["metrics"] == 5 and payload["numbers_read"] == 5
                 and payload["gates_compared"] == 5
                 and payload["stages"] >= payload["min_stages"],
                 f"门槛={payload['metrics']} §1.3 读出={payload['numbers_read']} "
                 f"阶段={payload['stages']} 对账={payload['gates_compared']} "
                 f"问题={payload['problems_total']} {payload['problems']}")

    def codes(probe=None, spec=None, stage=None) -> List[str]:
        result = _gc.audit(probe or config,
                           success if spec is None else spec,
                           stages if stage is None else stage)
        return sorted({item["code"] for item in result["details"]})

    def where(probe=None, spec=None, stage=None) -> List[str]:
        result = _gc.audit(probe or config,
                           success if spec is None else spec,
                           stages if stage is None else stage)
        return [item["where"] for item in result["details"]]

    relaxed = copy.deepcopy(config)
    relaxed["acceptance"]["min_auto_accept_rate"] = 0.50
    report.check(names[1],
                 codes(relaxed) == [_gc.GATE_MISMATCH]
                 and where(relaxed) == ["acceptance.min_auto_accept_rate"],
                 f"自动接受率 0.60→0.50：{where(relaxed)} {codes(relaxed)}")

    dropped = copy.deepcopy(config)
    dropped["acceptance"].pop("min_grounding_hit_rate", None)
    report.check(names[2],
                 codes(dropped) == [_gc.GATE_MISSING]
                 and where(dropped) == ["acceptance.min_grounding_hit_rate"],
                 f"删掉溯源命中率门槛：{where(dropped)} {codes(dropped)}")

    # 两节自己打架时先报 doc_conflict；同一处缺陷会让配置也跟着「对不上」其中一节，
    # 所以 gate_mismatch 同时出现是第二个骨牌，不是第二条判据。
    drifted = copy.deepcopy(success)
    drifted["numbers"]["min_auto_accept_rate"] = 0.61
    report.check(names[3],
                 codes(spec=drifted) == [_gc.DOC_CONFLICT, _gc.GATE_MISMATCH]
                 and where(spec=drifted)[0] == "§10.3 阶段 4 vs §1.3",
                 f"§1.3 改成 61%：{where(spec=drifted)} {codes(spec=drifted)}")

    thinned_stages = copy.deepcopy(stages)
    thinned_stages["stages"].pop(3)
    relaxed_min = copy.deepcopy(config)
    relaxed_min["evaluation"]["gate_contract"]["min_stages"] = 4
    report.check(names[4],
                 codes(relaxed_min, stage=thinned_stages) == [_gc.STAGE_MISSING]
                 and where(relaxed_min, stage=thinned_stages) == ["SCENARIO.md §10.3 阶段 3"],
                 f"删掉阶段 3 那行：{where(relaxed_min, stage=thinned_stages)} "
                 f"{codes(relaxed_min, stage=thinned_stages)}")

    strict = copy.deepcopy(config)
    strict["evaluation"]["gate_contract"]["min_stages"] = 6
    report.check(names[5],
                 codes(strict) == [_gc.STAGE_THIN]
                 and where(strict) == ["SCENARIO.md §10.3"],
                 f"min_stages 抬到 6：{where(strict)} {codes(strict)}")

    original_readers = _gc.READERS
    _gc.READERS = ()
    try:
        unread = codes()
    finally:
        _gc.READERS = original_readers
    report.check(names[6],
                 unread == [_gc.GATE_UNREAD] and where() is not None
                 and len([item for item in _gc.audit(config, success, stages)["details"]
                          if item["code"] == _gc.GATE_UNREAD]) == 0,
                 f"把读者表清空：{unread}")

    decision_path = [ROOT / "docpipe" / name for name in
                     ("trust.py", "extract.py", "run.py", "serve.py", "export.py", "report.py")]
    markers = ("from . import gatecontract", "from .gatecontract import",
               "import gatecontract")
    wired = [str(path.relative_to(ROOT)) for path in decision_path
             if path.exists() and any(marker in path.read_text(encoding="utf-8")
                                      for marker in markers)]
    report.check(names[7], not wired, f"imported_by_decision_path={wired}")


def _check_doc_type_contract(report: Report, config: Dict) -> None:
    """新增一类文档「要动几处」的接入契约（docpipe.doctypeaudit）。

    两处静默兜底逼出这条检查：`synth.generate_document` 过去用 `else:` 把任何未知类型
    合成成征信报告（标注写着新类型、页面却是征信），而 `extractor: table_cell` 声明
    从来没有代码读——分派靠路径里有没有 `[i]`，声明写错也照样取数。这里对出厂配置
    逐类型对账，再用一份「什么毛病都占一点」的草稿类型证明每一码都会咬人。
    """
    import copy as _copy

    from . import doctypeaudit as _dt

    manifest_path = ROOT / "samples" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.exists() else None
    payload = _dt.audit(config, manifest)
    report.check("doctype.shipped_contract_is_clean",
                 payload["problems_total"] == 0 and payload["type_count"] >= 4
                 and not payload["details"]
                 and payload["cross_paths"]["checked"] >= 1,
                 f"类型={payload['type_count']} 字段={payload['field_count']} "
                 f"问题={payload['problems_total']} {payload['problems']} "
                 f"跨文档路径={payload['cross_paths']['checked']}")

    thin = _copy.deepcopy(config)
    thin["doc_types"]["SYNTH_THIN"] = {
        "label": "草稿票据",
        "keywords": {"草稿": 1},
        "tables": {},
        "fields": [
            {"path": "st.only", "label": "唯一", "type": "string", "required": True,
             "risk": "not_a_tier", "extractor": "tabel_cell", "rules": []},
        ],
    }
    thin["evaluation"]["doc_type_contract"]["require_synth"].append("SYNTH_THIN")
    thin["cross_document"]["rules"]["V-X-01"]["paths"].append("st.ghost")
    payload_thin = _dt.audit(thin, manifest)
    thin_codes = {problem["code"] for entry in payload_thin["types"]
                  for problem in entry["problems"]}
    thin_codes |= {problem["code"] for problem in payload_thin["details"]}
    wanted = {_dt.UNDETECTABLE, _dt.THIN_REQUIRED, _dt.NO_BUILDER, _dt.NO_SAMPLES,
              _dt.UNKNOWN_RISK, _dt.EXTRACTOR_UNKNOWN, _dt.DEAD_CROSS_PATH}
    report.check("doctype.bites_on_a_thin_declaration", wanted <= thin_codes,
                 f"缺={sorted(wanted - thin_codes)} 命中={sorted(thin_codes)}")

    mismatch = _copy.deepcopy(config)
    mismatch["doc_types"]["BANK_STATEMENT"]["fields"].append(
        {"path": "bs.designation", "label": "摘要", "type": "string", "required": False,
         "risk": "free_text", "extractor": "table_cell", "rules": []})
    payload_mismatch = _dt.audit(mismatch, manifest)
    mismatch_codes = [problem for entry in payload_mismatch["types"]
                      if entry["doc_type"] == "BANK_STATEMENT"
                      for problem in entry["problems"]
                      if problem["code"] == _dt.EXTRACTOR_PATH_MISMATCH]
    report.check("doctype.extractor_declaration_must_match_path_shape",
                 len(mismatch_codes) == 4,
                 f"table_cell 不带 [i]、缺 table/column/array -> "
                 f"{[item['detail'] for item in mismatch_codes]}")

    strict = _copy.deepcopy(config)
    strict["evaluation"]["doc_type_contract"]["min_required_fields"] = 99
    loose = _copy.deepcopy(config)
    loose["evaluation"]["doc_type_contract"]["min_required_fields"] = 1
    strict_count = len([entry for entry in _dt.audit(strict, manifest)["types"]
                        if any(problem["code"] == _dt.THIN_REQUIRED
                               for problem in entry["problems"])])
    loose_count = len([entry for entry in _dt.audit(loose, manifest)["types"]
                       if any(problem["code"] == _dt.THIN_REQUIRED
                              for problem in entry["problems"])])
    report.check("doctype.threshold_follows_config",
                 strict_count == payload["type_count"] and loose_count == 0,
                 f"99 -> {strict_count}/{payload['type_count']} 1 -> {loose_count}")

    mystery = {"documents": [{"doc_type": "MYSTERY"}]}
    payload_mystery = _dt.audit(config, mystery)
    report.check("doctype.unknown_manifest_type_is_named",
                 _dt.UNKNOWN_MANIFEST_TYPE in payload_mystery["problems"],
                 f"problems={payload_mystery['problems']}")


def _check_document_refusal(report: Report, config: Dict) -> None:
    """SCENARIO §6.4 的 UNKNOWN_DOC_TYPE 现在挂在文档上（trust.document_refusal）。

    它过去只挂在字段上，而类型认不出时 schema 选不出来、一个字段都不会被抽出，于是
    「强制复核」退化成零字段零原因的空文档；单文件网页版还会把它整份藏掉。这一问只问
    三件事：拒答挂得上文档、认得出的类型不被误挂、两个出口都读得到。
    """
    import copy as _copy
    import tempfile as _tempfile

    from . import run as _run
    from .export import load_final_documents
    from .report import build_report
    from .trust import assess

    parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
              "pages": [{"page": 1, "width": 200, "height": 200}]}
    unknown = {"document_id": "ZZ99", "doc_type": "UNKNOWN", "doc_type_confidence": 0.31,
               "fields": {}, "source": "ZZ99.pdf"}
    known = {"document_id": "ZZ98", "doc_type": "BUSINESS_LICENSE",
             "doc_type_confidence": 0.9, "fields": {}, "source": "ZZ98.pdf"}
    thin = _copy.deepcopy(config)
    thin["calibration"]["artifact"] = None
    [refused] = assess([_copy.deepcopy(unknown)], {"ZZ99": parsed}, thin)
    [silent] = assess([_copy.deepcopy(known)], {"ZZ98": parsed}, thin)
    expected = (config.get("validation") or {}).get("review_actions", {}).get("UNKNOWN_DOC_TYPE")
    refusal = refused.get("document_refusal") or {}
    report.check("document_refusal.unknown_document_carries_the_code",
                 refusal.get("code") == "UNKNOWN_DOC_TYPE"
                 and refusal.get("field_count") == 0
                 and refusal.get("action") == expected,
                 f"code={refusal.get('code')} fields={refusal.get('field_count')} "
                 f"action_ok={refusal.get('action') == expected}")

    payload = _run._document_payload(refused, parsed, Path("ZZ99.pdf"), None, thin)
    report.check("document_refusal.known_type_stays_silent",
                 "document_refusal" not in silent and "document_refusal" in payload,
                 f"known_has_key={('document_refusal' in silent)} "
                 f"payload_has_key={('document_refusal' in payload)}")

    with _tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "ZZ99").mkdir()
        (root / "ZZ99" / "ZZ99.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps({
            "run": {"pipeline_version": "fixture",
                    "started_at": "2026-01-01T00:00:00+08:00"},
            "documents": [{"document_id": "ZZ99", "doc_type": "UNKNOWN",
                           "document_refusal": payload["document_refusal"],
                           "source": "ZZ99.pdf", "page_count": 1,
                           "path": "ZZ99/ZZ99.json", "field_count": 0,
                           "accept_count": 0, "review_count": 0}],
        }, ensure_ascii=False), encoding="utf-8")
        _, final_summary, _ = load_final_documents(root, config)
        html = Path(build_report(root, config=config)["path"]).read_text(encoding="utf-8")
    row = (final_summary.get("documents") or [{}])[0]
    report_source = (ROOT / "docpipe" / "report.py").read_text(encoding="utf-8")
    ui_source = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
    report.check("document_refusal.reaches_both_exits",
                 (row.get("document_refusal") or {}).get("code") == "UNKNOWN_DOC_TYPE"
                 and final_summary["overall"].get("refused_document_count") == 1
                 and "doc.refusal ||" in report_source
                 and '"refusal"' in html and "文档类型未识别" in html
                 and "state.doc.document_refusal" in ui_source,
                 f"final_row={bool(row.get('document_refusal'))} "
                 f"refused={final_summary['overall'].get('refused_document_count')} "
                 f"report_follow_through={('doc.refusal ||' in report_source)} "
                 f"ui_wired={('state.doc.document_refusal' in ui_source)}")


def count_fix(payload: Dict) -> None:
    """幽灵文档补总数的零头，让 layout 检查单独报出来，不被 count_mismatch 抢走。"""
    rows = payload.get("documents") or []
    payload["overall"]["field_count"] = sum(int(row.get("field_count") or 0) for row in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local acceptance checks")
    parser.add_argument("--out", default=str(ROOT / "out" / "holdout"))
    parser.add_argument("--labels", default=str(ROOT / "labels" / "holdout"))
    args = parser.parse_args(argv)
    result = run_acceptance(Path(args.out), Path(args.labels), load_config())
    for item in result["checks"]:
        print(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']} {item['detail']}")
    print("ACCEPTANCE:", "PASS" if result["ok"] else "FAIL")
    print("report:", result["report_path"])
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
