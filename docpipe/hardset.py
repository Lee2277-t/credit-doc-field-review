"""P0-1 hard set: controlled degradations, attributed per failure mode.

Six degradation families x five documents, generated on demand (never part of
samples/dev or samples/holdout). Two are vector-only (broken table rules, two
column layout) and keep a text layer; the rest are rasterised with the same
bbox-transform math as synth.scan_pdf.

The evaluation runs the real pipeline, then reports accept / error / grounding
rates per degradation, so a failure names a mode instead of one blended number.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import random
import re
import shutil
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .config import ROOT, load_config, resolve_path
from .evaluate import (_decision_stats, claim_targets, evaluate_predictions,
                    wilson_interval)
from .run import run
from .synth import (
    A4_WIDTH,
    GT,
    _new_page,
    _transform_bbox,
    add_footer,
    build_bank_statement,
    build_business_license,
    build_credit_report,
    build_income_statement,
    decimal_str,
    draw_kv,
    draw_rect,
    draw_table,
    draw_text,
    fit_size,
    format_money_for_print,
    make_bank_transactions,
    make_company,
    make_credit_data,
    make_income_data,
    set_gt,
)
from .textutil import now_iso

HARD_SET_SEED = 20261025
HARD_DIR = ROOT / "samples" / "hard"
HARD_LABELS = ROOT / "labels" / "hard"
HARD_OUT = ROOT / "out" / "hard"
HARD_WORK = ROOT / "work" / "hard"

DEGRADATIONS: Dict[str, Dict] = {
    "copier_noise": {"code": "NC", "label": "复印噪点", "mode": "raster"},
    "jpeg_heavy": {"code": "JP", "label": "JPEG 重度压缩", "mode": "raster"},
    "broken_lines": {"code": "LB", "label": "表格线断裂", "mode": "vector"},
    "crease": {"code": "FC", "label": "折痕阴影", "mode": "raster"},
    "low_resolution": {"code": "LR", "label": "低分辨率", "mode": "raster"},
    "two_column": {"code": "TC", "label": "双栏排版", "mode": "vector"},
}

PLAN: Dict[str, List[str]] = {
    "copier_noise": ["BUSINESS_LICENSE", "BANK_STATEMENT", "INCOME_STATEMENT",
                     "CREDIT_REPORT_ENTERPRISE", "BUSINESS_LICENSE"],
    "jpeg_heavy": ["BUSINESS_LICENSE", "BANK_STATEMENT", "INCOME_STATEMENT",
                   "CREDIT_REPORT_ENTERPRISE", "INCOME_STATEMENT"],
    "broken_lines": ["BANK_STATEMENT", "INCOME_STATEMENT", "BUSINESS_LICENSE",
                     "BANK_STATEMENT", "CREDIT_REPORT_ENTERPRISE"],
    "crease": ["BANK_STATEMENT", "CREDIT_REPORT_ENTERPRISE", "BUSINESS_LICENSE",
               "INCOME_STATEMENT", "BANK_STATEMENT"],
    "low_resolution": ["BUSINESS_LICENSE", "BANK_STATEMENT", "INCOME_STATEMENT",
                       "CREDIT_REPORT_ENTERPRISE", "BANK_STATEMENT"],
    "two_column": ["CREDIT_REPORT_ENTERPRISE", "BUSINESS_LICENSE",
                   "CREDIT_REPORT_ENTERPRISE", "BUSINESS_LICENSE",
                   "CREDIT_REPORT_ENTERPRISE"],
}

RASTER_SPECS = {
    "copier_noise": {"dpi": 300, "quality": 70, "angle": 0.35},
    "jpeg_heavy": {"dpi": 170, "quality": 22, "angle": 0.40},
    "crease": {"dpi": 300, "quality": 75, "angle": 0.50},
    "low_resolution": {"dpi": 120, "quality": 55, "angle": 0.30},
}

_DOC_ID_RE = re.compile(r"^[A-Z]{2}\d{2}$")


# ---------------------------------------------------------------------------
# image degradations
# ---------------------------------------------------------------------------

def _copier_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("L").convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    image = ImageEnhance.Contrast(image).enhance(0.86)
    array = np.asarray(image).astype(np.float32)
    noise = np.random.default_rng(rng.randrange(2 ** 31)).normal(
        0.0, rng.uniform(7.0, 11.0), array.shape[:2])
    array = np.clip(array + noise[:, :, None], 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(18, 28)):
        x, y = rng.randrange(image.width), rng.randrange(image.height)
        radius = rng.randint(1, 3)
        draw.ellipse([x, y, x + radius, y + radius],
                     fill=rng.choice([0, 15, 235, 245]))
    return image


def _jpeg_heavy(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.5)))
    return ImageEnhance.Contrast(image).enhance(0.94)


def _crease(image: Image.Image, rng: random.Random) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    width = array.shape[1]
    xs = np.arange(width, dtype=np.float32)
    center = rng.uniform(0.32, 0.68) * width
    band = rng.uniform(0.055, 0.095) * width
    shadow = 1.0 - 0.30 * np.exp(-0.5 * ((xs - center) / band) ** 2)
    array = array * shadow[None, :, None]
    highlight = 34.0 * np.exp(-0.5 * ((xs - center - 2.0) / 1.8) ** 2)
    array = array + highlight[None, :, None]
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    return image.filter(ImageFilter.GaussianBlur(radius=0.45))


def _low_resolution(image: Image.Image, rng: random.Random) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=0.35))


IMAGE_DEGRADATIONS: Dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "copier_noise": _copier_noise,
    "jpeg_heavy": _jpeg_heavy,
    "crease": _crease,
    "low_resolution": _low_resolution,
}


def _break_table_rules(doc, rng: random.Random) -> int:
    """Erase random horizontal stretches of every drawn rule and cell border."""
    erased = 0
    for page in doc:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            if rect.height <= 2.5 and rect.width >= 60:
                edges = [(rect.x0, rect.y0, rect.x1)]
            elif rect.height > 8 and rect.width >= 30:
                edges = [(rect.x0, rect.y0, rect.x1), (rect.x0, rect.y1, rect.x1)]
            else:
                continue
            for x0, y, x1 in edges:
                if rng.random() > 0.55:
                    continue
                span = x1 - x0
                cut = rng.uniform(0.18, 0.45) * span
                start = x0 + rng.uniform(0, max(0.0, span - cut))
                page.draw_line(fitz.Point(start, y), fitz.Point(start + cut, y),
                               color=(1, 1, 1), width=1.6)
                erased += 1
    return erased


# ---------------------------------------------------------------------------
# two-column vector builders
# ---------------------------------------------------------------------------

def _column_kv(page, gt: GT, x_label: float, x_value: float, y: float,
               label: str, value: str, path: Optional[str], max_width: float,
               page_no: int = 1, size: float = 9.5) -> List[float]:
    value_size = fit_size(value, max_width, size, 6.5)
    bbox = draw_kv(page, x_label, x_value, y, label, value, size=size,
                   value_size=value_size)
    if path:
        set_gt(gt, path, value, page_no, bbox)
    return bbox


def build_two_column_business_license(doc, company: Dict) -> GT:
    gt: GT = {}
    page = _new_page(doc)
    draw_rect(page, 35, 35, 560, 795, width=1.2)
    draw_text(page, A4_WIDTH / 2, 82, "营业执照（副本）", size=20, align="center", bold=True)
    draw_text(page, 60, 120, "统一社会信用代码：", size=10)
    set_gt(gt, "bl.unified_social_credit_code", company["uscc"], 1,
           draw_text(page, 172, 120, company["uscc"], size=11))

    left = [
        ("名称", company["company_name"], "bl.company_name"),
        ("类型", company["entity_type"], "bl.entity_type"),
        ("法定代表人", company["legal_rep"], "bl.legal_representative"),
        ("注册资本", company["capital_display"], "bl.registered_capital"),
        ("成立日期", company["establishment"].isoformat(), "bl.establishment_date"),
    ]
    right = [
        ("营业期限", f"{company['term_start'].isoformat()} 至 {company['term_end'].isoformat()}", None),
        ("住所", company["address"], "bl.address"),
        ("登记机关", company["authority"], "bl.registration_authority"),
        ("发照日期", company["issue_date"].isoformat(), "bl.issue_date"),
    ]
    top, step = 168, 52
    for index, (label, value, path) in enumerate(left):
        _column_kv(page, gt, 55, 140, top + index * step, label, value, path, 150)
    for index, (label, value, path) in enumerate(right):
        bbox = _column_kv(page, gt, 315, 405, top + index * step, label, value, path, 138)
        if label == "营业期限":
            set_gt(gt, "bl.business_term_start", company["term_start"].isoformat(), 1, bbox)
            set_gt(gt, "bl.business_term_end", company["term_end"].isoformat(), 1, bbox)
    if "bl.registered_capital" in gt:
        gt["bl.registered_capital"]["value"] = decimal_str(company["capital"])

    draw_text(page, 55, 432, "经营范围：", size=9.5)
    set_gt(gt, "bl.business_scope", company["scope"], 1,
           draw_text(page, 140, 432, company["scope"], size=9))
    page.draw_circle(fitz.Point(480, 690), 48, color=(0.8, 0.1, 0.1), width=2.0)
    draw_text(page, 480, 686, "示例市", size=9, color=(0.8, 0.1, 0.1), align="center")
    draw_text(page, 480, 700, "登记专用章", size=9, color=(0.8, 0.1, 0.1), align="center")
    add_footer(page, 1, 1)
    return gt


def build_two_column_credit_report(doc, company: Dict, data: Dict) -> GT:
    gt: GT = {}
    page1 = _new_page(doc)
    draw_text(page1, A4_WIDTH / 2, 58, "企业信用报告（简版）", size=18, align="center", bold=True)
    draw_text(page1, 48, 100, "一、企业基本信息", size=11.5)
    draw_text(page1, 312, 100, "二、信贷信息概要", size=11.5)

    report_period = (f"{data['report_period_start'].isoformat()} 至 "
                     f"{data['report_period_end'].isoformat()}")
    left = [
        ("企业名称", company["company_name"], "cr.company_name"),
        ("统一社会信用代码", company["uscc"], "cr.unified_social_credit_code"),
        ("法定代表人", company["legal_rep"], None),
        ("注册地址", company["address"], None),
        ("报告期间", report_period, "cr.report_period_start"),
    ]
    right = [
        ("报告编号", data["report_number"], "cr.report_number"),
        ("查询日期", data["query_date"].isoformat(), "cr.query_date"),
        ("未结清贷款笔数", str(data["loan_count"]), "cr.loan_account_count"),
        ("未结清贷款余额", format_money_for_print(data["outstanding"]), "cr.outstanding_loan_balance"),
        ("近24个月逾期账户数", str(data["overdue_accounts"]), "cr.overdue_account_count_24m"),
        ("最长逾期月数", str(data["max_overdue_months"]), "cr.max_overdue_months"),
        ("近3个月查询次数", str(data["query_count_3m"]), "cr.query_count_3m"),
    ]
    top, step = 130, 38
    for index, (label, value, path) in enumerate(left):
        _column_kv(page1, gt, 48, 150, top + index * step, label, value, path, 135, size=9)
    for index, (label, value, path) in enumerate(right):
        _column_kv(page1, gt, 312, 420, top + index * step, label, value, path, 130, size=9)

    draw_text(page1, 48, 388, "三、公共信息", size=11.5)
    flag_text = {"yes": "是", "no": "否", "unknown": "未知"}
    left_public = [
        ("呆账标志", flag_text[data["bad_debt"]], "cr.bad_debt_flag", data["bad_debt"]),
        ("资产处置标志", flag_text[data["asset_disposal"]], "cr.asset_disposal_flag", data["asset_disposal"]),
        ("对外担保标志", flag_text[data["guarantee"]], "cr.guarantee_flag", data["guarantee"]),
    ]
    right_public = [
        ("公共记录条数", str(data["public_records"]), "cr.public_record_count", str(data["public_records"])),
        ("信用等级", data["rating"], "cr.credit_rating", data["rating"]),
    ]
    for index, (label, display, path, value) in enumerate(left_public):
        set_gt(gt, path, value, 1,
               draw_kv(page1, 55, 170, 418 + index * 34, label, display, size=9))
    for index, (label, display, path, value) in enumerate(right_public):
        set_gt(gt, path, value, 1,
               draw_kv(page1, 312, 420, 418 + index * 34, label, display, size=9))
    add_footer(page1, 1, 2)

    page2 = _new_page(doc)
    draw_text(page2, A4_WIDTH / 2, 55, "企业信用报告（简版）（续）", size=16,
              align="center", bold=True)
    draw_text(page2, 55, 95, "四、未结清贷款明细", size=12)
    rows = [["贷款机构", "业务种类", "余额（元）", "五级分类"]]
    for loan in data["loans"]:
        rows.append([loan["institution"], loan["kind"],
                     format_money_for_print(loan["balance"]), loan["status"]])
    draw_table(page2, 55, 115, [150, 150, 150, 80], 25, rows, font_size=9,
               aligns=["left", "left", "right", "center"])
    draw_text(page2, 55, 330, "说明：本报告为示例合成数据，仅用于系统测试，不构成任何真实信用评价。",
              size=9, color=(0.35, 0.35, 0.35))
    add_footer(page2, 2, 2)
    return gt


# ---------------------------------------------------------------------------
# document construction
# ---------------------------------------------------------------------------

def _company_for(seed: int, number: int) -> Dict:
    company = make_company(random.Random(seed + 100 + number), number)
    # make_company 的年份随编号前移；硬例集编号到 40，会造出未来成立日期，
    # 触发真实管线里的 future_date 校验，淹没退化信号。
    establishment = date(2008 + number % 12, 1 + number % 12, 1 + (number * 3) % 27)
    company["establishment"] = establishment
    company["term_start"] = establishment
    company["term_end"] = establishment.replace(year=establishment.year + 20) - timedelta(days=1)
    company["issue_date"] = establishment + timedelta(days=number % 5)
    company["company_name"] = f"{company['company_name']}（{number:02d}）"
    return company


def _build_base(doc_type: str, company: Dict, rng: random.Random, doc
                ) -> Tuple[GT, Dict]:
    if doc_type == "BUSINESS_LICENSE":
        return build_business_license(doc, company), {}
    if doc_type == "INCOME_STATEMENT":
        data = make_income_data(rng)
        data["period_start"] = date(2025, 1, 1)
        data["period_end"] = date(2025, 12, 31)
        use_wan = rng.random() < 0.5
        # extra_rows 必须和 synth 一致：make_income_data 的营业利润已把新准则四行算进去，
        # 少印这四行的难例件页面上「营业利润」恒对不上，V-IS-01 会把整张表判成跨字段失败，
        # 测出来的是夹具不一致而不是退化信号。
        return build_income_statement(doc, company, data, use_wan,
                                      extra_rows=True), data
    if doc_type == "CREDIT_REPORT_ENTERPRISE":
        data = make_credit_data(rng, company)
        return build_credit_report(doc, company, data), data
    period_start = date(2025, 7, 1)
    period_end = date(2025, 12, 31)
    opening = Decimal(rng.randint(60000, 200000)) + Decimal("0.00")
    txns = make_bank_transactions(rng, period_start, period_end, opening,
                                  rng.randint(12, 16))
    data = {
        "bank_name": rng.choice(["示例银行", "示例商业银行", "示例农村商业银行"]),
        "branch_name": f"{rng.choice(['示例银行', '示例商业银行'])}高新区支行",
        "account_number": "".join(str(rng.randint(0, 9)) for _ in range(19)),
        "period_start": period_start,
        "period_end": period_end,
        "opening_balance": opening,
        "closing_balance": txns[-1]["balance"] if txns else opening,
        "total_income": sum((t["amount"] for t in txns if t["amount"] > 0), Decimal("0.00")),
        "total_expense": -sum((t["amount"] for t in txns if t["amount"] < 0), Decimal("0.00")),
        "transactions": txns,
    }
    return build_bank_statement(doc, company, data), data


def _rasterise(src_path: Path, out_path: Path, gt: GT, rng: random.Random,
               kind: str, spec: Optional[Dict] = None) -> GT:
    spec = spec or RASTER_SPECS[kind]
    transform = IMAGE_DEGRADATIONS[kind]
    angle = rng.uniform(-spec["angle"], spec["angle"])
    dpi = int(spec["dpi"])
    scale = dpi / 72.0
    src = fitz.open(src_path)
    out = fitz.open()
    matrices: Dict[int, Tuple[np.ndarray, float]] = {}
    for page_index, page in enumerate(src):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image = transform(image, rng)
        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = array.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(array, matrix, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        buffer = io.BytesIO()
        Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)).save(
            buffer, format="JPEG", quality=int(spec["quality"]), optimize=True)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=buffer.getvalue())
        matrices[page_index + 1] = (matrix, scale)
    out.save(out_path, garbage=4, deflate=True, no_new_id=True)
    out.close()
    src.close()

    transformed: GT = {}
    for path, entry in gt.items():
        matrix, page_scale = matrices[int(entry["page"])]
        transformed[path] = {
            "value": entry["value"],
            "page": entry["page"],
            "bbox": _transform_bbox(entry["bbox"], matrix, page_scale),
        }
    return transformed


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_generated(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file() and _DOC_ID_RE.match(path.stem):
            path.unlink()


def generate_hard_set(out_dir: Path = HARD_DIR, labels_dir: Path = HARD_LABELS,
                      seed: int = HARD_SET_SEED, per_kind: int = 5) -> Dict:
    out_dir = Path(out_dir)
    labels_dir = Path(labels_dir)
    base_dir = HARD_WORK / "base"
    for directory in (out_dir, labels_dir, base_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_generated(out_dir)
    _clear_generated(labels_dir)

    documents: List[Dict] = []
    counter = 0
    for kind, specs in DEGRADATIONS.items():
        doc_types = PLAN[kind][:per_kind]
        for index, doc_type in enumerate(doc_types, start=1):
            counter += 1
            number = 10 + counter
            doc_id = f"{specs['code']}{index:02d}"
            rng = random.Random(seed * 1000 + counter * 37)
            company = _company_for(seed, number)
            doc = fitz.open()
            if kind == "two_column" and doc_type == "CREDIT_REPORT_ENTERPRISE":
                data = make_credit_data(rng, company)
                gt = build_two_column_credit_report(doc, company, data)
            elif kind == "two_column" and doc_type == "BUSINESS_LICENSE":
                gt = build_two_column_business_license(doc, company)
            else:
                gt, _ = _build_base(doc_type, company, rng, doc)

            final_path = out_dir / f"{doc_id}.pdf"
            if specs["mode"] == "raster":
                base_path = base_dir / f"{doc_id}.pdf"
                doc.save(base_path, garbage=4, deflate=True, no_new_id=True)
                doc.close()
                gt = _rasterise(base_path, final_path, gt, rng, kind)
            else:
                if kind == "broken_lines":
                    _break_table_rules(doc, rng)
                doc.save(final_path, garbage=4, deflate=True, no_new_id=True)
                doc.close()

            label_path = labels_dir / f"{doc_id}.json"
            label_path.write_text(json.dumps({
                "document_id": doc_id,
                "doc_type": doc_type,
                "fields": gt,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

            with fitz.open(final_path) as check:
                page_count = check.page_count
            documents.append({
                "document_id": doc_id,
                "degradation": kind,
                "degradation_label": specs["label"],
                "doc_type": doc_type,
                "format": "scan" if specs["mode"] == "raster" else "text",
                "path": _rel(final_path),
                "label_path": _rel(label_path),
                "sha256": _sha256(final_path),
                "page_count": page_count,
                "company_name": company["company_name"],
                "field_count": len(gt),
            })

    manifest = {
        "version": 2,
        "seed": seed,
        "created_at": now_iso(),
        "note": "P0-1 难例集：合成数据 + 受控退化，仅用于鲁棒性诊断，不进入验收门槛。",
        "degradations": {
            kind: {"label": spec["label"], "mode": spec["mode"],
                   "documents": sum(1 for item in documents if item["degradation"] == kind)}
            for kind, spec in DEGRADATIONS.items()
        },
        "documents": documents,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _load_manifest(input_dir: Path) -> Dict:
    path = Path(input_dir) / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing hard-set manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_by_degradation(records: List[Dict], kind_by_doc: Dict[str, str],
                             targets: Optional[Dict] = None) -> List[Dict]:
    rows = []
    for kind, spec in DEGRADATIONS.items():
        subset = [item for item in records if kind_by_doc.get(item["document_id"]) == kind]
        if not subset:
            continue
        rows.append({
            "name": kind,
            "label": spec["label"],
            "documents": len({item["document_id"] for item in subset}),
            **_decision_stats(subset, targets),
        })
    return rows


def evaluate_hard_set(input_dir: Path = HARD_DIR, labels_dir: Path = HARD_LABELS,
                      out_dir: Path = HARD_OUT, config: Optional[Dict] = None,
                      force: bool = True) -> Dict:
    config = config or load_config()
    input_dir, labels_dir, out_dir = Path(input_dir), Path(labels_dir), Path(out_dir)
    manifest = _load_manifest(input_dir)
    kind_by_doc = {item["document_id"]: item["degradation"]
                   for item in manifest.get("documents", [])}
    doc_type_by_doc = {item["document_id"]: item["doc_type"]
                       for item in manifest.get("documents", [])}

    run(input_dir, out_dir, config=config, force=force)
    evaluation = evaluate_predictions(out_dir, labels_dir, config, include_records=True)
    records = evaluation["records"]

    by_degradation = _aggregate_by_degradation(records, kind_by_doc,
                                               claim_targets(config))

    documents = []
    for item in evaluation["documents"]:
        documents.append({
            "document_id": item["document_id"],
            "degradation": kind_by_doc.get(item["document_id"]),
            "doc_type": doc_type_by_doc.get(item["document_id"], item["doc_type"]),
            "field_count": item["field_count"],
            "accept_count": item["accept_count"],
            "review_count": item["review_count"],
            "auto_accept_rate": item["auto_accept_rate"],
            "accepted_error_count": item["accepted_error_count"],
            "grounding_hit_rate": item["grounding_hit_rate"],
        })

    payload = {
        "generated_at": now_iso(),
        "seed": manifest.get("seed"),
        "input_dir": str(input_dir),
        "labels_dir": str(labels_dir),
        "out_dir": str(out_dir),
        "overall": evaluation["summary"],
        "by_degradation": by_degradation,
        "documents": documents,
        "notes": [
            "合成数据 + 受控退化，只用于定位失败模式，不参与验收门槛。",
            "accepted_error_rate 的分母是自动接受字段数；review_rate 的分母是全部字段数。",
        ],
    }
    report_path = ROOT / "work" / "eval" / "hard_set.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


# ---------------------------------------------------------------------------
# resolution sweep (P0-5b pre-work): what would a graded gate hand back?
# ---------------------------------------------------------------------------

SWEEP_WORK = ROOT / "work" / "res_sweep"
SWEEP_MARKER = ".docpipe-resolution-sweep"
SWEEP_DEFAULTS = {
    "dpis": [120, 150, 180, 220, 260, 300],
    "documents_per_cell": 4,
    "jpeg_qualities": [55, 85],
    "angle": 0.2,
    "doc_types": ["BUSINESS_LICENSE", "BANK_STATEMENT", "INCOME_STATEMENT",
                  "CREDIT_REPORT_ENTERPRISE"],
}


def _sweep_options(config: Dict) -> Dict:
    merged = dict(SWEEP_DEFAULTS)
    merged.update(((config.get("diagnostics") or {}).get("resolution_sweep") or {}))
    return merged


def _clear_sweep_work() -> None:
    """Wipe only a directory this probe owns, marked by its own sentinel file."""
    if SWEEP_WORK.exists():
        if not (SWEEP_WORK / SWEEP_MARKER).exists() and any(SWEEP_WORK.iterdir()):
            raise SystemExit(f"{SWEEP_WORK} 不是分辨率扫描生成的目录，拒绝清空")
        shutil.rmtree(SWEEP_WORK)
    SWEEP_WORK.mkdir(parents=True, exist_ok=True)
    (SWEEP_WORK / SWEEP_MARKER).write_text(
        "docpipe.hardset sweep 生成，可安全删除。\n", encoding="utf-8")


def _tier_stats(records: List[Dict]) -> Dict:
    """Accuracy for one risk tier at one resolution, plus the automation it would keep.

    ``would_accept`` comes from a run with the resolution gate switched off, so it is
    exactly the set a tier exemption hands back: OCR confidence, grounding and the
    cross-field rules still had to pass on their own. ``would_accept_wrong`` is the error
    price of that exemption and is listed field by field rather than folded into a rate.
    """
    accepted = [item for item in records if item.get("decision") == "accept"]
    wrong = [item for item in accepted if not item.get("correct")]
    good = len(accepted) - len(wrong)
    hits = sum(1 for item in records if item.get("correct"))
    interval = wilson_interval(good, len(accepted)) if accepted else None
    return {
        "fields": len(records),
        "value_accuracy": round(hits / len(records), 4) if records else None,
        "would_accept": len(accepted),
        "would_accept_correct": good,
        "would_accept_wrong": len(wrong),
        "would_accept_accuracy": round(good / len(accepted), 4) if accepted else None,
        "would_accept_wilson_low": round(interval[0], 4) if interval else None,
        "wrong_examples": [{"document_id": item["document_id"], "path": item["path"],
                            "gt": item["gt"], "pred": item["pred"]} for item in wrong][:5],
    }


def sweep_resolution(config: Optional[Dict] = None, force: bool = True) -> Dict:
    """Rasterise one synthetic corpus across dpi x JPEG quality and attribute by risk tier.

    Diagnostic only: it never changes samples/hard, labels, or any production threshold.
    """
    config = config or load_config()
    options = _sweep_options(config)
    dpis = [int(dpi) for dpi in options["dpis"]]
    qualities = [int(quality) for quality in options["jpeg_qualities"]]
    doc_types = list(options["doc_types"])
    per_cell = int(options["documents_per_cell"])
    samples, labels, outs = (SWEEP_WORK / name for name in ("samples", "labels", "out"))
    _clear_sweep_work()
    for directory in (samples, labels, outs):
        directory.mkdir(parents=True, exist_ok=True)

    # Counterfactual run: keep the meter (parse still reports effective dpi) but drop the
    # page-wide abstention, otherwise every field lands in review and nothing is measurable.
    probe_config = copy.deepcopy(config)
    probe_config["validation"]["review_on_low_resolution"] = False
    parsed_dir = resolve_path(probe_config["paths"]["work"], probe_config) / "parsed"

    cells = [(dpi, quality) for quality in qualities for dpi in dpis]

    # Paired design: the same base documents are rasterised into every cell, and the
    # degradation draws (angle) are seeded per document rather than per cell, so a cell's
    # numbers differ from another cell's only by dpi x JPEG. Unpaired cells were useless:
    # two runs of 120q55 gave 37 and 1 new silent errors, and content-paired but
    # angle-unpaired cells made 110q85 look ten times worse than 110q55.
    panel: List[Tuple[str, Path, GT]] = []
    for index in range(per_cell):
        doc_type = doc_types[index % len(doc_types)]
        rng = random.Random(HARD_SET_SEED * 7 + index)
        doc = fitz.open()
        gt, _ = _build_base(doc_type, _company_for(HARD_SET_SEED, index), rng, doc)
        base_path = SWEEP_WORK / f"SWpanel_{index + 1:02d}.base.pdf"
        doc.save(base_path, garbage=4, deflate=True)
        doc.close()
        panel.append((doc_type, base_path, gt))

    built: Dict[str, List[str]] = {}
    for dpi, quality in cells:
        cell = f"{dpi}q{quality}"
        sample_dir, label_dir = samples / cell, labels / cell
        sample_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        spec = {"dpi": dpi, "quality": quality, "angle": float(options["angle"])}
        for index, (doc_type, base_path, gt) in enumerate(panel):
            doc_id = f"SW{cell}_{index + 1:02d}"
            rng = random.Random(HARD_SET_SEED * 11 + index)
            raster_gt = _rasterise(base_path, sample_dir / f"{doc_id}.pdf",
                                   copy.deepcopy(gt), rng, "low_resolution", spec)
            (label_dir / f"{doc_id}.json").write_text(json.dumps(
                {"document_id": doc_id, "doc_type": doc_type, "fields": raster_gt},
                ensure_ascii=False, indent=2), encoding="utf-8")
            built.setdefault(cell, []).append(doc_id)
    for _, base_path, _ in panel:
        base_path.unlink(missing_ok=True)

    rows = []
    for dpi, quality in cells:
        cell = f"{dpi}q{quality}"
        sample_dir, label_dir, out_dir = samples / cell, labels / cell, outs / cell
        run(sample_dir, out_dir, config=probe_config, force=force)
        evaluation = evaluate_predictions(out_dir, label_dir, probe_config,
                                          include_records=True)
        measured = []
        for doc_id in built[cell]:
            cache = parsed_dir / f"{doc_id}.json"
            if not cache.exists():
                continue
            parsed = json.loads(cache.read_text(encoding="utf-8"))
            measured.extend(page["raster"]["effective_dpi"]
                            for page in parsed.get("pages", [])
                            if (page.get("raster") or {}).get("effective_dpi"))
        grouped: Dict[str, List[Dict]] = {}
        for record in evaluation["records"]:
            grouped.setdefault(record["risk"], []).append(record)
        rows.append({
            "requested_dpi": dpi,
            "jpeg_quality": quality,
            "measured_dpi": sorted(set(round(float(dpi_value), 1) for dpi_value in measured)),
            "documents": len(built[cell]),
            "field_count": len(evaluation["records"]),
            "all": _tier_stats(evaluation["records"]),
            "by_risk": {risk: _tier_stats(recs) for risk, recs in sorted(grouped.items())},
        })
        print(f"dpi {dpi} jpeg {quality}: fields={rows[-1]['field_count']} "
              f"would_accept={rows[-1]['all']['would_accept']} "
              f"wrong={rows[-1]['all']['would_accept_wrong']}")
        for doc_id in built[cell]:
            (parsed_dir / f"{doc_id}.json").unlink(missing_ok=True)

    payload = {
        "generated_at": now_iso(),
        "seed": HARD_SET_SEED,
        "sweep": options,
        "method": ("把同一批基文档（配对设计：每份都栅格化进每一格）按 dpi x JPEG 质量退化，关闭页级低分辨率复核门跑真实链路，"
                   "再按 (dpi, JPEG, risk) 归因。would_accept 是该档字段在关掉分辨率门后仍被自动接受的部分，"
                   "即分级档位能放回去的量；would_accept_wrong 是放回去以后新增的静默错误。"),
        "note": "诊断数据，不进验收门槛；档位阈值只能在这张表面前决定，不得反过来调低既有阈值凑数。",
        "cells": rows,
    }
    report_path = ROOT / "work" / "eval" / "resolution_sweep.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def _print_sweep(payload: Dict) -> None:
    def cell_name(row: Dict) -> str:
        return f"{row['requested_dpi']}q{row['jpeg_quality']}"

    print(f"\n{'cell':>9}{'docs':>6}{'fields':>8}{'accept':>8}{'wrong':>7}{'acc':>8}"
          f"{'wilson':>8}  wrong by risk")
    for row in payload["cells"]:
        stats = row["all"]
        detail = " ".join(f"{risk}={item['would_accept_wrong']}"
                          for risk, item in row["by_risk"].items()
                          if item["would_accept_wrong"])
        print(f"{cell_name(row):>9}{row['documents']:>6}{stats['fields']:>8}"
              f"{stats['would_accept']:>8}{stats['would_accept_wrong']:>7}"
              f"{(stats['value_accuracy'] or 0):>8.3f}"
              f"{(stats['would_accept_wilson_low'] or 0):>8.3f}  {detail}")

    risks = sorted({risk for row in payload["cells"] for risk in row["by_risk"]})
    print(f"\n{'cell':>9}" + "".join(f"{risk[:10]:>11}" for risk in risks))
    print("would-accept accuracy per risk tier (1.000 = 放回去不会引入错值)")
    for row in payload["cells"]:
        line = "".join(
            f"{(row['by_risk'].get(risk) or {}).get('would_accept_accuracy') or 0:>11.3f}"
            for risk in risks)
        print(f"{cell_name(row):>9}{line}")
    print("report:", payload["report_path"])


SWEEP_SIGNAL_REPORT = ROOT / "work" / "eval" / "sweep_signals.json"


def _page_gray(pdf: Path) -> Optional[np.ndarray]:
    """First embedded raster of the page, decoded as grayscale."""
    with fitz.open(pdf) as doc:
        if not doc.page_count:
            return None
        page = doc[0]
        for image in page.get_images(full=True):
            raw = doc.extract_image(image[0]).get("image")
            if raw:
                return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float64)
    return None


def page_signals(gray: np.ndarray) -> Dict:
    """Page-level sharpness and compression proxies.

    Kept as a measurement only: fitting these into the page gate is what the
    sweep has to prove first. Blockiness tracks JPEG strength; Laplacian and the
    high-frequency share mix dpi with stroke density. None of them orders the
    silent-error counts on its own.
    """
    if gray.ndim != 2 or min(gray.shape) < 40:
        return {}
    laplacian = float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())

    vertical = np.abs(np.diff(gray, axis=1))
    v_boundary = (np.arange(vertical.shape[1]) % 8) == 7
    v_ratio = vertical[:, v_boundary].mean() / max(vertical[:, ~v_boundary].mean(), 1e-6)
    horizontal = np.abs(np.diff(gray, axis=0))
    h_boundary = (np.arange(horizontal.shape[0]) % 8) == 7
    h_ratio = horizontal[h_boundary, :].mean() / max(
        horizontal[~h_boundary, :].mean(), 1e-6)

    spectrum = np.fft.rfft2(gray)
    power = np.abs(spectrum) ** 2
    rows, columns = power.shape
    low_band = power[:, :max(1, int(columns * 0.06))].sum()
    total = power.sum()
    return {
        "laplacian_variance": round(laplacian, 1),
        "blockiness": round(float(max(v_ratio, h_ratio)), 3),
        "hf_ratio": round(float((total - low_band) / max(total, 1e-6)), 4),
    }


def probe_page_signals() -> Dict:
    """Measure page signals on the pages a sweep just rasterised, beside its errors.

    Reads ``work/res_sweep/samples`` and ``work/eval/resolution_sweep.json``; nothing
    is re-OCR'd and no threshold is changed. The output is the evidence table for
    whether a second gate axis exists at all.
    """
    sweep_path = ROOT / "work" / "eval" / "resolution_sweep.json"
    if not sweep_path.exists():
        raise SystemExit("先运行 python -m docpipe.hardset sweep，再量页面信号")
    if not (SWEEP_WORK / "samples").is_dir():
        raise SystemExit(f"{SWEEP_WORK / 'samples'} 不存在；先重跑 sweep")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    wrong = {(row["requested_dpi"], row["jpeg_quality"]): row
             for row in sweep["cells"]}

    rows = []
    for cell_dir in sorted((SWEEP_WORK / "samples").iterdir()):
        match = re.fullmatch(r"(\d+)q(\d+)", cell_dir.name)
        if not match or not cell_dir.is_dir():
            continue
        dpi, quality = int(match.group(1)), int(match.group(2))
        metrics = [page_signals(gray) for pdf in sorted(cell_dir.glob("*.pdf"))
                   if (gray := _page_gray(pdf)) is not None]
        metrics = [item for item in metrics if item]
        measured = wrong.get((dpi, quality)) or {}
        rows.append({
            "cell": cell_dir.name,
            "requested_dpi": dpi,
            "jpeg_quality": quality,
            "documents": len(metrics),
            "laplacian_variance": round(float(np.mean([m["laplacian_variance"]
                                                       for m in metrics])), 1) if metrics else None,
            "blockiness": round(float(np.mean([m["blockiness"] for m in metrics])), 3)
            if metrics else None,
            "hf_ratio": round(float(np.mean([m["hf_ratio"] for m in metrics])), 4)
            if metrics else None,
            "would_accept_wrong": (measured.get("all") or {}).get("would_accept_wrong"),
            "value_accuracy": (measured.get("all") or {}).get("value_accuracy"),
        })

    by_dpi: Dict[int, Dict[int, Dict]] = {}
    for row in rows:
        by_dpi.setdefault(row["requested_dpi"], {})[row["jpeg_quality"]] = row
    compression = []
    for dpi, cells in sorted(by_dpi.items()):
        if 55 not in cells or 85 not in cells:
            continue
        compression.append({
            "dpi": dpi,
            "q55_blockiness": cells[55]["blockiness"],
            "q85_blockiness": cells[85]["blockiness"],
            "separates": bool(cells[55]["blockiness"] is not None
                              and cells[85]["blockiness"] is not None
                              and cells[55]["blockiness"] > cells[85]["blockiness"]),
            "q55_wrong": cells[55]["would_accept_wrong"],
            "q85_wrong": cells[85]["would_accept_wrong"],
        })

    payload = {
        "generated_at": now_iso(),
        "sweep_report": str(sweep_path),
        "method": ("对 sweep 已经栅格化好的页面离线算三个整页信号（不重跑 OCR）："
                   "Laplacian 方差、8×8 块效应比、高频能量占比，并与该格 would_accept_wrong 并列。"),
        "note": ("诊断数据，不进验收门槛。块效应稳定区分压缩强度（10/10 个 dpi 上 q55 > q85），"
                 "但绝对量与静默错误数不单调：110q55 1.307 错 26、120q55 1.118 错 48、150q55 1.178 错 2。"
                 "Laplacian 方差混入笔画密度，同 dpi 的 JPEG 差异只有约 2–4%；高频占比随 dpi 上升，"
                 "但 q55/q85 几乎同值。要让第二根轴进信任决策，先证明它能按错误率排序，"
                 "再把信号移到 OCR 实际使用的像素空间（按行的笔画宽度/边缘锐度）。"),
        "compression_pairs": compression,
        "rows": rows,
        "report_path": str(SWEEP_SIGNAL_REPORT),
    }
    SWEEP_SIGNAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_SIGNAL_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return payload


def _print_signals(payload: Dict) -> None:
    print(f"{'cell':>8}{'docs':>6}{'lap_var':>12}{'block8':>10}{'hf_ratio':>10}"
          f"{'wrong':>7}{'acc':>8}")
    for row in payload["rows"]:
        print(f"{row['cell']:>8}{row['documents']:>6}"
              f"{(row['laplacian_variance'] or 0):>12.1f}"
              f"{(row['blockiness'] or 0):>10.3f}"
              f"{(row['hf_ratio'] or 0):>10.4f}"
              f"{(row['would_accept_wrong'] or 0):>7}"
              f"{(row['value_accuracy'] or 0):>8.3f}")
    separated = sum(1 for pair in payload["compression_pairs"] if pair["separates"])
    print(f"块效应 q55 > q85：{separated}/{len(payload['compression_pairs'])} 个 dpi；"
          "绝对量与错误率不成单调，不能单独作为门")
    print("report:", payload["report_path"])


LINE_SIGNAL_REPORT = ROOT / "work" / "eval" / "line_signals.json"
LINE_SIGNAL_NATIVE_REPORT = ROOT / "work" / "eval" / "line_signals_native.json"
LINE_METRICS = ("stroke_ratio", "edge_sharp", "laplacian_rel", "bg_noise")


def _auc(values: List[float], labels: List[bool]) -> Optional[float]:
    """Rank AUC with ties averaged (0.5 = no ordering, 1.0 = all wrong values above)."""
    paired = sorted(zip(values, labels), key=lambda item: item[0])
    positives = sum(1 for _, label in paired if label)
    negatives = len(paired) - positives
    if not positives or not negatives:
        return None
    ranks: List[float] = [0.0] * len(paired)
    index = 0
    while index < len(paired):
        end = index
        while end + 1 < len(paired) and paired[end + 1][0] == paired[index][0]:
            end += 1
        shared = (index + end) / 2.0 + 1.0
        for slot in range(index, end + 1):
            ranks[slot] = shared
        index = end + 1
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, paired) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _line_metrics(gray: np.ndarray, box: List[float]) -> Dict:
    """Stroke width and edge sharpness of one field's line in the OCR input image.

    ``stroke_ratio`` is the median stroke width over the ink height, so it compares
    fonts of different sizes; blur thickens it. ``edge_sharp`` is the 95th percentile
    gradient over the crop contrast; blur lowers it. Both are measured on the pixels
    OCR actually saw, not on the source raster.
    """
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return {}
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    low, high = np.percentile(crop, [5, 95])
    contrast = float(high - low)
    if contrast < 12:
        return {}
    threshold, _ = cv2.threshold(crop.astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = crop < threshold
    share = float(ink.mean())
    if not 0.01 <= share <= 0.75:
        return {}
    rows = np.flatnonzero(ink.any(axis=1))
    ink_height = max(1, int(rows[-1] - rows[0] + 1))
    distance = cv2.distanceTransform((ink * 255).astype(np.uint8), cv2.DIST_L2, 5)
    stroke = 2.0 * float(np.median(distance[ink]))
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    gradient = float(np.percentile(np.hypot(gx, gy), 95))
    laplacian = cv2.Laplacian(crop, cv2.CV_32F)
    paper = crop[~ink]
    return {
        "stroke_ratio": round(stroke / ink_height, 4),
        "edge_sharp": round(gradient / contrast, 4),
        "laplacian_rel": round(float(laplacian.var()) / (contrast ** 2), 4),
        "bg_noise": round(float(paper.std()) / contrast, 4) if paper.size else None,
    }


def _render_ocr_gray(pdf: Path, page_no: int, dpi: int, config: Dict) -> Optional[np.ndarray]:
    """The grayscale array that ``parse.ocr_parts`` hands to OCR, enhancement included."""
    from .preprocess import ocr_input
    with fitz.open(pdf) as doc:
        if page_no < 1 or page_no > doc.page_count:
            return None
        pix = doc[page_no - 1].get_pixmap(dpi=dpi, alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bgr, _ = ocr_input(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), config)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _native_gray(pdf: Path, page_no: int) -> Optional[Tuple[np.ndarray, float]]:
    """The embedded raster itself (no re-render), plus pixels per page point.

    The scan fixtures place one full-page JPEG, so the page-point bbox maps linearly.
    A page whose largest image does not cover it would need a per-image transform and
    is skipped rather than measured with the wrong scale.
    """
    with fitz.open(pdf) as doc:
        if page_no < 1 or page_no > doc.page_count:
            return None
        page = doc[page_no - 1]
        best = None
        for image in page.get_images(full=True):
            for shown in page.get_image_rects(image[0]):
                if shown.is_empty:
                    continue
                area = shown.width * shown.height
                if best is None or area > best[0]:
                    best = (area, image[0], shown)
        if best is None:
            return None
        _area, xref, shown = best
        coverage = (shown.width * shown.height) / max(
            1.0, page.rect.width * page.rect.height)
        if coverage < 0.9:
            return None
        info = doc.extract_image(xref)
        if not info.get("image"):
            return None
        picture = Image.open(io.BytesIO(info["image"])).convert("L")
        return (np.asarray(picture, dtype=np.float32),
                picture.width / max(shown.width, 1e-6))


def _rank_within(groups: Dict[str, List[Dict]], metric: str) -> Optional[float]:
    """Pooled AUC after ranking the metric inside each group (cells or cell x risk)."""
    values: List[float] = []
    labels: List[bool] = []
    for items in groups.values():
        measured = [item for item in items if item["metrics"].get(metric) is not None]
        if len(measured) < 3:
            continue
        ordered = sorted(measured, key=lambda item: item["metrics"][metric])
        ranks: List[float] = [0.0] * len(ordered)
        index = 0
        while index < len(ordered):
            end = index
            while end + 1 < len(ordered) and (ordered[end + 1]["metrics"][metric]
                                              == ordered[index]["metrics"][metric]):
                end += 1
            shared = (index + end) / 2.0
            for slot in range(index, end + 1):
                ranks[slot] = shared / max(1, len(ordered) - 1)
            index = end + 1
        for value, item in zip(ranks, ordered):
            values.append(value)
            labels.append(not item["correct"])
    return _auc(values, labels)


def probe_line_signals(config: Optional[Dict] = None,
                       space: str = "ocr") -> Dict:
    """P0-5c-2: can a per-line quality signal separate wrong from right at the same dpi?

    Reuses the sweep's rasterised corpus, labels and predictions, loading each page's
    pixels once and measuring every accepted field's line crop. ``space="ocr"`` uses
    the grayscale array OCR actually saw; ``space="native"`` uses the embedded scan
    so the 300 dpi re-render cannot hide compression. A signal only earns a place in
    calibration if it orders wrong above right (AUC >= 0.8) both inside a cell and
    after stratifying by risk tier; otherwise this records a negative result.
    """
    if space not in ("ocr", "native"):
        raise SystemExit("space 只能是 ocr 或 native")
    config = config or load_config()
    sweep_path = ROOT / "work" / "eval" / "resolution_sweep.json"
    if not sweep_path.exists():
        raise SystemExit("先运行 python -m docpipe.hardset sweep 生成扫描表面")
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    probe_config = copy.deepcopy(config)
    probe_config["validation"]["review_on_low_resolution"] = False
    dpi = int(config["parsing"].get("render_dpi", 300))

    cells: List[Dict] = []
    by_cell: Dict[str, List[Dict]] = {}
    by_cell_risk: Dict[str, List[Dict]] = {}
    for row in sweep["cells"]:
        cell = f"{row['requested_dpi']}q{row['jpeg_quality']}"
        sample_dir = SWEEP_WORK / "samples" / cell
        label_dir = SWEEP_WORK / "labels" / cell
        out_dir = SWEEP_WORK / "out" / cell
        if not (sample_dir.is_dir() and label_dir.is_dir() and out_dir.is_dir()):
            continue
        evaluation = evaluate_predictions(out_dir, label_dir, probe_config,
                                          include_records=True)
        records = {(record["document_id"], record["path"]): record
                   for record in evaluation["records"]}
        items: List[Dict] = []
        for label_path in sorted(label_dir.glob("*.json")):
            doc_id = label_path.stem
            pred_path = out_dir / doc_id / f"{doc_id}.json"
            if not pred_path.exists():
                continue
            prediction = json.loads(pred_path.read_text(encoding="utf-8"))
            images: Dict[int, Optional[Tuple[np.ndarray, float]]] = {}
            for path, field in prediction.get("fields", {}).items():
                record = records.get((doc_id, path))
                if record is None or field.get("decision") != "accept":
                    continue
                grounding = field.get("grounding") or {}
                bbox = grounding.get("bbox")
                page_no = int(grounding.get("page") or 0)
                if not bbox or page_no < 1:
                    continue
                if page_no not in images:
                    source = sample_dir / f"{doc_id}.pdf"
                    images[page_no] = (_native_gray(source, page_no) if space == "native"
                                       else (_render_ocr_gray(source, page_no, dpi, probe_config),
                                             dpi / 72.0))
                cached = images[page_no]
                if cached is None:
                    continue
                gray, scale = cached
                height_px = (float(bbox[3]) - float(bbox[1])) * scale
                pad = max(2.0, 0.12 * height_px)
                box = [float(bbox[0]) * scale - pad, float(bbox[1]) * scale - pad,
                       float(bbox[2]) * scale + pad, float(bbox[3]) * scale + pad]
                metrics = _line_metrics(gray, box)
                if not metrics:
                    continue
                items.append({
                    "document_id": doc_id, "path": path, "page": page_no,
                    "risk": field.get("risk"), "value_type": field.get("value_type"),
                    "correct": bool(record["correct"]),
                    "ocr_conf": (field.get("signals") or {}).get("ocr_conf"),
                    "confidence": field.get("confidence"),
                    "metrics": metrics,
                })
        if not items:
            continue

        def _cell_auc(metric: str) -> Optional[float]:
            measured = [item for item in items if item["metrics"].get(metric) is not None]
            wrong = [item for item in measured if not item["correct"]]
            right = [item for item in measured if item["correct"]]
            if len(wrong) < 5 or len(right) < 20:
                return None
            return _auc([item["metrics"][metric] for item in measured],
                        [not item["correct"] for item in measured])

        per_metric = {}
        for metric in LINE_METRICS:
            measured = [item for item in items if item["metrics"].get(metric) is not None]
            values = [float(item["metrics"][metric]) for item in measured]
            per_metric[metric] = {
                "wrong": sum(1 for item in measured if not item["correct"]),
                "right": sum(1 for item in measured if item["correct"]),
                "mean": round(float(np.mean(values)), 4) if values else None,
                "auc": _cell_auc(metric),
            }
        cells.append({
            "cell": cell,
            "requested_dpi": row["requested_dpi"],
            "jpeg_quality": row["jpeg_quality"],
            "accepted": len(items),
            "wrong": sum(1 for item in items if not item["correct"]),
            "right": sum(1 for item in items if item["correct"]),
            "metrics": {key: {"auc": (None if value["auc"] is None
                                      else round(value["auc"], 4)),
                              "mean": value["mean"],
                              "wrong": value["wrong"], "right": value["right"]}
                        for key, value in per_metric.items()},
            "ocr_conf_auc": _auc(
                [float(item["ocr_conf"]) for item in items
                 if item["ocr_conf"] is not None],
                [not item["correct"] for item in items if item["ocr_conf"] is not None]),
        })
        by_cell[cell] = items
        for item in items:
            by_cell_risk.setdefault(f"{cell}|{item['risk']}", []).append(item)
        print(f"{cell:>8} accepted={len(items):>4} wrong={sum(1 for i in items if not i['correct']):>3} "
              + " ".join(f"{metric}={per_metric[metric]['auc']:.3f}"
                         if per_metric[metric]["auc"] is not None
                         else f"{metric}=n/a" for metric in LINE_METRICS))

    pooled = {metric: _rank_within(by_cell, metric) for metric in LINE_METRICS}
    stratified = {metric: _rank_within(by_cell_risk, metric) for metric in LINE_METRICS}
    separation = {metric: (None if pooled[metric] is None else
                           round(max(pooled[metric], 1.0 - pooled[metric]), 4))
                  for metric in LINE_METRICS}
    separated = [metric for metric in LINE_METRICS
                 if separation[metric] is not None and separation[metric] >= 0.8
                 and stratified[metric] is not None
                 and max(stratified[metric], 1.0 - stratified[metric]) >= 0.8]
    best = max((metric for metric in LINE_METRICS if separation[metric] is not None),
               key=lambda metric: separation[metric], default=None)
    space_name = "OCR 输入" if space == "ocr" else "嵌入扫描原件"
    if separated:
        note = ("线级信号在同一格内且按风险分层后仍把错值排到正确值之上："
                + "、".join(separated) + f"（最高分离度 {separation[best]:.3f}）。"
                "可以作为校准特征继续验证，但本次未改生产阈值。")
    else:
        note = (f"此路不通（{space_name}像素空间）：在同一格内（同 dpi、同 JPEG）把「错值行」与「正确行」"
                f"按笔画宽度/边缘锐度排序，最好的一根是 {best}，分层前 AUC {pooled[best]:.3f}、"
                f"按风险分层后 {stratified[best]:.3f}"
                f"（分离度 {separation[best]:.3f}，门槛 0.80）。错值行没有稳定的更模糊倾向："
                "误读来自字符形状与上下文，而不是整行像素质量。所以线级质量信号不进入校准，也不加新阈值。")
    report_path = LINE_SIGNAL_NATIVE_REPORT if space == "native" else LINE_SIGNAL_REPORT
    payload = {
        "generated_at": now_iso(),
        "space": space,
        "render_dpi": dpi if space == "ocr" else None,
        "method": (f"复用 sweep 的 20 格配对样本、标注与预测，按字段 grounding bbox 在{space_name}"
                   "的灰度图上裁片；只统计自动接受字段，按 (cell) 与 (cell, risk) 分组算错值 vs 正确的"
                   "秩 AUC。AUC > 0.5 表示错值行的该指标更高；分离度取 max(AUC, 1-AUC)。"),
        "note": note,
        "threshold": 0.8,
        "baseline_ocr_conf_auc": {
            row["cell"]: (None if row["ocr_conf_auc"] is None
                          else round(row["ocr_conf_auc"], 4))
            for row in cells},
        "cells": cells,
        "pooled_within_cell": {metric: None if value is None else round(value, 4)
                               for metric, value in pooled.items()},
        "pooled_within_cell_risk": {metric: None if value is None else round(value, 4)
                                    for metric, value in stratified.items()},
        "separation": separation,
        "separated": bool(separated),
        "separated_metrics": separated,
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return payload


def _print_line_signals(payload: Dict) -> None:
    def number(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"space={payload['space']}")
    print(f"\n{'metric':<16}{'AUC(cell)':>11}{'AUC(cell,risk)':>16}{'sep':>8}")
    for metric in LINE_METRICS:
        print(f"{metric:<16}"
              f"{number(payload['pooled_within_cell'].get(metric)):>11}"
              f"{number(payload['pooled_within_cell_risk'].get(metric)):>16}"
              f"{number(payload['separation'].get(metric)):>8}")
    print(payload["note"])
    print("report:", payload["report_path"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_eval(payload: Dict) -> None:
    print(f"{'degradation':<18}{'docs':>5}{'fields':>7}{'accept':>9}{'acc_err':>9}"
          f"{'review':>9}{'ground':>9}")
    for row in payload["by_degradation"]:
        print(f"{row['name']:<18}{row['documents']:>5}{row['field_count']:>7}"
              f"{row['auto_accept_rate']:>9.3f}{row['accepted_error_rate']:>9.3f}"
              f"{row['review_rate']:>9.3f}"
              f"{(row['grounding_hit_rate'] or 0.0):>9.3f}")
    overall = payload["overall"]
    print(f"{'overall':<18}{overall['latency']['documents']:>5}{overall['field_count']:>7}"
          f"{overall['auto_accept_rate']:>9.3f}{overall['accepted_error_rate']:>9.3f}"
          f"{overall['review_rate']:>9.3f}"
          f"{(overall['grounding_hit_rate'] or 0.0):>9.3f}")
    print("report:", payload["report_path"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="P0-1 hard set: generate and evaluate")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write samples/hard + labels/hard")
    gen.add_argument("--out", default=str(HARD_DIR))
    gen.add_argument("--labels", default=str(HARD_LABELS))
    gen.add_argument("--seed", type=int, default=HARD_SET_SEED)
    gen.add_argument("--per-kind", type=int, default=5)
    ev = sub.add_parser("eval", help="run the pipeline and attribute by degradation")
    ev.add_argument("--input", default=str(HARD_DIR))
    ev.add_argument("--labels", default=str(HARD_LABELS))
    ev.add_argument("--out", default=str(HARD_OUT))
    ev.add_argument("--no-force", action="store_true")
    sw = sub.add_parser("sweep", help="dpi ladder x risk tier accuracy (P0-5b measurement)")
    sw.add_argument("--no-force", action="store_true")
    sub.add_parser("signals", help="page sharpness/compression proxies beside sweep errors")
    lines = sub.add_parser("linesignals",
                           help="line stroke/sharpness vs accepted errors (P0-5c-2)")
    lines.add_argument("--space", choices=("ocr", "native"), default="ocr",
                       help="ocr = the array handed to OCR; native = the embedded scan")
    args = parser.parse_args(argv)

    if args.command == "generate":
        manifest = generate_hard_set(Path(args.out), Path(args.labels),
                                     args.seed, args.per_kind)
        for item in manifest["documents"]:
            print(f"{item['document_id']}  {item['degradation']:<14} "
                  f"{item['doc_type']:<24} {item['format']:<5} "
                  f"pages={item['page_count']}  {item['path']}")
        print(f"generated {len(manifest['documents'])} documents -> {args.out}")
        return

    if args.command == "sweep":
        _print_sweep(sweep_resolution(force=not args.no_force))
        return

    if args.command == "signals":
        _print_signals(probe_page_signals())
        return

    if args.command == "linesignals":
        _print_line_signals(probe_line_signals(space=args.space))
        return

    payload = evaluate_hard_set(Path(args.input), Path(args.labels), Path(args.out),
                                force=not args.no_force)
    _print_eval(payload)


if __name__ == "__main__":
    main()
