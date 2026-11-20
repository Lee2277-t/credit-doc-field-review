"""Synthetic credit-review document generator.

The generator writes text-layer PDFs and image-only "scanned" PDFs plus exact
field labels (value + page + bbox). It deliberately keeps every entity fictional.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageEnhance, ImageFilter

from .textutil import (
    decimal_str,
    format_money_for_print,
    make_uscc,
    normalize_text,
)

ROOT = Path(__file__).resolve().parent.parent
A4_WIDTH = 595.28
A4_HEIGHT = 841.89
FONT_FILE = Path("C:/Windows/Fonts/simsun.ttc")
BOLD_FONT_FILE = Path("C:/Windows/Fonts/simhei.ttf")
FONT = "F0"
FONT_BOLD = "F1"
_FONTS = {
    FONT: fitz.Font(fontfile=str(FONT_FILE)),
    FONT_BOLD: fitz.Font(fontfile=str(BOLD_FONT_FILE)),
}
SEED = 20261024

COMPANY_PREFIXES = ["示例华信", "示例恒达", "示例远航", "示例嘉禾", "示例致远", "示例启明"]
COMPANY_SUFFIXES = ["科技有限公司", "商贸有限公司", "机械制造有限公司", "供应链管理有限公司"]
LEGAL_REPS = ["陈嘉禾", "林致远", "周明远", "许清和", "沈知远", "顾南乔"]
ADDRESSES = [
    "示例市高新区创新大道88号3幢5层",
    "示例市经开区创业路16号A座1203室",
    "示例市滨江区科技园东路9号2号楼",
    "示例市新城区兴业街66号1幢8层",
]
SCOPES = [
    "计算机软硬件技术开发、技术咨询、技术服务；电子产品、办公用品销售；企业管理咨询。",
    "机械设备及配件的研发、生产、销售；货物进出口；技术进出口；普通货物仓储服务。",
    "建材、金属材料、五金交电、日用百货的批发与零售；供应链管理服务；商务信息咨询。",
    "食品经营（仅销售预包装食品）；农副产品收购与销售；电子商务技术服务；道路货物运输。",
]
BANKS = ["示例银行", "示例商业银行", "示例农村商业银行"]
BRANCHES = ["高新区支行", "经开区支行", "滨江支行", "新城支行"]
SUMMARIES = ["货款", "采购款", "工资", "报销", "税费", "服务费", "利息", "转账", "租金", "保证金"]
COUNTERPARTIES = [
    "示例供应链有限公司",
    "示例商贸有限公司",
    "示例物流有限公司",
    "示例信息技术有限公司",
    "示例设备制造有限公司",
    "示例物业服务有限公司",
]
LOAN_INSTITUTIONS = ["示例银行", "示例商业银行", "示例村镇银行", "示例融资担保公司"]

GTValue = Dict[str, object]
GT = Dict[str, GTValue]


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _bbox(x: float, y: float, width: float, size: float) -> List[float]:
    return [round(x, 2), round(y - size, 2), round(x + width, 2), round(y + size * 0.22, 2)]


def _new_page(doc, width: float = A4_WIDTH, height: float = A4_HEIGHT):
    page = doc.new_page(width=width, height=height)
    page.insert_font(fontname=FONT, fontfile=str(FONT_FILE))
    page.insert_font(fontname=FONT_BOLD, fontfile=str(BOLD_FONT_FILE))
    return page


def text_width(text: str, size: float, font: str = FONT) -> float:
    return float(_FONTS[font].text_length(text, size))


def fit_size(text: str, max_width: float, base: float, minimum: float = 5.0) -> float:
    size = base
    while size > minimum and text_width(str(text), size) > max_width:
        size -= 0.4
    return max(minimum, size)


def draw_text(page, x: float, y: float, text: str, size: float = 10,
              color=(0, 0, 0), align: str = "left", bold: bool = False) -> List[float]:
    text = str(text)
    font = FONT_BOLD if bold else FONT
    width = text_width(text, size, font)
    if align == "right":
        x = x - width
    elif align == "center":
        x = x - width / 2
    page.insert_text((x, y), text, fontname=font, fontsize=size, color=color)
    return _bbox(x, y, width, size)


def draw_kv(page, x_label: float, x_value: float, y: float, label: str, value: str,
            size: float = 10, value_size: Optional[float] = None) -> List[float]:
    draw_text(page, x_label, y, f"{label}：", size=size)
    return draw_text(page, x_value, y, value, size=value_size or size)


def draw_line(page, x0, y0, x1, y1, width=0.6, color=(0, 0, 0)):
    page.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1), color=color, width=width)


def draw_rect(page, x0, y0, x1, y1, width=0.8, color=(0, 0, 0), fill=None):
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=color, width=width, fill=fill)


def draw_table(page, x: float, y: float, widths: List[float], row_height: float,
               rows: List[List[str]], font_size: float = 8.0,
               aligns: Optional[List[str]] = None,
               header_rows: int = 1) -> Tuple[Dict[Tuple[int, int], Dict], float]:
    aligns = aligns or ["left"] * len(widths)
    row_count = len(rows)
    col_count = len(widths)
    total_width = sum(widths)
    # fills and grid
    for r in range(row_count):
        cy = y + r * row_height
        if r < header_rows:
            page.draw_rect(fitz.Rect(x, cy, x + total_width, cy + row_height),
                           color=None, fill=(0.92, 0.92, 0.92))
        for c in range(col_count):
            cx = x + sum(widths[:c])
            page.draw_rect(fitz.Rect(cx, cy, cx + widths[c], cy + row_height),
                           color=(0.45, 0.45, 0.45), width=0.4)
    for r in range(row_count):
        cy = y + r * row_height
        for c in range(col_count):
            cx = x + sum(widths[:c])
            text = "" if rows[r][c] is None else str(rows[r][c])
            size = fit_size(text, widths[c] - 8, font_size, 5.0)
            if aligns[c] == "right":
                tx = cx + widths[c] - 5
                align = "right"
            elif aligns[c] == "center":
                tx = cx + widths[c] / 2
                align = "center"
            else:
                tx = cx + 4
                align = "left"
            draw_text(page, tx, cy + row_height * 0.68, text, size=size, align=align)
    cells: Dict[Tuple[int, int], Dict] = {}
    for r in range(row_count):
        for c in range(col_count):
            cx = x + sum(widths[:c])
            cy = y + r * row_height
            cells[(r, c)] = {
                "bbox": [round(cx, 2), round(cy, 2), round(cx + widths[c], 2), round(cy + row_height, 2)],
                "text": rows[r][c],
            }
    return cells, y + row_count * row_height


def add_footer(page, page_no: int, total: int):
    draw_text(page, A4_WIDTH / 2, 820, f"第 {page_no} 页 共 {total} 页", size=8,
              color=(0.35, 0.35, 0.35), align="center")


def set_gt(gt: GT, path: str, value: str, page_no: int, bbox: List[float]):
    gt[path] = {"value": str(value), "page": page_no, "bbox": [round(v, 2) for v in bbox]}


# ---------------------------------------------------------------------------
# data generation
# ---------------------------------------------------------------------------

def make_account(rng: random.Random, length: int = 19) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def make_company(rng: random.Random, number: int) -> Dict:
    name = f"{COMPANY_PREFIXES[number % len(COMPANY_PREFIXES)]}{COMPANY_SUFFIXES[number % len(COMPANY_SUFFIXES)]}"
    body = "".join(rng.choice("0123456789ABCDEFGHJKLMNPQRTUWXY") for _ in range(12))
    uscc = make_uscc("91110" + f"{number:02d}", body)
    establishment = date(2012 + number, rng.randint(1, 12), rng.randint(1, 28))
    term_end = establishment.replace(year=establishment.year + 20) - timedelta(days=1)
    capital = rng.choice([Decimal("5000000"), Decimal("10000000"), Decimal("20000000"), Decimal("30000000")])
    return {
        "company_name": name,
        "uscc": uscc,
        "entity_type": "有限责任公司（自然人投资或控股）",
        "legal_rep": LEGAL_REPS[number % len(LEGAL_REPS)],
        "capital": capital,
        "capital_display": rng.choice([f"{int(capital / 10000)}万元", f"{format_money_for_print(capital)}元"]),
        "establishment": establishment,
        "term_start": establishment,
        "term_end": term_end,
        "address": ADDRESSES[number % len(ADDRESSES)],
        "scope": SCOPES[number % len(SCOPES)],
        "authority": "示例市市场监督管理局",
        "issue_date": establishment + timedelta(days=rng.randint(0, 3)),
    }


def make_bank_transactions(rng: random.Random, period_start: date, period_end: date,
                           opening: Decimal, count: int) -> List[Dict]:
    days = max(1, (period_end - period_start).days)
    offsets = sorted(rng.randint(0, days) for _ in range(count))
    balance = opening
    transactions: List[Dict] = []
    for i, offset in enumerate(offsets):
        txn_date = period_start + timedelta(days=offset)
        is_income = rng.random() < 0.44
        if is_income:
            amount = Decimal(rng.randint(1000, 90000)) + Decimal(rng.choice([0, 50, 99])) / Decimal(100)
            amount = amount.quantize(Decimal("0.01"))
        else:
            amount = Decimal(rng.randint(500, 60000)) + Decimal(rng.choice([0, 50, 99])) / Decimal(100)
            amount = amount.quantize(Decimal("0.01"))
            if amount >= balance:
                amount = max(Decimal("100.00"), (balance * Decimal("0.4")).quantize(Decimal("0.01")))
            amount = -amount
        if balance + amount < 0:
            amount = -min(abs(amount), max(Decimal("100.00"), balance - Decimal("100.00")))
        new_balance = (balance + amount).quantize(Decimal("0.01"))
        if new_balance < 0:
            amount = -balance
            new_balance = Decimal("0.00")
        transactions.append({
            "index": i,
            "date": txn_date,
            "summary": rng.choice(SUMMARIES),
            "counterparty": rng.choice(COUNTERPARTIES),
            "counterparty_account": make_account(rng, 16),
            "direction": "income" if amount > 0 else "expense",
            "amount": amount,
            "balance": new_balance,
            "txn_id": f"{txn_date.strftime('%Y%m%d')}{i + 1:04d}",
        })
        balance = new_balance
    return transactions


def make_income_data(rng: random.Random) -> Dict:
    revenue = Decimal(rng.randint(8000000, 30000000)) + Decimal("0.00")
    cost = (revenue * Decimal(str(rng.uniform(0.55, 0.75)))).quantize(Decimal("0.01"))
    taxes = (revenue * Decimal(str(rng.uniform(0.008, 0.015)))).quantize(Decimal("0.01"))
    selling = (revenue * Decimal(str(rng.uniform(0.02, 0.05)))).quantize(Decimal("0.01"))
    admin = (revenue * Decimal(str(rng.uniform(0.04, 0.08)))).quantize(Decimal("0.01"))
    rd = (revenue * Decimal(str(rng.uniform(0.01, 0.04)))).quantize(Decimal("0.01"))
    finance = (revenue * Decimal(str(rng.uniform(-0.01, 0.015)))).quantize(Decimal("0.01"))
    other = (revenue * Decimal(str(rng.uniform(0.0, 0.01)))).quantize(Decimal("0.01"))
    investment = (revenue * Decimal(str(rng.uniform(0.0, 0.01)))).quantize(Decimal("0.01"))
    # 新准则的四行印在「投资收益」与「营业利润」之间，必须和营业利润一起算；
    # 否则页面上的营业利润与各明细行对不上，校验会把整张表判成跨字段失败。
    credit_impairment = -(revenue * Decimal("0.004")).quantize(Decimal("0.01"))
    asset_impairment = -(revenue * Decimal("0.002")).quantize(Decimal("0.01"))
    asset_disposal = (revenue * Decimal("0.003")).quantize(Decimal("0.01"))
    fair_value = (revenue * Decimal("0.0005")).quantize(Decimal("0.01"))
    operating = (revenue - cost - taxes - selling - admin - rd - finance + other + investment
                 + fair_value + credit_impairment + asset_impairment
                 + asset_disposal).quantize(Decimal("0.01"))
    non_op_income = (revenue * Decimal(str(rng.uniform(0.0, 0.005)))).quantize(Decimal("0.01"))
    non_op_expense = (revenue * Decimal(str(rng.uniform(0.0, 0.004)))).quantize(Decimal("0.01"))
    total_profit = (operating + non_op_income - non_op_expense).quantize(Decimal("0.01"))
    income_tax = (max(Decimal("0"), total_profit) * Decimal("0.25")).quantize(Decimal("0.01"))
    net_profit = (total_profit - income_tax).quantize(Decimal("0.01"))
    scale = Decimal(str(rng.uniform(0.72, 0.98)))
    prior_revenue = (revenue * scale).quantize(Decimal("0.01"))
    prior_net = (net_profit * Decimal(str(rng.uniform(0.6, 1.15)))).quantize(Decimal("0.01"))
    return {
        "revenue": revenue, "cost_of_sales": cost, "taxes_surcharges": taxes,
        "selling_expenses": selling, "admin_expenses": admin, "rd_expenses": rd,
        "finance_expenses": finance, "other_income": other,
        "investment_income": investment, "operating_profit": operating,
        "non_operating_income": non_op_income, "non_operating_expense": non_op_expense,
        "total_profit": total_profit, "income_tax": income_tax, "net_profit": net_profit,
        "prior_revenue": prior_revenue, "prior_net_profit": prior_net,
        # 新准则利润表附加行：两笔减值冲减利润，处置/公允价值变动是偶发收益。
        "credit_impairment_loss": credit_impairment,
        "asset_impairment_loss": asset_impairment,
        "asset_disposal_gain": asset_disposal,
        "fair_value_change_gain": fair_value,
        "total_comprehensive_income": (net_profit + revenue * Decimal("0.001")).quantize(Decimal("0.01")),
    }


def make_credit_data(rng: random.Random, company: Dict) -> Dict:
    loan_count = rng.randint(2, 4)
    loans = []
    for i in range(loan_count):
        loans.append({
            "institution": rng.choice(LOAN_INSTITUTIONS),
            "kind": rng.choice(["流动资金贷款", "经营贷款", "信用贷款", "抵押贷款"]),
            "balance": (Decimal(rng.randint(200000, 5000000)) + Decimal("0.00")).quantize(Decimal("0.01")),
            "status": rng.choice(["正常", "正常", "正常", "关注"]),
        })
    outstanding = sum((loan["balance"] for loan in loans), Decimal("0.00"))
    query_date = date(2025, 12, rng.randint(10, 28))
    overdue_accounts = rng.choice([0, 0, 0, 1, 2])
    max_overdue = 0 if overdue_accounts == 0 else rng.randint(1, 6)
    return {
        "report_number": f"ECR{query_date.strftime('%Y%m%d')}{rng.randint(100000, 999999)}",
        "query_date": query_date,
        "report_period_start": date(2020, 1, 1),
        "report_period_end": query_date,
        "loan_count": loan_count,
        "loans": loans,
        "outstanding": outstanding,
        "overdue_accounts": overdue_accounts,
        "max_overdue_months": max_overdue,
        "query_count_3m": rng.randint(0, 8),
        "bad_debt": "no",
        "asset_disposal": rng.choice(["no", "no", "yes"]),
        "guarantee": rng.choice(["no", "no", "yes"]),
        "public_records": rng.choice([0, 0, 0, 1, 2]),
        "rating": rng.choice(["AA", "AA", "A", "AAA"]),
    }


# ---------------------------------------------------------------------------
# document builders
# ---------------------------------------------------------------------------

def build_business_license(doc, company: Dict) -> GT:
    gt: GT = {}
    page = _new_page(doc)
    page_no = 1
    draw_rect(page, 35, 35, 560, 795, width=1.2)
    draw_text(page, A4_WIDTH / 2, 82, "营业执照（副本）", size=20, align="center", bold=True)
    draw_text(page, 60, 115, "统一社会信用代码：", size=10)
    set_gt(gt, "bl.unified_social_credit_code", company["uscc"], page_no,
           draw_text(page, 172, 115, company["uscc"], size=11))

    rows = [
        ("名称", company["company_name"]),
        ("类型", company["entity_type"]),
        ("法定代表人", company["legal_rep"]),
        ("注册资本", company["capital_display"]),
        ("成立日期", company["establishment"].isoformat()),
        ("营业期限", f"{company['term_start'].isoformat()} 至 {company['term_end'].isoformat()}"),
        ("住所", company["address"]),
        ("经营范围", company["scope"]),
        ("登记机关", company["authority"]),
        ("发照日期", company["issue_date"].isoformat()),
    ]
    field_map = {
        "名称": "bl.company_name",
        "类型": "bl.entity_type",
        "法定代表人": "bl.legal_representative",
        "注册资本": "bl.registered_capital",
        "成立日期": "bl.establishment_date",
        "经营范围": "bl.business_scope",
        "登记机关": "bl.registration_authority",
        "发照日期": "bl.issue_date",
        "住所": "bl.address",
    }
    y = 155
    row_h = 48
    for i, (label, value) in enumerate(rows):
        cy = y + i * row_h
        draw_line(page, 55, cy, 540, cy, width=0.5, color=(0.55, 0.55, 0.55))
        draw_text(page, 68, cy + 27, f"{label}：", size=10)
        size = 10 if label not in {"经营范围", "住所", "类型"} else 8.5
        bbox = draw_text(page, 170, cy + 27, value, size=size)
        if label in field_map:
            set_gt(gt, field_map[label], value, page_no, bbox)
        if label == "营业期限":
            set_gt(gt, "bl.business_term_start", company["term_start"].isoformat(), page_no, bbox)
            set_gt(gt, "bl.business_term_end", company["term_end"].isoformat(), page_no, bbox)
        if label == "注册资本":
            gt["bl.registered_capital"]["value"] = decimal_str(company["capital"])
    draw_line(page, 55, y + len(rows) * row_h, 540, y + len(rows) * row_h,
              width=0.5, color=(0.55, 0.55, 0.55))
    # simple red registration seal, kept away from field values
    page.draw_circle(fitz.Point(480, 690), 48, color=(0.8, 0.1, 0.1), width=2.0)
    draw_text(page, 480, 686, "示例市", size=9, color=(0.8, 0.1, 0.1), align="center")
    draw_text(page, 480, 700, "登记专用章", size=9, color=(0.8, 0.1, 0.1), align="center")
    add_footer(page, 1, 1)
    return gt


def _bank_page_title(page, continuation: bool):
    title = "企业对公账户交易明细" + ("（续）" if continuation else "")
    draw_text(page, A4_WIDTH / 2, 52, title, size=16, align="center", bold=True)


def _bank_table(page, y: float, txns: List[Dict], start_index: int,
                gt: GT, page_no: int) -> float:
    widths = [66, 56, 78, 86, 34, 66, 64, 72]
    header = [["交易日期", "摘要", "对方户名", "对方账号", "收支", "交易金额", "余额", "交易流水号"]]
    rows = header + [[
        t["date"].isoformat(), t["summary"], t["counterparty"],
        t["counterparty_account"], "收入" if t["direction"] == "income" else "支出",
        format_money_for_print(t["amount"]), format_money_for_print(t["balance"]), t["txn_id"]
    ] for t in txns]
    cells, end_y = draw_table(page, 35, y, widths, 22, rows, font_size=7.0,
                              aligns=["left", "left", "left", "left", "center", "right", "right", "left"])
    for local_i, txn in enumerate(txns):
        gt_path = f"bs.transactions[{start_index + local_i}]"
        set_gt(gt, gt_path + ".txn_date", txn["date"].isoformat(), page_no,
               cells[(local_i + 1, 0)]["bbox"])
        set_gt(gt, gt_path + ".summary", txn["summary"], page_no, cells[(local_i + 1, 1)]["bbox"])
        set_gt(gt, gt_path + ".counterparty_name", txn["counterparty"], page_no, cells[(local_i + 1, 2)]["bbox"])
        set_gt(gt, gt_path + ".counterparty_account", txn["counterparty_account"], page_no,
               cells[(local_i + 1, 3)]["bbox"])
        set_gt(gt, gt_path + ".direction", txn["direction"], page_no, cells[(local_i + 1, 4)]["bbox"])
        set_gt(gt, gt_path + ".amount", decimal_str(txn["amount"]), page_no, cells[(local_i + 1, 5)]["bbox"])
        set_gt(gt, gt_path + ".balance", decimal_str(txn["balance"]), page_no, cells[(local_i + 1, 6)]["bbox"])
        set_gt(gt, gt_path + ".txn_id", txn["txn_id"], page_no, cells[(local_i + 1, 7)]["bbox"])
    return end_y


def build_bank_statement(doc, company: Dict, data: Dict) -> GT:
    gt: GT = {}
    txns: List[Dict] = data["transactions"]
    first_cap = 18
    cont_cap = 24
    if len(txns) <= first_cap:
        chunks = [txns]
    else:
        chunks = [txns[:first_cap]]
        rest = txns[first_cap:]
        for i in range(0, len(rest), cont_cap):
            chunks.append(rest[i:i + cont_cap])
    total_pages = len(chunks)
    # summary needs a page if the last table ends too low
    last_rows = len(chunks[-1])
    last_table_end = (210 + 22 * (last_rows + 1)) if len(chunks) == 1 else (105 + 22 * (last_rows + 1))
    summary_page = None
    if last_table_end > 700:
        total_pages += 1
        summary_page = total_pages
    index = 0
    for page_idx, chunk in enumerate(chunks):
        page = _new_page(doc)
        page_no = page_idx + 1
        continuation = page_idx > 0
        _bank_page_title(page, continuation)
        if not continuation:
            draw_text(page, 50, 88, data["bank_name"], size=11)
            set_gt(gt, "bs.account_name", company["company_name"], page_no,
                   draw_kv(page, 50, 120, 112, "户名", company["company_name"], size=10))
            draw_text(page, 330, 112, "账号：", size=10)
            set_gt(gt, "bs.account_number", data["account_number"], page_no,
                   draw_text(page, 380, 112, data["account_number"], size=9.5))
            set_gt(gt, "bs.bank_name", data["bank_name"], page_no,
                   draw_kv(page, 50, 120, 136, "开户行", data["bank_name"], size=10))
            set_gt(gt, "bs.branch_name", data["branch_name"], page_no,
                   draw_kv(page, 330, 410, 136, "开户支行", data["branch_name"], size=9))
            draw_text(page, 50, 160, "币种：", size=10)
            set_gt(gt, "bs.currency", "CNY", page_no, draw_text(page, 105, 160, "人民币", size=10))
            draw_text(page, 330, 160, "期初余额：", size=10)
            set_gt(gt, "bs.opening_balance", decimal_str(data["opening_balance"]), page_no,
                   draw_text(page, 410, 160, format_money_for_print(data["opening_balance"]), size=10))
            draw_text(page, 50, 184, "流水期间：", size=10)
            period_text = f"{data['period_start'].isoformat()} 至 {data['period_end'].isoformat()}"
            period_bbox = draw_text(page, 130, 184, period_text, size=9.5)
            set_gt(gt, "bs.period_start", data["period_start"].isoformat(), page_no, period_bbox)
            set_gt(gt, "bs.period_end", data["period_end"].isoformat(), page_no, period_bbox)
            table_y = 210
        else:
            draw_text(page, 50, 82, f"户名：{company['company_name']}", size=9)
            draw_text(page, 330, 82, f"账号：{data['account_number']}", size=9)
            table_y = 105
        table_end = _bank_table(page, table_y, chunk, index, gt, page_no)
        index += len(chunk)
        add_footer(page, page_no, total_pages)
        if page_idx == len(chunks) - 1 and summary_page is None:
            _draw_bank_summary(page, table_end + 30, data, gt, page_no)
    if summary_page is not None:
        page = _new_page(doc)
        _bank_page_title(page, True)
        _draw_bank_summary(page, 120, data, gt, summary_page)
        add_footer(page, summary_page, total_pages)
    return gt


def _draw_bank_summary(page, y: float, data: Dict, gt: GT, page_no: int):
    draw_text(page, 50, y, "合计信息", size=12)
    y += 28
    set_gt(gt, "bs.closing_balance", decimal_str(data["closing_balance"]), page_no,
           draw_kv(page, 50, 150, y, "期末余额", format_money_for_print(data["closing_balance"]), size=10))
    y += 24
    set_gt(gt, "bs.total_income", decimal_str(data["total_income"]), page_no,
           draw_kv(page, 50, 150, y, "收入合计", format_money_for_print(data["total_income"]), size=10))
    y += 24
    set_gt(gt, "bs.total_expense", decimal_str(data["total_expense"]), page_no,
           draw_kv(page, 50, 150, y, "支出合计", format_money_for_print(data["total_expense"]), size=10))


def build_debit_credit_fixture(doc, company: Dict, data: Dict) -> GT:
    """银行流水「借方发生额 / 贷方发生额」双列版式，仅供单测。

    对公流水更常见的导出版式其实不是「收支方向 + 交易金额」，而是把两列金额
    并排印出来，方向只能从「哪一列有数」读出来。这里没有竖线，走的仍是
    版面块/段引擎路径，用来验证 optional_headers 与 column_enum 的取数。
    """
    gt: GT = {}
    txns: List[Dict] = data["transactions"]
    page = _new_page(doc)
    _bank_page_title(page, False)
    draw_kv(page, 50, 120, 112, "户名", company["company_name"], size=10)
    draw_text(page, 330, 112, "账号：", size=10)
    draw_text(page, 380, 112, data["account_number"], size=9.5)
    draw_kv(page, 50, 120, 136, "开户行", data["bank_name"], size=10)
    draw_text(page, 50, 160, "币种：", size=10)
    draw_text(page, 105, 160, "人民币", size=10)
    draw_text(page, 330, 160, "期初余额：", size=10)
    draw_text(page, 410, 160, format_money_for_print(data["opening_balance"]), size=10)
    draw_text(page, 50, 184, "流水期间：", size=10)
    draw_text(page, 130, 184,
              f"{data['period_start'].isoformat()} 至 {data['period_end'].isoformat()}", size=9.5)

    widths = [58, 62, 90, 74, 52, 52, 56, 52]
    headers = ["交易日期", "摘要", "对方户名", "对方账号",
               "借方发生额", "贷方发生额", "余额", "交易流水号"]
    x = 35.0
    for index, header in enumerate(headers):
        align = "left" if index == 0 else "center"
        tx = x + 4 if align == "left" else x + widths[index] / 2
        draw_text(page, tx, 230, header, size=8, align=align)
        x += widths[index]
    for row_index, txn in enumerate(txns):
        y = 252 + row_index * 20
        debit = format_money_for_print(abs(txn["amount"])) if txn["direction"] == "expense" else ""
        credit = format_money_for_print(abs(txn["amount"])) if txn["direction"] == "income" else ""
        values = [txn["date"].isoformat(), txn["summary"], txn["counterparty"],
                  txn["counterparty_account"], debit, credit,
                  format_money_for_print(txn["balance"]), txn["txn_id"]]
        boxes: List[List[float]] = []
        x = 35.0
        for index, value in enumerate(values):
            align = "right" if index in (4, 5, 6) else ("left" if index in (0, 1, 2, 3, 7) else "center")
            tx = x + (widths[index] - 5 if align == "right" else 4)
            boxes.append(draw_text(page, tx, y, value, size=8, align=align)
                         if value else [0.0, 0.0, 0.0, 0.0])
            x += widths[index]
        gt_path = f"bs.transactions[{row_index}]"
        set_gt(gt, gt_path + ".txn_date", txn["date"].isoformat(), 1, boxes[0])
        set_gt(gt, gt_path + ".summary", txn["summary"], 1, boxes[1])
        set_gt(gt, gt_path + ".counterparty_name", txn["counterparty"], 1, boxes[2])
        set_gt(gt, gt_path + ".counterparty_account", txn["counterparty_account"], 1, boxes[3])
        set_gt(gt, gt_path + ".direction", txn["direction"], 1, boxes[4 if txn["direction"] == "expense" else 5])
        set_gt(gt, gt_path + ".amount", decimal_str(txn["amount"]),
               1, boxes[4 if txn["direction"] == "expense" else 5])
        set_gt(gt, gt_path + ".balance", decimal_str(txn["balance"]), 1, boxes[6])
        set_gt(gt, gt_path + ".txn_id", txn["txn_id"], 1, boxes[7])
    table_end = 252 + len(txns) * 20
    page.draw_line(fitz.Point(35, 216), fitz.Point(531, 216), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(35, 236), fitz.Point(531, 236), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(35, table_end - 14), fitz.Point(531, table_end - 14),
                   color=(0, 0, 0), width=0.5)
    add_footer(page, 1, 1)
    return gt


def build_bank_statement_fixture(doc, company: Dict, data: Dict, extra: str = "") -> GT:
    """One-page statement for table-robustness tests.

    No vertical ruling lines on purpose: pdfplumber lattice needs verticals to
    build cells, so the segment engine runs even on a text-layer page — exactly
    the path where layout assistance and merged-cell detection must work.
    Horizontal rules stay: they give PyMuPDF's layout model a table-shaped
    signal (it labels fully borderless text as plain 'text') without enabling
    lattice. Never part of the golden sets.

    ``extra``:
      "two_tables"  adds a small 利息明细 table below the main one with the same
                    x offsets — without layout assistance its rows pollute the
                    transactions table;
      "stray_note"  adds an out-of-column annotation beside one data row — the
                    column matcher must reject it and record an anomaly.
    """
    gt: GT = {}
    txns: List[Dict] = data["transactions"]
    page = _new_page(doc)
    _bank_page_title(page, False)
    draw_kv(page, 50, 120, 112, "户名", company["company_name"], size=10)
    draw_text(page, 330, 112, "账号：", size=10)
    draw_text(page, 380, 112, data["account_number"], size=9.5)
    draw_kv(page, 50, 120, 136, "开户行", data["bank_name"], size=10)
    draw_text(page, 50, 160, "币种：", size=10)
    draw_text(page, 105, 160, "人民币", size=10)
    draw_text(page, 330, 160, "期初余额：", size=10)
    draw_text(page, 410, 160, format_money_for_print(data["opening_balance"]), size=10)
    draw_text(page, 50, 184, "流水期间：", size=10)
    draw_text(page, 130, 184,
              f"{data['period_start'].isoformat()} 至 {data['period_end'].isoformat()}", size=9.5)

    widths = [66, 84, 90, 34, 66, 64]
    headers = ["交易日期", "摘要", "对方户名", "收支", "交易金额", "余额"]
    x = 35.0
    for index, header in enumerate(headers):
        align = "left" if index == 0 else "center"
        tx = x + 4 if align == "left" else x + widths[index] / 2
        draw_text(page, tx, 230, header, size=9, align=align)
        x += widths[index]
    for row_index, txn in enumerate(txns):
        y = 252 + row_index * 20
        values = [txn["date"].isoformat(), txn["summary"], txn["counterparty"],
                  "收入" if txn["direction"] == "income" else "支出",
                  format_money_for_print(txn["amount"]),
                  format_money_for_print(txn["balance"])]
        boxes: List[List[float]] = []
        x = 35.0
        for index, value in enumerate(values):
            align = "right" if index in (4, 5) else ("center" if index == 3 else "left")
            tx = x + (widths[index] - 5 if align == "right"
                      else widths[index] / 2 if align == "center" else 4)
            boxes.append(draw_text(page, tx, y, value, size=9, align=align))
            x += widths[index]
        gt_path = f"bs.transactions[{row_index}]"
        set_gt(gt, gt_path + ".txn_date", txn["date"].isoformat(), 1, boxes[0])
        set_gt(gt, gt_path + ".summary", txn["summary"], 1, boxes[1])
        set_gt(gt, gt_path + ".counterparty_name", txn["counterparty"], 1, boxes[2])
        set_gt(gt, gt_path + ".direction", txn["direction"], 1, boxes[3])
        set_gt(gt, gt_path + ".amount", decimal_str(txn["amount"]), 1, boxes[4])
        set_gt(gt, gt_path + ".balance", decimal_str(txn["balance"]), 1, boxes[5])
    table_end = 252 + len(txns) * 20
    # 水平框线（无竖线）：给版面模型的表格信号，又不让 pdfplumber 成格。
    page.draw_line(fitz.Point(35, 216), fitz.Point(445, 216), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(35, 236), fitz.Point(445, 236), color=(0, 0, 0), width=0.5)
    page.draw_line(fitz.Point(35, table_end - 14), fitz.Point(445, table_end - 14),
                   color=(0, 0, 0), width=0.5)

    if extra == "two_tables":
        y = table_end + 40
        page.draw_line(fitz.Point(35, y - 13), fitz.Point(285, y - 13),
                       color=(0, 0, 0), width=0.5)
        page.draw_line(fitz.Point(35, y + 4), fitz.Point(285, y + 4),
                       color=(0, 0, 0), width=0.5)
        page.draw_line(fitz.Point(35, y + 48), fitz.Point(285, y + 48),
                       color=(0, 0, 0), width=0.5)
        draw_text(page, 39, y, "交易日期", size=9)
        draw_text(page, 35 + 66 + 4, y, "摘要", size=9)
        draw_text(page, 35 + 66 + 84 + 4, y, "利息金额", size=9)
        for row, (day, amount) in enumerate((("2025-08-15", "123.45"),
                                             ("2025-11-20", "98.76"))):
            yy = y + 22 + row * 20
            draw_text(page, 39, yy, day, size=9)
            draw_text(page, 35 + 66 + 4, yy, "利息", size=9)
            draw_text(page, 35 + 66 + 84 + 90 - 5, yy, amount, size=9, align="right")
    elif extra == "stray_note":
        # 与第 4 个数据行同一行高，横坐标落在所有列半径之外。
        draw_text(page, 530, 252 + 3 * 20, "复核：张三", size=9)
    add_footer(page, 1, 1)
    return gt


def build_income_statement(doc, company: Dict, data: Dict, use_wan: bool,
                           extra_rows: bool = False) -> GT:
    gt: GT = {}
    page = _new_page(doc)
    draw_text(page, A4_WIDTH / 2, 55, "利润表", size=18, align="center", bold=True)
    set_gt(gt, "is.company_name", company["company_name"], 1,
           draw_kv(page, 50, 135, 92, "编制单位", company["company_name"], size=10))
    unit_text = "万元" if use_wan else "元"
    set_gt(gt, "is.currency_unit", "CNY_WAN_YUAN" if use_wan else "CNY_YUAN", 1,
           draw_kv(page, 470, 515, 92, "单位", unit_text, size=10))
    period_text = f"{data['period_start'].isoformat()} 至 {data['period_end'].isoformat()}"
    set_gt(gt, "is.statement_period_start", data["period_start"].isoformat(), 1,
           draw_kv(page, 50, 135, 120, "报表期间", period_text, size=10))
    set_gt(gt, "is.statement_period_end", data["period_end"].isoformat(), 1,
           gt["is.statement_period_start"]["bbox"])

    rows_data = [
        ("营业收入", "revenue", "prior_revenue"),
        ("营业成本", "cost_of_sales", None),
        ("税金及附加", "taxes_surcharges", None),
        ("销售费用", "selling_expenses", None),
        ("管理费用", "admin_expenses", None),
        ("研发费用", "rd_expenses", None),
        ("财务费用", "finance_expenses", None),
        ("其他收益", "other_income", None),
        ("投资收益", "investment_income", None),
        ("营业利润", "operating_profit", None),
        ("营业外收入", "non_operating_income", None),
        ("营业外支出", "non_operating_expense", None),
        ("利润总额", "total_profit", None),
        ("所得税费用", "income_tax", None),
        ("净利润", "net_profit", "prior_net_profit"),
    ]
    if extra_rows:
        # 新准则模板的四个增值行插在投资收益与营业利润之间；综合收益总额收在最后。
        # 数据由调用方给（单元测试自己造数），不参与金标集生成。
        rows_data = (rows_data[:9]
                     + [("信用减值损失", "credit_impairment_loss", None),
                        ("资产减值损失", "asset_impairment_loss", None),
                        ("资产处置收益", "asset_disposal_gain", None),
                        ("公允价值变动收益", "fair_value_change_gain", None)]
                     + rows_data[9:]
                     + [("综合收益总额", "total_comprehensive_income", None)])
    unit_arg = "wan" if use_wan else "yuan"

    def displayed_yuan(value: Decimal) -> Decimal:
        # GT must match what the page actually shows: 万元 values round to 0.01.
        if use_wan:
            return (value / Decimal(10000)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 10000
        return value

    rows = [["项目", "本期金额", "上期金额"]]
    for label, key, prior_key in rows_data:
        current = format_money_for_print(data[key], unit=unit_arg, parentheses_negative=True)
        prior = format_money_for_print(data[prior_key], unit=unit_arg, parentheses_negative=True) if prior_key else ""
        rows.append([label, current, prior])
    cells, _ = draw_table(page, 40, 150, [280, 130, 130], 25, rows, font_size=9,
                          aligns=["left", "right", "right"])
    for r, (label, key, prior_key) in enumerate(rows_data, start=1):
        set_gt(gt, f"is.{key}", decimal_str(displayed_yuan(data[key])), 1, cells[(r, 1)]["bbox"])
        if prior_key:
            set_gt(gt, f"is.{prior_key}", decimal_str(displayed_yuan(data[prior_key])), 1, cells[(r, 2)]["bbox"])
    draw_text(page, 60, 760, f"法定代表人：{company['legal_rep']}", size=9)
    draw_text(page, 360, 760, "制表人：财务部", size=9)
    add_footer(page, 1, 1)
    return gt


def build_credit_report(doc, company: Dict, data: Dict) -> GT:
    gt: GT = {}
    page1 = _new_page(doc)
    draw_text(page1, A4_WIDTH / 2, 55, "企业信用报告（简版）", size=18, align="center", bold=True)
    set_gt(gt, "cr.report_number", data["report_number"], 1,
           draw_kv(page1, 60, 175, 90, "报告编号", data["report_number"], size=9.5))
    set_gt(gt, "cr.query_date", data["query_date"].isoformat(), 1,
           draw_kv(page1, 380, 455, 90, "查询日期", data["query_date"].isoformat(), size=9.5))

    draw_text(page1, 55, 125, "一、企业基本信息", size=12)
    y = 150
    info = [
        ("企业名称", company["company_name"], "cr.company_name"),
        ("统一社会信用代码", company["uscc"], "cr.unified_social_credit_code"),
        ("法定代表人", company["legal_rep"], None),
        ("注册地址", company["address"], None),
    ]
    for label, value, path in info:
        bbox = draw_kv(page1, 65, 190, y, label, value, size=9.5)
        if path:
            set_gt(gt, path, value, 1, bbox)
        y += 24
    report_period = f"{data['report_period_start'].isoformat()} 至 {data['report_period_end'].isoformat()}"
    period_bbox = draw_kv(page1, 65, 190, y, "报告期间", report_period, size=9.5)
    set_gt(gt, "cr.report_period_start", data["report_period_start"].isoformat(), 1, period_bbox)

    draw_text(page1, 55, 290, "二、信贷信息概要", size=12)
    y = 315
    summary_fields = [
        ("未结清贷款笔数", str(data["loan_count"]), "cr.loan_account_count"),
        ("未结清贷款余额", format_money_for_print(data["outstanding"]), "cr.outstanding_loan_balance"),
        ("近24个月逾期账户数", str(data["overdue_accounts"]), "cr.overdue_account_count_24m"),
        ("最长逾期月数", str(data["max_overdue_months"]), "cr.max_overdue_months"),
        ("近3个月查询次数", str(data["query_count_3m"]), "cr.query_count_3m"),
    ]
    for label, value, path in summary_fields:
        bbox = draw_kv(page1, 65, 215, y, label, value, size=9.5)
        set_gt(gt, path, value, 1, bbox)
        y += 24

    draw_text(page1, 55, 455, "三、公共信息", size=12)
    y = 480
    flag_text = {"yes": "是", "no": "否", "unknown": "未知"}
    public_fields = [
        ("呆账标志", flag_text[data["bad_debt"]], "cr.bad_debt_flag", data["bad_debt"]),
        ("资产处置标志", flag_text[data["asset_disposal"]], "cr.asset_disposal_flag", data["asset_disposal"]),
        ("对外担保标志", flag_text[data["guarantee"]], "cr.guarantee_flag", data["guarantee"]),
        ("公共记录条数", str(data["public_records"]), "cr.public_record_count", str(data["public_records"])),
        ("信用等级", data["rating"], "cr.credit_rating", data["rating"]),
    ]
    for label, display, path, value in public_fields:
        bbox = draw_kv(page1, 65, 215, y, label, display, size=9.5)
        set_gt(gt, path, value, 1, bbox)
        y += 24
    add_footer(page1, 1, 2)

    page2 = _new_page(doc)
    draw_text(page2, A4_WIDTH / 2, 55, "企业信用报告（简版）（续）", size=16, align="center", bold=True)
    draw_text(page2, 55, 95, "四、未结清贷款明细", size=12)
    rows = [["贷款机构", "业务种类", "余额（元）", "五级分类"]]
    for loan in data["loans"]:
        rows.append([loan["institution"], loan["kind"], format_money_for_print(loan["balance"]), loan["status"]])
    draw_table(page2, 55, 115, [150, 150, 150, 80], 25, rows, font_size=9,
               aligns=["left", "left", "right", "center"])
    draw_text(page2, 55, 330, "说明：本报告为示例合成数据，仅用于系统测试，不构成任何真实信用评价。",
              size=9, color=(0.35, 0.35, 0.35))
    add_footer(page2, 2, 2)
    return gt


# ---------------------------------------------------------------------------
# scan conversion
# ---------------------------------------------------------------------------

def _transform_bbox(bbox: List[float], matrix: np.ndarray, scale: float) -> List[float]:
    points = np.array([
        [bbox[0] * scale, bbox[1] * scale, 1.0],
        [bbox[2] * scale, bbox[1] * scale, 1.0],
        [bbox[0] * scale, bbox[3] * scale, 1.0],
        [bbox[2] * scale, bbox[3] * scale, 1.0],
    ])
    transformed = points @ matrix.T
    xs = transformed[:, 0] / scale
    ys = transformed[:, 1] / scale
    return [round(float(xs.min()), 2), round(float(ys.min()), 2),
            round(float(xs.max()), 2), round(float(ys.max()), 2)]


def scan_pdf(src_path: Path, out_path: Path, gt: GT, rng: random.Random,
             dpi: int = 300, degraded: bool = False) -> GT:
    src = fitz.open(src_path)
    out = fitz.open()
    page_angles: Dict[int, Tuple[float, np.ndarray, float]] = {}
    for page_index, page in enumerate(src):
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if degraded:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.7))
            image = ImageEnhance.Contrast(image).enhance(0.82)
            angle = rng.uniform(-0.9, 0.9)
            quality = 62
        else:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.32))
            image = ImageEnhance.Contrast(image).enhance(0.96)
            angle = rng.uniform(-0.55, 0.55)
            quality = 82
        array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = array.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(array, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        buffer = io.BytesIO()
        Image.fromarray(rotated).save(buffer, format="JPEG", quality=quality, optimize=True)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=buffer.getvalue())
        scale = dpi / 72.0
        page_angles[page_index + 1] = (angle, matrix, scale)
    out.save(out_path, garbage=4, deflate=True, no_new_id=True)
    out.close()
    src.close()

    transformed: GT = {}
    for path, entry in gt.items():
        page_no = int(entry["page"])
        angle, matrix, scale = page_angles[page_no]
        transformed[path] = {
            "value": entry["value"],
            "page": page_no,
            "bbox": _transform_bbox(entry["bbox"], matrix, scale),
        }
    return transformed


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

DOC_TYPES = {
    "D001": "BUSINESS_LICENSE", "D002": "BUSINESS_LICENSE", "D003": "BUSINESS_LICENSE",
    "D004": "BANK_STATEMENT", "D005": "BANK_STATEMENT", "D006": "BANK_STATEMENT",
    "D007": "INCOME_STATEMENT", "D008": "INCOME_STATEMENT",
    "D009": "CREDIT_REPORT_ENTERPRISE", "D010": "CREDIT_REPORT_ENTERPRISE",
    "H001": "BUSINESS_LICENSE", "H002": "BUSINESS_LICENSE", "H003": "BUSINESS_LICENSE",
    "H004": "BANK_STATEMENT", "H005": "BANK_STATEMENT", "H006": "BANK_STATEMENT",
    "H007": "INCOME_STATEMENT", "H008": "INCOME_STATEMENT",
    "H009": "CREDIT_REPORT_ENTERPRISE", "H010": "CREDIT_REPORT_ENTERPRISE",
}
SCAN_DOCS = {"D002", "D003", "D005", "D008", "D010", "H002", "H003", "H005", "H008", "H010"}
DEGRADED_DOCS = {"D010", "H010"}
COMPANY_INDEX = {
    "D001": 0, "D004": 0, "D007": 0, "D009": 0,
    "D002": 1, "D005": 1, "D008": 1, "D010": 1,
    "D003": 2, "D006": 2,
    "H001": 3, "H004": 3, "H007": 3, "H009": 3,
    "H002": 4, "H005": 4, "H008": 4, "H010": 4,
    "H003": 5, "H006": 5,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_license(doc, company: Dict, rng: random.Random, doc_id: str):
    return build_business_license(doc, company)


def _build_bank(doc, company: Dict, rng: random.Random, doc_id: str):
    period_start = date(2025, 7, 1)
    period_end = date(2025, 12, 31)
    opening = Decimal(rng.randint(60000, 200000)) + Decimal("0.00")
    txns = make_bank_transactions(rng, period_start, period_end, opening, rng.randint(45, 60))
    total_income = sum((t["amount"] for t in txns if t["amount"] > 0), Decimal("0.00"))
    total_expense = -sum((t["amount"] for t in txns if t["amount"] < 0), Decimal("0.00"))
    data = {
        "bank_name": rng.choice(BANKS),
        "branch_name": f"{rng.choice(BANKS)}{rng.choice(BRANCHES)}",
        "account_number": make_account(rng, rng.choice([16, 17, 19])),
        "period_start": period_start,
        "period_end": period_end,
        "opening_balance": opening,
        "closing_balance": txns[-1]["balance"] if txns else opening,
        "total_income": total_income,
        "total_expense": total_expense,
        "transactions": txns,
    }
    return build_bank_statement(doc, company, data)


def _build_income(doc, company: Dict, rng: random.Random, doc_id: str):
    data = make_income_data(rng)
    data["period_start"] = date(2025, 1, 1)
    data["period_end"] = date(2025, 12, 31)
    use_wan = (doc_id in {"D008", "H007"})
    return build_income_statement(doc, company, data, use_wan, extra_rows=True)


def _build_credit(doc, company: Dict, rng: random.Random, doc_id: str):
    data = make_credit_data(rng, company)
    return build_credit_report(doc, company, data)


# 合成集按文档类型分派构建器。这里过去是 `else:` 兜底成征信报告：合成集里只要出现一个
# 没写构建器的类型，产出的 PDF 就是一份征信报告，标注却声称是那个类型，整条流水线照常
# 跑绿。新类型要么注册在这里，要么别进合成集——`evaluation.doc_type_contract.require_synth`
# 与 docpipe.doctypeaudit 会对着这张表逐类对账。
SYNTH_BUILDERS = {
    "BUSINESS_LICENSE": _build_license,
    "BANK_STATEMENT": _build_bank,
    "INCOME_STATEMENT": _build_income,
    "CREDIT_REPORT_ENTERPRISE": _build_credit,
}


def generate_document(doc_id: str, split: str, companies: List[Dict],
                      rng: random.Random, out_dir: Path) -> Dict:
    doc_type = DOC_TYPES[doc_id]
    company = companies[COMPANY_INDEX[doc_id]]
    builder = SYNTH_BUILDERS.get(doc_type)
    if builder is None:
        raise ValueError(f"文档类型 {doc_type!r} 没有合成构建器：先在 docpipe/synth.py 的 "
                         f"SYNTH_BUILDERS 注册一个，或把它从合成集里拿掉")
    doc = fitz.open()
    gt = builder(doc, company, rng, doc_id)

    text_path = out_dir / f"{doc_id}.pdf"
    doc.save(text_path, garbage=4, deflate=True, no_new_id=True)
    doc.close()

    final_path = text_path
    if doc_id in SCAN_DOCS:
        scan_path = out_dir / f"{doc_id}.scan.pdf"
        gt = scan_pdf(text_path, scan_path, gt, rng, degraded=(doc_id in DEGRADED_DOCS))
        text_path.unlink()
        final_path = scan_path

    label_dir = ROOT / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{doc_id}.json"
    label_path.write_text(json.dumps({
        "document_id": doc_id,
        "doc_type": doc_type,
        "fields": gt,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with fitz.open(final_path) as check:
        page_count = check.page_count
    return {
        "document_id": doc_id,
        "split": split,
        "doc_type": doc_type,
        "format": "scan" if doc_id in SCAN_DOCS else "text",
        "degraded": doc_id in DEGRADED_DOCS,
        "path": (final_path.relative_to(ROOT) if final_path.is_relative_to(ROOT)
                 else final_path).as_posix(),
        "label_path": str(label_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(final_path),
        "page_count": page_count,
        "company_name": company["company_name"],
        "company_index": COMPANY_INDEX[doc_id],
    }


def generate_all(out_root: Path = ROOT / "samples", seed: int = SEED):
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (ROOT / "work" / "synth").mkdir(parents=True, exist_ok=True)
    rng_master = random.Random(seed)
    companies = [make_company(random.Random(seed + 100 + i), i) for i in range(6)]
    documents = []
    for doc_id in sorted(DOC_TYPES):
        split = "dev" if doc_id.startswith("D") else "holdout"
        target_dir = out_root / split
        target_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed * 1000 + sum(ord(ch) for ch in doc_id))
        documents.append(generate_document(doc_id, split, companies, rng, target_dir))
    manifest = {
        "version": 2,
        "seed": seed,
        "created_at": "2026-10-24",
        "note": "全部企业、人名、账号、报告编号均为虚构合成数据，仅用于管线测试。",
        "documents": documents,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic credit-review samples")
    parser.add_argument("--out", default=str(ROOT / "samples"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    manifest = generate_all(Path(args.out), args.seed)
    print(f"generated {len(manifest['documents'])} documents -> {args.out}")
    for doc in manifest["documents"]:
        print(f"  {doc['document_id']}  {doc['doc_type']:<24} {doc['format']:<5} "
              f"pages={doc['page_count']}  {doc['path']}")


if __name__ == "__main__":
    main()
