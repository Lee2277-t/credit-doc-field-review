"""Text normalization and value parsing shared by synthesis, extraction and evaluation."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
MONEY_CENT = Decimal("0.01")


def normalize_text(value) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(ZERO_WIDTH)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_code(value) -> str:
    text = normalize_text(value).upper()
    return re.sub(r"[\s\-]", "", text)


def csv_cell(value, value_type: Optional[str] = None):
    """Keep spreadsheet programs from treating document text as a formula."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    if value_type in {"decimal", "integer"} and re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value):
        return value
    if value.startswith(("'", "\t", "\r", "\n")) or normalize_text(value).startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def strip_label(text: str, label: str) -> str:
    text = normalize_text(text)
    for candidate in (label, f"{label}：", f"{label}:"):
        if text.startswith(candidate):
            return text[len(candidate):].strip()
    return text


def parse_decimal(value, unit_hint: Optional[str] = None) -> Optional[Decimal]:
    """Parse a Chinese financial number into a Decimal. Returns None if not parseable.

    Handles commas, currency symbols, trailing minus, parentheses, and 万/万元.
    """
    if value is None:
        return None
    text = normalize_text(value)
    if not text or text in {"-", "--", "—", "－", "***", "无", "不适用"}:
        return None
    text = text.replace("￥", "").replace("¥", "").replace("RMB", "").replace("CNY", "")
    text = text.replace("人民币", "").replace("元", "元")
    # OCR often reads the ¥/￥ glyph as a Latin Y on printed tax forms.
    text = re.sub(r"^[Yy](?=\d)", "", text)
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.startswith("（") and text.endswith("）"):
        negative = True
        text = text[1:-1]
    if text.endswith("-"):
        negative = True
        text = text[:-1]
    text = text.strip().replace(",", "").replace("，", "").replace(" ", "")
    multiplier = Decimal(1)
    if "万元" in text:
        multiplier = Decimal(10000)
        text = text.replace("万元", "")
    elif text.endswith("万"):
        multiplier = Decimal(10000)
        text = text[:-1]
    elif "元" in text:
        text = text.replace("元", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if negative:
        number = -number
    number *= multiplier
    if unit_hint == "wan":
        number *= Decimal(10000)
    return number.quantize(MONEY_CENT, rounding=ROUND_HALF_UP)


def decimal_str(value, places: str = "0.01") -> str:
    if value is None:
        return ""
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return str(value.quantize(Decimal(places), rounding=ROUND_HALF_UP))


DATE_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[-/.年](?P<m>\d{1,2})[-/.月](?P<d>\d{1,2})日?"),
    re.compile(r"(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})"),
]


def parse_date(value, allow_month: bool = False) -> Optional[date]:
    if value is None:
        return None
    text = normalize_text(value)
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
            except ValueError:
                return None
    # OCR often drops or misplaces separators ("2025-1109", "202511-17").
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) == 8 and digits[:2] in {"19", "20"}:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
        except ValueError:
            return None
    if allow_month:
        match = re.search(r"(?P<y>\d{4})[-/.年](?P<m>\d{1,2})月?", text)
        if match:
            try:
                return date(int(match.group("y")), int(match.group("m")), 1)
            except ValueError:
                return None
    return None


def parse_period(value) -> tuple[Optional[date], Optional[date]]:
    if value is None:
        return None, None
    text = normalize_text(value)
    parts = re.split(r"\s*(?:至|到|~|～|—|－)\s*", text, maxsplit=1)
    if len(parts) == 2:
        return parse_date(parts[0]), parse_date(parts[1])
    dates = re.findall(r"\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?", text)
    if len(dates) >= 2:
        return parse_date(dates[0]), parse_date(dates[1])
    return None, None


def parse_int(value) -> Optional[int]:
    if value is None:
        return None
    text = normalize_text(value).replace(",", "").replace("，", "")
    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def format_money_for_print(value: Decimal, unit: str = "yuan", parentheses_negative: bool = False) -> str:
    number = Decimal(value)
    if unit == "wan":
        number = (number / Decimal(10000)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        suffix = ""
    else:
        number = number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        suffix = ""
    negative = number < 0
    absolute = abs(number)
    text = f"{absolute:,.2f}{suffix}"
    if negative:
        return f"({text})" if parentheses_negative else f"-{text}"
    return text


def uscc_check_char(first17: str) -> str:
    charset = "0123456789ABCDEFGHJKLMNPQRTUWXY"
    weights = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]
    total = sum(charset.index(ch) * weight for ch, weight in zip(first17, weights))
    remainder = 31 - total % 31
    if remainder == 31:
        remainder = 0
    return charset[remainder]


def validate_uscc(value: str) -> bool:
    code = normalize_code(value)
    if not re.fullmatch(r"[0-9A-HJ-NP-RTUW-Y]{18}", code):
        return False
    try:
        return uscc_check_char(code[:17]) == code[17]
    except ValueError:
        return False


def make_uscc(prefix: str, body: str) -> str:
    raw = (prefix + body)[:17]
    return raw + uscc_check_char(raw)


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
