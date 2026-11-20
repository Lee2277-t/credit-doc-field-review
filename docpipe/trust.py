"""Trust layer: validators, signal fusion, calibration and accept/review decisions.

All thresholds, tolerances and reason codes come from config/scenario.yaml.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import ROOT, field_spec, load_config, risk_threshold
from .extract import normalize_enum
from .textutil import normalize_code, normalize_text, now_iso, parse_date, parse_decimal
from .textutil import validate_uscc
from .parse import row_accounting_options

TRUST_VERSION = "0.1.0"

CROSS_FIELD_IDS = {"V-BS-01", "V-BS-02", "V-BS-03", "V-BS-05", "V-BS-06", "V-BS-07",
                   "V-IS-01", "V-IS-02", "V-IS-03", "V-IS-04", "V-IS-05", "V-IS-06",
                   "V-VI-01", "V-VI-02", "V-VI-03", "V-VI-04",
                   "V-CV-01", "V-CV-02", "V-CV-03",
                   "V-QR-01", "V-X-01", "V-X-02", "V-X-04"}

# SCENARIO §5.4 里代码真会发出的跨文档判决，以及每条参与比对的字段路径。
# 严重度词、是否强制复核、字段清单都不在这里写第二遍——全部读 `cross_document.rules`。
CROSS_DOC_EMITTED = ("V-X-01", "V-X-02", "V-X-04")
CROSS_DOC_USCC_PATHS = ("bl.unified_social_credit_code", "cr.unified_social_credit_code")
CROSS_DOC_NAME_PATHS = {"BUSINESS_LICENSE": "bl.company_name",
                         "BANK_STATEMENT": "bs.account_name",
                         "INCOME_STATEMENT": "is.company_name",
                         "CREDIT_REPORT_ENTERPRISE": "cr.company_name"}

# SCENARIO §8.1 要求 review_reasons 只用 §6.4 的原因码，但本实现会发出的码比那张表多
# （符号推断、局部补读、并框……都是本实现特有的证据）。这份注册表是「代码在发什么」的
# 唯一出处：docpipe/doccontract.py 拿它跟 `hard_abstain_reasons`（谁强制复核）与
# `validation.review_actions`（谁带「怎么办」）逐条对表——三张表任何一张漏一条都会被门槛抓住。
REVIEW_REASON_CODES = (
    "GROUNDING_MISSING", "LOW_OCR", "FORMAT_FAIL", "CROSS_FIELD_FAIL",
    "EXTRACTOR_CONFLICT", "AMBIGUOUS_UNIT", "AMBIGUOUS_DATE", "UNKNOWN_DOC_TYPE",
    "PARSE_LOW_QUALITY", "NON_CNY_CURRENCY", "REQUIRED_EMPTY", "TEMPLATE_MISMATCH",
    "CHAIN_UNRELIABLE", "SIGN_INFERRED", "SEAL_OVERLAP", "TABLE_ANOMALY",
    "QR_CONFLICT", "LOW_RESOLUTION", "LINE_FALLBACK", "MATRIX_FALLBACK",
    "LOCAL_RECROP", "RAW_UNVERIFIED", "PLACEHOLDER_EMPTY", "LOW_CONFIDENCE",
    "ENUM_UNKNOWN",
)
# 与决策一起写下的码不是「哪条证据坏了」，而是「分数没过门」：它不必进
# hard_abstain_reasons（那张表是坏证据清单），但同样必须有文案，而且只许在
# decision 已经落成 review 之后出现。
SELF_ABSTAIN_CODES = ("LOW_CONFIDENCE",)


def normalize_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[i]", path)


def _decimal(field: Optional[Dict]) -> Optional[Decimal]:
    if not field:
        return None
    return parse_decimal(field.get("value"))


def _add(validators: Dict[str, List[Dict]], path: str, rule_id: str, status: str,
         detail: Optional[str] = None):
    validators.setdefault(path, []).append({"id": rule_id, "status": status, "detail": detail})


def _status_of(validators: List[Dict], rule_id: str) -> Optional[str]:
    for item in validators:
        if item["id"] == rule_id:
            return item["status"]
    return None


def _dates_in_period(first: Optional[date], last: Optional[date], slack: int, when: date) -> bool:
    if first is None or last is None:
        return False
    return (first.toordinal() - slack) <= when.toordinal() <= (last.toordinal() + slack)


def validate_document_fields(doc: Dict, parsed: Dict, config: Dict,
                             validators: Dict[str, List[Dict]]) -> None:
    doc_type = doc["doc_type"]
    fields = doc["fields"]
    today = date.today()
    slack = int(config["validation"].get("transaction_date_slack_days", 3))
    future_paths = set(config["validation"].get("future_date_paths") or [])
    future_slack = int(config["validation"].get("future_date_slack_days", 0))
    future_limit = date.fromordinal(today.toordinal() + future_slack)

    for path, field in fields.items():
        spec = field_spec(doc_type, normalize_path(path), config) or {}
        value = field.get("value")
        if value in (None, ""):
            continue
        if spec.get("type") == "date":
            parsed_date = parse_date(value)
            if parsed_date is None:
                _add(validators, path, "V-ID-04", "fail", "invalid date")
            elif parsed_date > future_limit and path in future_paths:
                _add(validators, path, "V-ID-04", "fail", "date in the future")
        rule_ids = set(spec.get("rules") or [])
        if spec.get("type") == "decimal":
            number = parse_decimal(value)
            if number is None:
                _add(validators, path, "V-ID-05", "fail", "unparseable decimal")
            elif re.search(r"\.\d{3,}$", str(value)):
                _add(validators, path, "V-ID-05", "fail", "more than 2 decimals")
            elif number < 0 and "NON_NEGATIVE" in rule_ids:
                # §4.5 的「≥ 0」写在字段表里而不是某条规则行里：哪个字段不许为负
                # 由字段自己的 rules 声明，代码不另存一张路径清单。
                _add(validators, path, "V-ID-05", "fail",
                     "negative amount in a non-negative field")
        if spec.get("type") == "integer":
            try:
                if int(value) < 0:
                    _add(validators, path, "V-ID-06", "fail", "negative count")
            except (TypeError, ValueError):
                _add(validators, path, "V-ID-06", "fail", "not an integer")
        if "USCC" in rule_ids and not validate_uscc(str(value)):
            _add(validators, path, "V-ID-01", "fail", "USCC checksum mismatch")
        if "ACCOUNT" in rule_ids or "ACCOUNT_OPT" in rule_ids:
            code = normalize_code(str(value))
            if not re.fullmatch(r"[0-9*]{8,32}", code):
                _add(validators, path, "V-ID-02", "fail", "account format")
        if "REPORT_NO" in rule_ids:
            code = normalize_code(str(value))
            if not re.fullmatch(r"[A-Z0-9]{8,32}", code):
                _add(validators, path, "V-ID-03", "fail", "report number format")
        if spec.get("type") == "enum" and field.get("status") not in (None, "ok"):
            # SCENARIO §4.2：只接受配置里的枚举；别名映射不唯一也必须进复核，
            # 不许按字典顺序挑第一个。以前这两件事都只写在 payload 的 status 里，
            # 判决层不读它，未映射的原文照样自动接受。
            ambiguous = (field.get("meta") or {}).get("enum_ambiguous") or []
            detail = ("enum alias maps to more than one value: " + "/".join(ambiguous)
                      if ambiguous else "value not in the configured enum")
            _add(validators, path, "V-ID-07", "fail", detail)
        enum_name = spec.get("enum")
        if spec.get("type") == "enum" and enum_name:
            # §4.6 的「未知则复核」：unknown 是配置里认的词（V-ID-07 放行），但
            # 「文件自己说不知道」不等于「读对了」。哪些枚举值必须交人确认由
            # validation.enum_review_values 按枚举名点名，代码不认字面量。
            hold = (config["validation"].get("enum_review_values") or {}).get(enum_name) or []
            if str(value).strip() in {str(item) for item in hold}:
                _add(validators, path, "V-ID-10", "fail",
                     f"enum value {value!r} means the document does not say")
        if spec.get("type") == "string" and (spec.get("min_len") is not None
                                            or spec.get("max_len") is not None):
            # §4.3/§4.4 字段表的「长度 2～80」「非空时长度 ≥ 4」写在硬校验列里：
            # 上下限由字段自己的 min_len/max_len 声明，代码不另存一张长度表。
            # 一个字的「企业名称」几乎一定是标签被切走半截，比读错字更难被发现。
            size = len(normalize_text(str(value)))
            low = spec.get("min_len")
            high = spec.get("max_len")
            if (low is not None and size < int(low)) \
                    or (high is not None and size > int(high)):
                _add(validators, path, "V-ID-09", "fail",
                     f"length {size} outside declared range {low}..{high}")
        code_patterns = config["validation"].get("code_patterns") or {}
        for rule in rule_ids & set(code_patterns):
            pattern = code_patterns[rule]
            if not re.fullmatch(pattern, normalize_code(str(value))):
                _add(validators, path, "V-ID-08", "fail", f"{rule} format")
        if (field.get("meta") or {}).get("qr_conflict") \
                and config["validation"].get("review_on_qr_conflict", True):
            _add(validators, path, "V-QR-01", "fail", "QR payload disagrees with OCR")

    # business license date ordering
    if doc_type == "BUSINESS_LICENSE":
        establishment = parse_date(fields.get("bl.establishment_date", {}).get("value"))
        issue = parse_date(fields.get("bl.issue_date", {}).get("value"))
        term_start = parse_date(fields.get("bl.business_term_start", {}).get("value"))
        term_end = parse_date(fields.get("bl.business_term_end", {}).get("value"))
        if establishment and issue and establishment > issue:
            for path in ("bl.establishment_date", "bl.issue_date"):
                _add(validators, path, "V-ID-04", "fail", "establishment after issue date")
        if term_start and term_end and term_start > term_end:
            for path in ("bl.business_term_start", "bl.business_term_end"):
                _add(validators, path, "V-ID-04", "fail", "term start after end")
        capital = _decimal(fields.get("bl.registered_capital"))
        if capital is not None and capital <= 0:
            _add(validators, "bl.registered_capital", "V-ID-05", "fail", "capital must be positive")

    if doc_type == "CREDIT_REPORT_ENTERPRISE":
        query = parse_date(fields.get("cr.query_date", {}).get("value"))
        period_start = parse_date(fields.get("cr.report_period_start", {}).get("value"))
        if query and period_start and period_start > query:
            for path in ("cr.query_date", "cr.report_period_start"):
                _add(validators, path, "V-ID-04", "fail", "period start after query date")

    if doc_type == "INCOME_STATEMENT":
        period_start = parse_date(fields.get("is.statement_period_start", {}).get("value"))
        period_end = parse_date(fields.get("is.statement_period_end", {}).get("value"))
        if period_start and period_end and period_start > period_end:
            for path in ("is.statement_period_start", "is.statement_period_end"):
                _add(validators, path, "V-ID-04", "fail", "period start after end")


def _statement_row_accounting(fields: Dict, parsed: Dict, config: Dict) -> Dict:
    """SCENARIO §5.2 V-BS-07：保留下来的行、抽出来的笔数、丢掉的行三者必须对得上。

    解析层每丢掉一行都记一个原因（`blank` / `header` / `footer` / `missing_required` /
    `unaccounted`）；这里只认配置里的良性原因。认不出的丢弃行、以及「某页保留下来的行数
    != 该页抽出的笔数」都要点名到页——一行真交易消失比一行读错更难被发现。
    """
    options = row_accounting_options(config)
    benign = {str(item) for item in (options.get("benign_skip_reasons") or [])}
    kept: Dict[int, int] = defaultdict(int)
    unaccounted: Dict[int, List[str]] = defaultdict(list)
    for page in parsed.get("pages") or []:
        page_no = page.get("page")
        for table in page.get("tables") or []:
            if table.get("name") != "transactions":
                continue
            number = table.get("page")
            if number is None:
                number = page_no
            if number is None:
                continue
            kept[int(number)] += len(table.get("rows") or [])
            for skip in table.get("skipped_rows") or []:
                reason = str(skip.get("reason") or "")
                if reason not in benign:
                    unaccounted[int(number)].append(
                        {"reason": reason, "text": str(skip.get("text") or "")})
    extracted: Dict[int, set] = defaultdict(set)
    for path, field in fields.items():
        match = re.match(r"bs\.transactions\[(\d+)\]\.", path)
        if not match:
            continue
        page = (field.get("grounding") or {}).get("page")
        if page is not None:
            extracted[int(page)].add(int(match.group(1)))
    mismatched = sorted(page for page in kept if len(extracted.get(page, set())) != kept[page])
    return {"kept": dict(sorted(kept.items())),
            "extracted": {page: len(rows) for page, rows in sorted(extracted.items())},
            "unaccounted": {page: items for page, items in sorted(unaccounted.items())},
            "mismatched_pages": mismatched}


def validate_bank_statement(doc: Dict, parsed: Dict, config: Dict,
                            validators: Dict[str, List[Dict]]) -> Dict:
    fields = doc["fields"]
    tolerance = Decimal(str(config["validation"].get("balance_tolerance", 0.01)))
    max_ratio = float(config["validation"].get("max_chain_failure_ratio", 0.1))
    slack = int(config["validation"].get("transaction_date_slack_days", 3))
    period_start = parse_date(fields.get("bs.period_start", {}).get("value"))
    period_end = parse_date(fields.get("bs.period_end", {}).get("value"))
    opening = _decimal(fields.get("bs.opening_balance"))
    closing_field = fields.get("bs.closing_balance")
    closing = _decimal(closing_field)

    rows: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for path, field in fields.items():
        match = re.match(r"bs\.transactions\[(\d+)\]\.(\w+)$", path)
        if match:
            rows[int(match.group(1))][match.group(2)] = field
    indexes = sorted(rows)

    chain_failures = 0
    triggers: List[str] = []
    previous_balance = opening
    previous_date: Optional[date] = None
    for index in indexes:
        row = rows[index]
        amount_path = f"bs.transactions[{index}].amount"
        balance_path = f"bs.transactions[{index}].balance"
        date_path = f"bs.transactions[{index}].txn_date"
        amount, balance = _decimal(row.get("amount")), _decimal(row.get("balance"))
        if balance is not None and balance < 0:
            _add(validators, balance_path, "V-ID-05", "fail", "negative balance")

        txn_date = parse_date(row.get("txn_date", {}).get("value"))
        if txn_date is None:
            _add(validators, date_path, "V-ID-04", "fail", "unparseable transaction date")
        else:
            if period_start and period_end and not _dates_in_period(period_start, period_end, slack, txn_date):
                _add(validators, date_path, "V-BS-04", "fail", "date outside statement period")
            if previous_date and txn_date < previous_date:
                _add(validators, date_path, "V-BS-04", "fail", "dates not monotonic")
            previous_date = txn_date

        direction = row.get("direction", {}).get("value")
        if amount is not None and direction:
            if direction == "income" and amount < 0:
                _add(validators, amount_path, "V-BS-06", "fail", "income with negative amount")
            elif direction == "expense" and amount > 0:
                _add(validators, amount_path, "V-BS-06", "fail", "expense with positive amount")
            sign_source = (row.get("amount", {}).get("meta") or {}).get("sign_source")
            if sign_source == "direction":
                _add(validators, amount_path, "V-BS-06", "warn", "sign inferred from direction")

        if amount is not None and balance is not None and previous_balance is not None:
            expected = previous_balance + amount
            delta = abs(expected - balance)
            status = "pass" if delta <= tolerance else "fail"
            detail = f"prev {previous_balance} + {amount} = {expected}, balance {balance}"
            _add(validators, amount_path, "V-BS-01", status, detail)
            _add(validators, balance_path, "V-BS-01", status, detail)
            if status == "fail":
                chain_failures += 1
        if balance is not None:
            previous_balance = balance

    failure_ratio = chain_failures / len(indexes) if indexes else 0.0
    chain_unreliable = bool(indexes) and failure_ratio > max_ratio
    if chain_unreliable:
        triggers.append("ratio")

    if indexes and closing is not None:
        last_balance = _decimal(rows[indexes[-1]].get("balance"))
        if last_balance is not None:
            status = "pass" if abs(last_balance - closing) <= tolerance else "fail"
            detail = f"last balance {last_balance}, closing {closing}"
            _add(validators, "bs.closing_balance", "V-BS-02", status, detail)
            _add(validators, f"bs.transactions[{indexes[-1]}].balance", "V-BS-02", status, detail)
            if status == "fail":
                chain_unreliable = True
                triggers.append("closing")

    total_income = _decimal(fields.get("bs.total_income"))
    total_expense = _decimal(fields.get("bs.total_expense"))
    if None not in (opening, total_income, total_expense, closing):
        expected = opening + total_income - total_expense
        status = "pass" if abs(expected - closing) <= tolerance else "fail"
        detail = f"opening {opening} + income {total_income} - expense {total_expense} = {expected}"
        for path in ("bs.opening_balance", "bs.total_income", "bs.total_expense", "bs.closing_balance"):
            _add(validators, path, "V-BS-03", status, detail)
        if status == "fail":
            chain_unreliable = True
            triggers.append("totals")

    # account consistency across pages
    account = normalize_code(fields.get("bs.account_number", {}).get("value", ""))
    if account:
        found = set()
        for page in parsed.get("pages", []):
            for line in page.get("lines", []):
                for match in re.finditer(r"账号[:：]?\s*([0-9*]{8,32})", line.get("text", "")):
                    found.add(normalize_code(match.group(1)))
        if any(value != account for value in found):
            chain_unreliable = True
            triggers.append("account")
            for path in list(fields):
                if path.startswith("bs.account_name") or path.endswith(".amount") or path == "bs.account_number":
                    _add(validators, path, "V-BS-05", "fail", "account number differs across pages")

    # V-BS-07：交易笔数与页面上的数据行一致，被丢掉的每一行都要能证明是表头/页脚。
    accounting = _statement_row_accounting(fields, parsed, config)
    flagged = set(accounting["mismatched_pages"]) | set(accounting["unaccounted"])
    if flagged:
        lost_page = any(not accounting["extracted"].get(page) for page in flagged)
        for index in indexes:
            page_field = fields.get(f"bs.transactions[{index}].txn_date") or {}
            page = (page_field.get("grounding") or {}).get("page")
            if page is None:
                continue
            page = int(page)
            if page not in flagged and not lost_page:
                continue
            detail = (f"page {page}: kept {accounting['kept'].get(page)} rows, "
                      f"extracted {accounting['extracted'].get(page, 0)}")
            if accounting["unaccounted"].get(page):
                lost = "; ".join(f"{item['reason']} {item['text']}".strip()
                                 for item in accounting["unaccounted"][page])
                detail += f", dropped {lost}"[:160]
            for key in [item for item in fields
                        if re.match(rf"bs\.transactions\[{index}\]\.", item)]:
                _add(validators, key, "V-BS-07", "fail", detail)
    doc["row_accounting"] = accounting

    if chain_unreliable:
        for path in fields:
            if path in {"bs.opening_balance", "bs.closing_balance", "bs.total_income", "bs.total_expense"} \
                    or re.match(r"bs\.transactions\[\d+\]\.(amount|balance)$", path):
                _add(validators, path, "V-BS-01", "fail", "chain failure propagation")
    return {"chain_failures": chain_failures, "rows": len(indexes),
            "failure_ratio": round(failure_ratio, 4), "chain_unreliable": chain_unreliable,
            "triggers": triggers}


# SCENARIO §5.3 V-IS-04 后半句：本期列与上期列不得混列。列身份唯一能自证的东西是
# 金额框的横向位置：同一页按字段声明的期间列聚中位线，落在另一列带里的数就是混列，
# 两列中位线近到分不开就是无法确认。阈值与列名读配置，代码里不写第二套。
INCOME_COLUMN_KEYS = ("enabled", "table", "period_columns", "min_column_gap",
                      "assignment_margin")


def income_column_options(config: Dict) -> Optional[Dict]:
    """`validation.income_columns` 缺席就整段沉默；配了就必须一条不缺。"""
    section = (config.get("validation") or {}).get("income_columns")
    if section is None:
        return None
    missing = [key for key in INCOME_COLUMN_KEYS if key not in section]
    if missing:
        raise ValueError("validation.income_columns 缺 %s：§5.3 V-IS-04 的列名与阈值"
                         "必须有出处" % missing)
    options = dict(section)
    options["table"] = str(section["table"])
    options["period_columns"] = [str(column) for column in section["period_columns"] or []]
    options["min_column_gap"] = float(section["min_column_gap"])
    options["assignment_margin"] = float(section["assignment_margin"])
    return options


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _income_column_mix(doc: Dict, config: Dict,
                       validators: Dict[str, List[Dict]]) -> Dict:
    """§5.3 V-IS-04 后半句：本期列与上期列不得混列。

    按页聚列：每个金额字段先由字段配置拿到它声明的列，再用溯源框的横向中位线代表它
    实际落在哪里。离另一列的中位线更近（且超出 assignment_margin）就是混列；两列中位线
    近到分不开就是无法确认。契约对这两种情形只写了一个去处——全部金额复核——所以判决
    打在报表的每个金额字段上：只点一个字段，同一套错列在别的行上照样会漏过去。
    """
    options = income_column_options(config)
    if options is None or not options.get("enabled"):
        return {"enabled": False}
    columns = options["period_columns"]
    gap = options["min_column_gap"]
    margin = options["assignment_margin"]
    paths = [path for path, field in doc["fields"].items()
             if field.get("value_type") == "decimal"]
    per_page: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)
    for path in paths:
        field = doc["fields"][path]
        if field.get("value") in (None, ""):
            continue
        spec = field_spec(doc.get("doc_type") or "", path, config) or {}
        if spec.get("extractor") != "table_row" or spec.get("table") != options["table"]:
            continue
        column = str(spec.get("column") or "")
        if column not in columns:
            continue
        grounding = field.get("grounding") or {}
        bbox = grounding.get("bbox") or []
        if len(bbox) < 4:
            continue
        per_page[int(grounding.get("page") or 0)].append(
            (path, column, (float(bbox[0]) + float(bbox[2])) / 2.0))
    mixed: List[Dict] = []
    unconfirmed: List[Dict] = []
    layout: Dict[str, Dict] = {}
    for page in sorted(per_page):
        items = per_page[page]
        centers: Dict[str, List[float]] = defaultdict(list)
        for _path, column, center in items:
            centers[column].append(center)
        present = [column for column in columns if centers.get(column)]
        medians = {column: _median(centers[column]) for column in present}
        layout[str(page)] = {column: {"median": round(medians[column], 2),
                                      "fields": len(centers[column])}
                             for column in present}
        if len(present) < 2:
            # 只有一列金额时没有可混的第二列，条款自然满足。
            continue
        too_close = [(left, right) for index, left in enumerate(present)
                     for right in present[index + 1:]
                     if abs(medians[left] - medians[right]) < gap]
        if too_close:
            for left, right in too_close:
                unconfirmed.append({"page": page, "columns": [left, right],
                                    "gap": round(abs(medians[left] - medians[right]), 2)})
            # 列带本身分不开时，逐字段归列没有意义，不再往下判。
            continue
        for path, column, center in items:
            nearest = min(present, key=lambda name: abs(medians[name] - center))
            if nearest == column:
                continue
            own = abs(medians[column] - center)
            other = abs(medians[nearest] - center)
            item = {"path": path, "declared": column, "nearest": nearest,
                    "center": round(center, 2), "own_median": round(medians[column], 2),
                    "nearest_median": round(medians[nearest], 2)}
            if own - other > margin:
                mixed.append(item)
            else:
                # 两列都说得通：契约不许猜，按「无法确认」处理。
                unconfirmed.append(item)
    if not mixed and not unconfirmed:
        return {"enabled": True, "layout": layout, "mixed": [], "unconfirmed": []}
    problems = []
    if mixed:
        problems.append("mixed " + ", ".join(
            f"{item['path']} declared {item['declared']} but sits at {item['center']} "
            f"(own {item['own_median']}, {item['nearest']} {item['nearest_median']})"
            for item in mixed[:3]))
    if unconfirmed:
        problems.append("unconfirmed " + ", ".join(
            str(item.get("path") or "/".join(item.get("columns") or []))
            for item in unconfirmed[:3]))
    detail = ("period columns not separable: " + "; ".join(problems)
              + f"; layout {json.dumps(layout, ensure_ascii=False)}")[:320]
    for path in paths:
        _add(validators, path, "V-IS-04", "fail", detail)
    return {"enabled": True, "layout": layout, "mixed": mixed, "unconfirmed": unconfirmed}


def validate_income_statement(doc: Dict, config: Dict,
                              validators: Dict[str, List[Dict]]) -> None:
    fields = doc["fields"]
    base_tolerance = Decimal(str(config["validation"].get("accounting_tolerance", 1.0)))
    wan_term = Decimal(str(config["validation"].get("accounting_tolerance_wan_per_term", 50)))
    unit_wan = doc.get("currency_unit") == "CNY_WAN_YUAN"

    def tolerance(terms: int) -> Decimal:
        # Each 万元 operand rounds to 0.01 万元 on the page (±50 元).
        return base_tolerance + (wan_term * terms if unit_wan else Decimal(0))

    def value(path):
        return _decimal(fields.get(path))

    revenue = value("is.revenue")
    cost = value("is.cost_of_sales")
    taxes = value("is.taxes_surcharges")
    selling = value("is.selling_expenses")
    admin = value("is.admin_expenses")
    rd = value("is.rd_expenses") or Decimal(0)
    finance = value("is.finance_expenses")
    other = value("is.other_income") or Decimal(0)
    investment = value("is.investment_income") or Decimal(0)
    # 新准则把公允价值变动、信用/资产减值、资产处置放在投资收益之后；
    # 它们同样参与营业利润，漏掉时这几行非零的报表会整体误判 V-IS-01。
    fair_value = value("is.fair_value_change_gain") or Decimal(0)
    credit_impairment = value("is.credit_impairment_loss") or Decimal(0)
    asset_impairment = value("is.asset_impairment_loss") or Decimal(0)
    asset_disposal = value("is.asset_disposal_gain") or Decimal(0)
    operating = value("is.operating_profit")
    non_op_income = value("is.non_operating_income")
    non_op_expense = value("is.non_operating_expense")
    total_profit = value("is.total_profit")
    income_tax = value("is.income_tax")
    net_profit = value("is.net_profit")

    v_is_01 = None
    if None not in (revenue, cost, taxes, selling, admin, finance, operating):
        expected = (revenue - cost - taxes - selling - admin - rd - finance
                    + other + investment + fair_value + credit_impairment
                    + asset_impairment + asset_disposal)
        status = "pass" if abs(expected - operating) <= tolerance(10) else "fail"
        detail = f"expected operating profit {expected}, extracted {operating}"
        for path in ("is.revenue", "is.cost_of_sales", "is.taxes_surcharges", "is.selling_expenses",
                     "is.admin_expenses", "is.rd_expenses", "is.finance_expenses", "is.other_income",
                     "is.investment_income", "is.fair_value_change_gain",
                     "is.credit_impairment_loss", "is.asset_impairment_loss",
                     "is.asset_disposal_gain", "is.operating_profit"):
            _add(validators, path, "V-IS-01", status, detail)
        v_is_01 = status

    if None not in (operating, non_op_income, non_op_expense, total_profit):
        expected = operating + non_op_income - non_op_expense
        status = "pass" if abs(expected - total_profit) <= tolerance(4) else "fail"
        detail = f"expected total profit {expected}, extracted {total_profit}"
        for path in ("is.operating_profit", "is.non_operating_income",
                     "is.non_operating_expense", "is.total_profit"):
            _add(validators, path, "V-IS-02", status, detail)

    if None not in (total_profit, income_tax, net_profit):
        expected = total_profit - income_tax
        status = "pass" if abs(expected - net_profit) <= tolerance(3) else "fail"
        detail = f"expected net profit {expected}, extracted {net_profit}"
        for path in ("is.total_profit", "is.income_tax", "is.net_profit"):
            _add(validators, path, "V-IS-03", status, detail)

    # §4.5 is.income_tax「≤ 利润总额（软）」：所得税费用高过利润总额，通常是税格
    # 或利润总额读错；亏损弥补、汇算清缴退税时也可能真出现，所以只降 cross_field
    # 信号（§6.1 的 warn = 0.3），严重度词与开关都读配置。亏损表的所得税本来就是
    # 0，0 > 负的利润总额在数值上成立却毫无异常含义，所以只在利润总额为正时比较。
    ceiling = config["validation"].get("income_tax_ceiling") or {}
    if ceiling.get("enabled", True) and None not in (total_profit, income_tax) \
            and total_profit > 0 and income_tax > total_profit + tolerance(2):
        _add(validators, "is.income_tax", "V-IS-06", str(ceiling.get("severity", "warn")),
             f"income tax {income_tax} exceeds total profit {total_profit}")

    _income_column_mix(doc, config, validators)
    unit = doc.get("currency_unit")
    if unit is None:
        for path, field in fields.items():
            if field.get("value_type") == "decimal" and not any(
                    item["id"] == "V-IS-04" for item in validators.get(path, [])):
                _add(validators, path, "V-IS-04", "fail", "currency unit undetermined")


def _percent(value) -> Optional[Decimal]:
    """Rate text such as 13%, 13.0000 or 0.13 → 0.13, without trusting position.

    百分号优先：按量级猜「>1 就是百分数」在 1%（小规模纳税人征收率）与 0.5%
    这类不大于 1 的税率上会把 1% 读成 100%、把 0.5% 读成 50%，进而让
    V-VI-02/V-CV-01 用错误的税率判税额。没有百分号时才退回量级启发式，
    此时 "1" 这类写法本身有歧义，按 1.0 判会让税额核对失败转复核（保守方向）。
    """
    if value in (None, ""):
        return None
    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = Decimal(match.group(1))
    if "%" in text or "％" in text:
        return number / Decimal(100)
    return number / Decimal(100) if number > 1 else number


def _rows(fields: Dict, prefix: str) -> Dict[int, Dict[str, Dict]]:
    rows: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for path, field in fields.items():
        match = re.match(re.escape(prefix) + r"\[(\d+)\]\.(\w+)$", path)
        if match:
            rows[int(match.group(1))][match.group(2)] = field
    return rows


def _add_fail(validators: Dict[str, List[Dict]], paths: Iterable[str], rule_id: str,
              status: str, detail: str) -> None:
    for path in paths:
        _add(validators, path, rule_id, status, detail)


def validate_vat_invoice(doc: Dict, config: Dict,
                         validators: Dict[str, List[Dict]]) -> None:
    fields = doc["fields"]
    tol = Decimal(str(config["validation"].get("tax_tolerance", 0.05)))
    amount = _decimal(fields.get("vi.total_amount"))
    tax = _decimal(fields.get("vi.total_tax"))
    total = _decimal(fields.get("vi.total_with_tax"))
    total_paths = ("vi.total_amount", "vi.total_tax", "vi.total_with_tax")

    if None not in (amount, tax, total):
        expected = amount + tax
        status = "pass" if abs(expected - total) <= tol else "fail"
        _add_fail(validators, total_paths, "V-VI-01", status,
                  f"amount {amount} + tax {tax} = {expected}, 价税合计 {total}")

    item_amounts: List[Decimal] = []
    item_taxes: List[Decimal] = []
    for index, row in sorted(_rows(fields, "vi.items").items()):
        base = f"vi.items[{index}]"
        item_amount = _decimal(row.get("amount"))
        item_tax = _decimal(row.get("tax_amount"))
        rate = _percent(row.get("tax_rate", {}).get("value"))
        price = _decimal(row.get("unit_price"))
        quantity = _decimal(row.get("quantity"))
        if None not in (item_amount, item_tax, rate):
            expected = (item_amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            status = "pass" if abs(expected - item_tax) <= tol else "fail"
            _add_fail(validators, (f"{base}.amount", f"{base}.tax_rate", f"{base}.tax_amount"),
                      "V-VI-02", status,
                      f"amount {item_amount} × {rate} = {expected}, 税额 {item_tax}")
        if None not in (price, quantity, item_amount):
            expected = (price * quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            status = "pass" if abs(expected - item_amount) <= tol else "fail"
            _add_fail(validators, (f"{base}.unit_price", f"{base}.quantity", f"{base}.amount"),
                      "V-VI-03", status,
                      f"单价 {price} × 数量 {quantity} = {expected}, 金额 {item_amount}")
        if item_amount is not None:
            item_amounts.append(item_amount)
        if item_tax is not None:
            item_taxes.append(item_tax)

    if item_amounts and amount is not None:
        summed = sum(item_amounts, Decimal(0))
        status = "pass" if abs(summed - amount) <= tol else "fail"
        _add_fail(validators, ("vi.total_amount",), "V-VI-04", status,
                  f"行金额合计 {summed}, 票面合计 {amount}")
    if item_taxes and tax is not None:
        summed = sum(item_taxes, Decimal(0))
        status = "pass" if abs(summed - tax) <= tol else "fail"
        _add_fail(validators, ("vi.total_tax",), "V-VI-04", status,
                  f"行税额合计 {summed}, 票面税额 {tax}")


def validate_customs_payment(doc: Dict, config: Dict,
                             validators: Dict[str, List[Dict]]) -> None:
    fields = doc["fields"]
    tol = Decimal(str(config["validation"].get("tax_tolerance", 0.05)))
    total = _decimal(fields.get("cv.total_amount"))
    item_taxes: List[Decimal] = []
    for index, row in sorted(_rows(fields, "cv.items").items()):
        base = f"cv.items[{index}]"
        dutiable = _decimal(row.get("dutiable_value"))
        item_tax = _decimal(row.get("tax_amount"))
        rate = _percent(row.get("tax_rate", {}).get("value"))
        if None not in (dutiable, item_tax, rate):
            expected = (dutiable * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            status = "pass" if abs(expected - item_tax) <= tol else "fail"
            _add_fail(validators, (f"{base}.dutiable_value", f"{base}.tax_rate",
                                   f"{base}.tax_amount"),
                      "V-CV-01", status,
                      f"完税价格 {dutiable} × {rate} = {expected}, 税款 {item_tax}")
        if item_tax is not None:
            item_taxes.append(item_tax)
    if item_taxes and total is not None:
        summed = sum(item_taxes, Decimal(0))
        status = "pass" if abs(summed - total) <= tol else "fail"
        _add_fail(validators, ("cv.total_amount",), "V-CV-02", status,
                  f"行税款合计 {summed}, 票面合计 {total}")

    if config["validation"].get("customs_number_match", True):
        payment = normalize_code(fields.get("cv.payment_number", {}).get("value", ""))
        declaration = normalize_code(fields.get("cv.declaration_number", {}).get("value", ""))
        if payment and declaration:
            status = "pass" if payment.startswith(declaration) else "fail"
            _add_fail(validators, ("cv.payment_number", "cv.declaration_number"),
                      "V-CV-03", status,
                      f"缴款书号码 {payment}, 报关单编号 {declaration}")


def cross_document_rules(config: Dict) -> Dict[str, Dict]:
    """SCENARIO §5.4 的规则表；缺段缺条一律不猜——严重度词与是否复核都必须有出处。"""
    rules = (config.get("cross_document") or {}).get("rules") or {}
    return {str(key): dict(value or {}) for key, value in rules.items()}


def _cross_doc_review_ids(config: Dict) -> List[str]:
    """契约里声明「失败就要复核」的规则 id（不论严重度词是 warn 还是 fail）。"""
    return [rule_id for rule_id, entry in cross_document_rules(config).items()
            if entry.get("review")]



# SCENARIO §5.3 V-IS-05 与 §5.4 V-X-04 比的是同一对日期：利润表期末 vs 征信查询日。
# 查询日早于期末＝「报表还没出就被引用」（V-X-04）；两者相差超过 stale_months 个月＝
# 「拿旧报表审今天的主体」（V-IS-05）。阈值、严重度词、降级对象全读配置，代码里不写第二个 18。
STATEMENT_RECENCY_PATHS = ("is.statement_period_end", "cr.query_date")
STATEMENT_RECENCY_KEYS = ("severity", "stale_months", "future_tolerance_days", "degraded_paths")


def statement_recency_options(config: Dict) -> Optional[Dict]:
    """`validation.statement_recency` 缺席就整段沉默；配了就必须一条不缺。"""
    section = (config.get("validation") or {}).get("statement_recency")
    if section is None:
        return None
    missing = [key for key in STATEMENT_RECENCY_KEYS if key not in section]
    if missing:
        raise ValueError("validation.statement_recency 缺 %s：§5.3 V-IS-05 与 §5.4 V-X-04 "
                         "的阈值与降级对象必须有出处" % missing)
    options = dict(section)
    options["stale_months"] = int(section["stale_months"])
    options["future_tolerance_days"] = int(section["future_tolerance_days"])
    options["degraded_paths"] = [str(path) for path in section["degraded_paths"] or []]
    return options


def _month_gap(start: date, end: date) -> int:
    """按整月算 end − start：19 号对 20 号算 0 个月，20 号对 19 号算 −1 个月。"""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months - 1 if end.day < start.day else months


def _statement_recency(docs: List[Dict], config: Dict,
                       validators: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """§5.3 V-IS-05 / §5.4 V-X-04：按主体把利润表与征信报告配成对，再比两份日期。

    归组口径与 V-X-01 相同（名称归一化后相同＝同一主体）。配不上对就不判——「不知道这
    两份是不是同一家」不是「日期对得上」。V-X-04 打在契约点名的两个日期上（读
    `cross_document.rules.V-X-04.paths`），V-IS-05 只降级契约点名的对象（读
    `validation.statement_recency.degraded_paths`）：两边都不许在代码里另挑字段。
    """
    options = statement_recency_options(config)
    if options is None:
        return {"pairs": 0, "future": 0, "stale": 0}
    future_rule = cross_document_rules(config).get("V-X-04") or {}
    future_paths = [str(path) for path in future_rule.get("paths") or []]
    if not future_paths:
        raise ValueError("cross_document.rules.V-X-04.paths 缺失：§5.4 点名要比的那两个字段")
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        name_path = CROSS_DOC_NAME_PATHS.get(doc["doc_type"])
        name = normalize_text(doc["fields"].get(name_path, {}).get("value", "")) if name_path else ""
        if name:
            groups[name].append(doc)
    stats = {"pairs": 0, "future": 0, "stale": 0}
    for members in groups.values():
        statements = [doc for doc in members if doc["doc_type"] == "INCOME_STATEMENT"]
        reports = [doc for doc in members if doc["doc_type"] == "CREDIT_REPORT_ENTERPRISE"]
        for statement in statements:
            period_end = parse_date(statement["fields"].get("is.statement_period_end", {}).get("value"))
            if period_end is None:
                continue
            for report in reports:
                queried = parse_date(report["fields"].get("cr.query_date", {}).get("value"))
                if queried is None:
                    continue
                stats["pairs"] += 1
                ahead = (period_end - queried).days
                if ahead > options["future_tolerance_days"]:
                    stats["future"] += 1
                    detail = f"statement period ends {ahead} day(s) after the credit query"
                    # 一条警告落在两个日期上：从哪一侧看都要能看见「这一对日期对不上」。
                    for owner in (statement, report):
                        for path in future_paths:
                            if path in owner["fields"]:
                                _add(validators[owner["document_id"]], path, "V-X-04",
                                     str(future_rule["severity"]), detail)
                gap = _month_gap(queried, period_end)
                if abs(gap) > options["stale_months"]:
                    stats["stale"] += 1
                    detail = f"statement period and credit query are {abs(gap)} months apart"
                    for path in options["degraded_paths"]:
                        if path in statement["fields"]:
                            _add(validators[statement["document_id"]], path, "V-IS-05",
                                 str(options["severity"]), detail)
    return stats


def validate_cross_document(docs: List[Dict], config: Dict,
                            validators: Dict[str, Dict[str, List[Dict]]]) -> None:
    """§5.4 的 V-X-01/V-X-02 与 §5.3 V-IS-05 / §5.4 V-X-04：按主体归组，判决打在契约点名的字段上。

    归组只用看得见的标识符：名称相同的文档归一组（V-X-02 在这组里比 USCC），USCC 相同的文档
    归一组（V-X-01 在这组里比名称）。批次里只有一个 USCC 时，没有 USCC 的文档（流水、利润表）
    也算这个主体的材料——§2.2「四类文档指向同一借款企业」的机械形态；批次里有多个主体时它们不
    归组，因为没有任何线索能把户名归到某个 USCC 上，硬归会把三家不同公司比成不一致。
    """
    rules = cross_document_rules(config)
    for rule_id in CROSS_DOC_EMITTED:
        if rule_id not in rules:
            raise ValueError("cross_document.rules.%s 缺失：§5.4 的规则必须先在配置里声明，"
                             "不在代码里退回第二个词" % rule_id)
    by_name: Dict[str, List[Tuple[Dict, str]]] = defaultdict(list)
    by_uscc: Dict[str, List[Dict]] = defaultdict(list)
    loose: List[Dict] = []
    for doc in docs:
        name_path = CROSS_DOC_NAME_PATHS.get(doc["doc_type"])
        name = normalize_text(doc["fields"].get(name_path, {}).get("value", "")) if name_path else ""
        if name:
            by_name[name].append((doc, name_path))
        codes = {normalize_code(doc["fields"].get(path, {}).get("value", ""))
                 for path in CROSS_DOC_USCC_PATHS}
        codes.discard("")
        if len(codes) == 1:
            by_uscc[next(iter(codes))].append(doc)
        else:
            loose.append(doc)

    for members in by_name.values():
        usccs = set()
        for doc, _ in members:
            for path in CROSS_DOC_USCC_PATHS:
                code = normalize_code(doc["fields"].get(path, {}).get("value", ""))
                if code:
                    usccs.add(code)
        if len(usccs) > 1:
            # 契约写的是「两者复核」——两者是那两个 USCC 字段，不是被拿来归组的名称字段。
            for doc, _ in members:
                for path in CROSS_DOC_USCC_PATHS:
                    if normalize_code(doc["fields"].get(path, {}).get("value", "")):
                        _add(validators[doc["document_id"]], path, "V-X-02",
                             rules["V-X-02"]["severity"], "same company name but different USCC")

    attach_loose = len(by_uscc) == 1
    for group in by_uscc.values():
        collected = []
        for doc in group + (loose if attach_loose else []):
            name_path = CROSS_DOC_NAME_PATHS.get(doc["doc_type"])
            if not name_path:
                continue
            value = normalize_text(doc["fields"].get(name_path, {}).get("value", ""))
            if value:
                collected.append((doc, name_path, value))
        if len({value for _, _, value in collected}) > 1:
            for doc, path, _ in collected:
                _add(validators[doc["document_id"]], path, "V-X-01",
                     rules["V-X-01"]["severity"], "same USCC but different company name")

    _statement_recency(docs, config, validators)


class Calibrator:
    def __init__(self, config: Dict):
        self.config = config
        self.min_samples = int(config.get("calibration", {}).get("min_samples_per_field", 30))
        self.cap = float(config.get("calibration", {}).get("low_sample_confidence_cap", 0.94))
        self.sample_grouping = config.get("calibration", {}).get("sample_grouping", "field_auto")
        artifact_path = config.get("calibration", {}).get("artifact")
        self.artifact = None
        self.field_samples: Dict[str, int] = {}
        self.group_samples: Dict[str, int] = {}
        if artifact_path:
            path = Path(artifact_path)
            if not path.is_absolute():
                path = ROOT / path
            if path.exists():
                self.artifact = json.loads(path.read_text(encoding="utf-8"))
                self._require_matching_features()
                self.field_samples = self.artifact.get("field_samples", {})
                self.group_samples = self.artifact.get("group_samples", {})
                self.sample_grouping = self.artifact.get("sample_grouping") or self.sample_grouping

    def _require_matching_features(self) -> None:
        """SCENARIO §6.2：校准产物与当前特征向量必须同形同序。

        `predict()` 用 `zip(coef, features)` 点乘，旧产物遇上新配置（多一个风险档或
        文档类型）不会报错，只会静默少算最后几维——置信度就此悄悄变了，而产物里
        只留下一个 0～0.99 的数。产物带 `features` 时逐名比对；不带的老产物无从
        比对，交给 `docpipe.modelaudit` 报 `features_missing`。
        """
        names = self.artifact.get("features")
        if names is None:
            return
        names = [str(item) for item in names]
        expected = feature_names(self.config)
        if names != expected:
            raise ValueError(
                f"校准产物的特征向量与当前配置不一致：产物 {len(names)} 维、当前 "
                f"{len(expected)} 维（顺序按 thresholds.risk_tiers / doc_types 的配置顺序）。"
                f"改过配置就要重跑 python -m docpipe.calibrate，否则系数会按位置错配。")
        coef_length = len(self.artifact.get("coef") or [])
        if coef_length != len(expected):
            raise ValueError(
                f"校准产物 coef 有 {coef_length} 个、特征向量 {len(expected)} 维："
                f"zip() 会静默截断，请重跑 python -m docpipe.calibrate。")

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    def samples_for(self, field: Dict, doc_type: str) -> int:
        if self.sample_grouping == "field_auto":
            field_count = int(self.field_samples.get(normalize_path(field["path"]), 0))
            if field_count >= self.min_samples:
                return field_count
            return int(self.group_samples.get(doc_type, 0))
        if self.sample_grouping == "doc_type":
            return int(self.group_samples.get(doc_type, 0))
        return int(self.field_samples.get(normalize_path(field["path"]), 0))

    def predict(self, features: List[float]) -> float:
        if not self.artifact:
            return 0.0
        coef = self.artifact["coef"]
        intercept = float(self.artifact["intercept"])
        z = intercept + sum(c * x for c, x in zip(coef, features))
        z = max(-30.0, min(30.0, z))
        probability = 1.0 / (1.0 + math.exp(-z))
        iso = self.artifact.get("isotonic") or {}
        xs, ys = iso.get("x") or [], iso.get("y") or []
        if xs and ys:
            if probability <= xs[0]:
                probability = ys[0]
            elif probability >= xs[-1]:
                probability = ys[-1]
            else:
                for index in range(1, len(xs)):
                    if probability <= xs[index]:
                        left, right = xs[index - 1], xs[index]
                        ratio = (probability - left) / (right - left) if right > left else 1.0
                        probability = ys[index - 1] + ratio * (ys[index] - ys[index - 1])
                        break
        return max(0.0, min(0.99, probability))


def feature_names(config: Dict) -> List[str]:
    names = ["ocr_conf", "rule_exact", "format_valid", "table_score",
             "cross_field_score", "grounding_present"]
    names += [f"risk_{risk}" for risk in config["thresholds"]["risk_tiers"]]
    names += [f"doc_{doc_type}" for doc_type in config["doc_types"]]
    return names


def field_features(field: Dict, doc_type: str, config: Dict) -> List[float]:
    signals = field.get("signals") or {}
    values = [
        float(signals.get("ocr_conf") or 0.0),
        float(signals.get("rule_exact") or 0.0),
        float(signals.get("format_valid") or 0.0),
        float(signals.get("table_score") if signals.get("table_score") is not None else 1.0),
        float(signals.get("cross_field_score") if signals.get("cross_field_score") is not None else 0.5),
        float(signals.get("grounding_present") or 0.0),
    ]
    risk = field.get("risk", "")
    values += [1.0 if risk == item else 0.0 for item in config["thresholds"]["risk_tiers"]]
    values += [1.0 if doc_type == item else 0.0 for item in config["doc_types"]]
    return values


def heuristic_confidence(field: Dict, config: Dict, doc_type: Optional[str] = None) -> float:
    calibration = config.get("calibration") or {}
    weights = calibration.get("heuristic_weights") or {}
    signals = field.get("signals") or {}
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        value = signals.get(key)
        if value is None:
            value = 0.5
        total += float(weight) * float(value)
        weight_sum += float(weight)
    base = total / weight_sum if weight_sum else 0.5
    if doc_type in (calibration.get("heuristic_cap_doc_types") or []):
        base = min(base, float(calibration.get("low_sample_confidence_cap", 0.94)))
    return round(max(0.0, min(0.99, base)), 3)


def grounding_tolerance(config: Dict) -> float:
    """SCENARIO §3.1 的页边容差：解析层丢弃与字段层判决必须读同一个数。"""
    return float(((config.get("parsing") or {}).get("grounding") or {}).get("tolerance_pt", 1.0))


def _grounding_ok(field: Dict, page_count: int, pages: Dict[int, Dict],
                  tolerance: float) -> Tuple[bool, Optional[str]]:
    grounding = field.get("grounding")
    if not grounding or grounding.get("page") is None or not grounding.get("bbox"):
        return False, "grounding missing"
    page_no = int(grounding["page"])
    if page_no < 1 or page_no > page_count:
        return False, "page out of range"
    bbox = grounding["bbox"]
    if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
        return False, "degenerate bbox"
    page = pages.get(page_no)
    if page:
        if (bbox[0] < -tolerance or bbox[1] < -tolerance
                or bbox[2] > page["width"] + tolerance
                or bbox[3] > page["height"] + tolerance):
            return False, "bbox outside page"
    return True, None


def _seal_overlap(field: Dict, pages: Dict[int, Dict], config: Dict) -> bool:
    """True when a coloured seal covers part of the value the field was read from.

    Seal suppression makes the OCR input readable again but the ink under a stamp is
    still degraded, so any field sitting under one goes to a human rather than being
    auto-accepted on a score the enhancement produced.
    """
    validation = config.get("validation") or {}
    if not validation.get("review_on_seal_overlap", True):
        return False
    grounding = field.get("grounding") or {}
    page = pages.get(int(grounding.get("page") or 0))
    if not page or not grounding.get("bbox"):
        return False
    regions = ((page.get("preprocess") or {}).get("enhancement") or {}).get("seal_regions") or []
    if not regions:
        return False
    width = float(page.get("width") or 0.0)
    height = float(page.get("height") or 0.0)
    if width <= 0 or height <= 0:
        return False
    left, top, right, bottom = grounding["bbox"]
    area = max(0.0, right - left) * max(0.0, bottom - top)
    if area <= 0:
        return False
    ratio = float(validation.get("seal_overlap_ratio", 0.15))
    for region in regions:
        overlap = (max(0.0, min(right, region[2] * width) - max(left, region[0] * width))
                   * max(0.0, min(bottom, region[3] * height) - max(top, region[1] * height)))
        if overlap / area > ratio:
            return True
    return False


def _low_resolution_hold(raster: Optional[Dict], risk: str, config: Dict) -> bool:
    """Whether a page flagged ``LOW_RESOLUTION`` should hold back this field.

    Graded mode (``parsing.page_quality.risk_min_dpi.enabled``) reads one number per
    risk tier off the measured sampling density: a page that fails the flat
    ``min_effective_dpi`` can still be enough for the fields that need few pixels.
    Tiers above the flat gate go the other way — they review sooner than the page gate
    does. Without a measured dpi on the page the field is held regardless: guessing that
    a page we cannot measure is fine for one risk tier is exactly the silent error the
    gate exists to stop.

    Shipped with the switch off: the dpi x JPEG sweep (``docpipe.hardset sweep`` ->
    ``work/eval/resolution_sweep.json``) found no risk tier that can be exempted below the
    flat gate without adding silent errors, so the default stays all-or-nothing per page.
    """
    options = (config.get("parsing") or {}).get("page_quality") or {}
    tiers = options.get("risk_min_dpi") or {}
    measured = raster.get("effective_dpi") if raster else None
    if not tiers.get("enabled") or measured is None:
        return True
    required = tiers.get(risk or "", tiers.get("default", options.get("min_effective_dpi", 150)))
    return float(measured) < float(required)


def statement_currency(doc: Dict, config: Dict) -> Optional[str]:
    """银行流水的币种代码；读不到就是 None，绝不按人民币兜底。

    §4.4.1 的判据是「第一版只自动接受 CNY，其他币种全部金额复核」。币种是整份
    流水的判决输入，不是币种那一格自己的事：只拦币种格，美元流水的余额、合计、
    交易金额照样自动接受，等于把美元数字按人民币交付。枚举里没有的写法
    （如「美元」「USD」）原样返回，同样不等于 CNY。
    """
    value = ((doc.get("fields") or {}).get("bs.currency") or {}).get("value")
    if value in (None, ""):
        return None
    return normalize_enum(str(value), "currency", config) or str(value).strip()


def decide_field(field: Dict, doc: Dict, parsed: Dict, config: Dict,
                 validators: List[Dict], calibrator: Calibrator) -> Dict:
    reasons: List[str] = []
    page_count = int(parsed.get("page_count") or len(parsed.get("pages", [])))
    pages = {page["page"]: page for page in parsed.get("pages", [])}
    grounded, grounding_detail = _grounding_ok(field, page_count, pages,
                                         grounding_tolerance(config))
    signals = field.setdefault("signals", {})
    signals["grounding_present"] = 1 if grounded else 0

    has_fail = any(item["status"] == "fail" for item in validators)
    has_warn = any(item["status"] == "warn" for item in validators)
    id_fail = any(item["status"] == "fail" and item["id"].startswith("V-ID") for item in validators)
    cross_fail = any(item["status"] == "fail" and item["id"] in CROSS_FIELD_IDS for item in validators)
    cross_warn = any(item["status"] == "warn" and item["id"] in CROSS_FIELD_IDS for item in validators)
    # §5.4 那句「warn；不一致的字段全部复核」：严重度词照契约，但契约要求复核的跨文档规则
    # 不能因为词写成 warn 就自动放行。
    cross_required = any(item["id"] in _cross_doc_review_ids(config)
                         and item["status"] in ("warn", "fail") for item in validators)
    signals["format_valid"] = 0.0 if id_fail else 1.0
    signals["cross_field_score"] = 0.0 if (cross_fail or cross_required) \
        else (0.3 if cross_warn else 1.0)

    value = field.get("value")
    required = bool(field.get("required"))
    if doc["doc_type"] == "UNKNOWN":
        reasons.append("UNKNOWN_DOC_TYPE")
    quality = parsed.get("quality") or {}
    min_page_quality = float((config.get("parsing") or {}).get("min_page_quality", 0.5))
    if doc["doc_type"] != "UNKNOWN" \
            and float(quality.get("parse_score") or 0.0) < min_page_quality:
        reasons.append("PARSE_LOW_QUALITY")
    grounding_page = pages.get(int((field.get("grounding") or {}).get("page") or 0))
    if grounding_page and "LOW_RESOLUTION" in (grounding_page.get("quality_flags") or []) \
            and config["validation"].get("review_on_low_resolution", True) \
            and _low_resolution_hold(grounding_page.get("raster"), field.get("risk"), config):
        reasons.append("LOW_RESOLUTION")
    if not grounded:
        reasons.append("GROUNDING_MISSING")
    if required and value in (None, ""):
        reasons.append("REQUIRED_EMPTY")
    if float(signals.get("ocr_conf") or 0.0) < float(config["validation"].get("low_ocr_threshold", 0.6)):
        reasons.append("LOW_OCR")
    if id_fail:
        # §4.6「未知则复核」与「格式不合法」要人做的动作不同：前者是「确认原文
        # 确实这么写、再判断这条标志影响不影响准入」，后者是「按原文逐字重录」。
        # 所以 V-ID-10 单独一个码，不并进 FORMAT_FAIL。
        format_ids = [item["id"] for item in validators
                      if item["status"] == "fail" and item["id"].startswith("V-ID")]
        if any(rule_id != "V-ID-10" for rule_id in format_ids):
            reasons.append("FORMAT_FAIL")
        if "V-ID-10" in format_ids:
            reasons.append("ENUM_UNKNOWN")
    if cross_fail or cross_required:
        if any(item["id"] == "V-BS-01" and item["detail"] == "chain failure propagation" for item in validators):
            reasons.append("CHAIN_UNRELIABLE")
        reasons.append("CROSS_FIELD_FAIL")
    meta = field.get("meta") or {}
    if meta.get("sign_source") == "direction" \
            and config["validation"].get("review_on_inferred_sign", True):
        reasons.append("SIGN_INFERRED")
    if meta.get("unit_unknown"):
        reasons.append("AMBIGUOUS_UNIT")
    if meta.get("parse_error") == "date":
        reasons.append("AMBIGUOUS_DATE")
    if doc["doc_type"] == "BANK_STATEMENT":
        # 有整份字段表才有文档级币种可判；只给单格字段的对象（单测夹具）没有
        # bs.currency，这一条不该在一个看不见币种的对象上猜结果。
        fields_table = doc.get("fields") or {}
        currency = statement_currency(doc, config)
        if field.get("path") == "bs.currency":
            if value not in (None, "") and currency != "CNY":
                reasons.append("NON_CNY_CURRENCY")
        elif field.get("risk") == "monetary" and "bs.currency" in fields_table \
                and currency != "CNY":
            reasons.append("NON_CNY_CURRENCY")
    min_table_score = float((config.get("validation") or {}).get("min_table_score", 0.5))
    if field.get("extractor_id", "").startswith(("table_cell", "table_row")) \
            and float(signals.get("table_score") or 0.0) < min_table_score:
        reasons.append("TEMPLATE_MISMATCH")
    if _seal_overlap(field, pages, config):
        reasons.append("SEAL_OVERLAP")
    if meta.get("table_anomaly") and config["validation"].get("review_on_table_anomaly", True):
        reasons.append("TABLE_ANOMALY")
    if meta.get("line_fallback") and config["validation"].get("review_on_line_fallback", True):
        reasons.append("LINE_FALLBACK")
    if meta.get("matrix_fallback") and config["validation"].get("review_on_matrix_fallback", True):
        reasons.append("MATRIX_FALLBACK")
    if meta.get("local_recrop") and config["validation"].get("review_on_local_recrop", True):
        reasons.append("LOCAL_RECROP")
    if meta.get("qr_conflict") and config["validation"].get("review_on_qr_conflict", True):
        reasons.append("QR_CONFLICT")
    if meta.get("extractor_conflict") and \
            config["validation"].get("review_on_extractor_conflict", True):
        reasons.append("EXTRACTOR_CONFLICT")
    if meta.get("raw_unverified") and \
            config["validation"].get("review_on_raw_unverified", True):
        reasons.append("RAW_UNVERIFIED")
    if meta.get("placeholder") and \
            config["validation"].get("review_on_placeholder", True):
        reasons.append("PLACEHOLDER_EMPTY")

    if calibrator.ready:
        confidence = calibrator.predict(field_features(field, doc["doc_type"], config))
        samples = calibrator.samples_for(field, doc["doc_type"])
        basis = "calibrated"
        if samples < calibrator.min_samples:
            confidence = min(confidence, calibrator.cap)
            basis = "calibrated_low_samples"
    else:
        confidence = heuristic_confidence(field, config, doc["doc_type"])
        samples = 0
        basis = "heuristic"
    confidence = round(max(0.0, min(0.99, confidence)), 3)
    field["confidence"] = confidence
    # The basis travels with the number: 0.94 from a calibrated model and 0.94 from a
    # cold-start heuristic must not look the same downstream.
    field["confidence_basis"] = basis
    field["calibration_samples"] = samples

    threshold = risk_threshold({
        "path": normalize_path(field["path"]),
        "risk": field.get("risk") or "optional_low_risk"}, config)
    global_min = float(config["thresholds"].get("global_min_confidence", 0.90))
    hard_reasons = set(config.get("hard_abstain_reasons") or [])
    hard_hit = any(reason in hard_reasons for reason in reasons)
    if hard_hit or has_fail:
        decision = "review"
    elif confidence < global_min or confidence < threshold:
        decision = "review"
        reasons.append("LOW_CONFIDENCE")
    elif not grounded:
        decision = "review"
    else:
        decision = "accept"
    if decision == "accept":
        reasons = []
    field["decision"] = decision
    field["validators"] = validators
    field["review_reasons"] = sorted(set(reasons))
    return field



def document_refusal(doc: Dict, config: Dict) -> Optional[Dict]:
    """§6.4 的 UNKNOWN_DOC_TYPE 在字段层没有落脚点：类型认不出时连 schema 都选不出来，
    一个字段都不会被抽出，于是「强制复核」退化成一份零字段、零原因的空文档交出去。
    拒答因此挂在文档上；文案仍走 `validation.review_actions`，不另写一份话。"""
    if str(doc.get("doc_type") or "") != "UNKNOWN":
        return None
    actions = (config.get("validation") or {}).get("review_actions") or {}
    return {
        "code": "UNKNOWN_DOC_TYPE",
        "field_count": len(doc.get("fields") or {}),
        "doc_type_confidence": round(float(doc.get("doc_type_confidence") or 0.0), 3),
        "action": str(actions.get("UNKNOWN_DOC_TYPE") or ""),
    }

def assess(docs: List[Dict], parsed_by_id: Dict[str, Dict], config: Optional[Dict] = None,
           cross_document: bool = True) -> List[Dict]:
    config = config or load_config()
    calibrator = Calibrator(config)
    validators: Dict[str, Dict[str, List[Dict]]] = {doc["document_id"]: {} for doc in docs}
    for doc in docs:
        parsed = parsed_by_id[doc["document_id"]]
        if doc["doc_type"] != "UNKNOWN":
            validate_document_fields(doc, parsed, config, validators[doc["document_id"]])
        if doc["doc_type"] == "BANK_STATEMENT":
            # 链条诊断（几行接不上、错到什么比例、被哪条证据判坏）过去算完就丢，
            # 复核员只看到一个 CHAIN_UNRELIABLE，不知道是 50 行里错 2 行还是错 30 行。
            doc["chain_stats"] = validate_bank_statement(
                doc, parsed, config, validators[doc["document_id"]])
        if doc["doc_type"] == "INCOME_STATEMENT":
            validate_income_statement(doc, config, validators[doc["document_id"]])
        if doc["doc_type"] == "VAT_INVOICE":
            validate_vat_invoice(doc, config, validators[doc["document_id"]])
        if doc["doc_type"] == "CUSTOMS_VAT_PAYMENT":
            validate_customs_payment(doc, config, validators[doc["document_id"]])
    if cross_document and len(docs) > 1:
        validate_cross_document(docs, config, validators)

    for doc in docs:
        parsed = parsed_by_id[doc["document_id"]]
        for path, field in doc["fields"].items():
            field_validators = validators[doc["document_id"]].get(path, [])
            decide_field(field, doc, parsed, config, field_validators, calibrator)
    for doc in docs:
        refusal = document_refusal(doc, config)
        if refusal:
            doc["document_refusal"] = refusal
    return docs
