"""Layout parsing: text layer, OCR, visual lines and table cells with page/bbox.

Output contract (also written to work/parsed/<id>.json):
  document: id, doc_type, page_count, engine, pages
  page: page, width, height, engine, quality, lines, tables, grounding_dropped[]
  line: text, bbox, confidence, parts[]
  table: name, page, bbox, rows[] with canonical cells and cell bboxes
  grounding_dropped: every block/cell thrown away for lacking a page bbox (SCENARIO §7.1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf as fitz

from .config import ROOT, all_field_labels, doc_types, load_config, resolve_path, table_spec
from . import ocr, preprocess
from .textutil import normalize_text

PARSER_VERSION = "0.6.1"

# 行被丢弃的每一种理由（SCENARIO §5.2 V-BS-07）。前四种能证明「这不是一笔交易」，
# 后三种不能：label_unknown 是非流水表的结构性过滤，unaccounted/narrow 是「说不出
# 是什么」的行——它们必须让该页交易字段全部复核。
ROW_SKIP_REASONS = ("blank", "header", "footer", "missing_required", "label_unknown",
                    "unaccounted", "narrow")
_LAYOUT_STATE: Dict[str, Optional[bool]] = {"ready": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _union(boxes: Iterable[List[float]]) -> List[float]:
    boxes = [box for box in boxes if box and len(box) == 4]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(box[0] for box in boxes), 2),
        round(min(box[1] for box in boxes), 2),
        round(max(box[2] for box in boxes), 2),
        round(max(box[3] for box in boxes), 2),
    ]


def _overlap_ratio(box: List[float], other: List[float]) -> float:
    """Share of ``box`` covered by ``other`` (0..1), used for block membership."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    overlap = (max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
               * max(0.0, min(box[3], other[3]) - max(box[1], other[1])))
    return overlap / area


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or "")))


# ---------------------------------------------------------------------------
# engine selection (SCENARIO §13: parsing.pdf_text_engine / parsing.table_engine)
# ---------------------------------------------------------------------------

TEXT_ENGINES = {"pymupdf"}
TABLE_ENGINES = {"pdfplumber"}


def text_engine(config: Dict) -> str:
    """文本层引擎名。未注册的名字报错并列出可选值，不退回默认（同 OCR 注册表）。"""
    name = str(config["parsing"].get("pdf_text_engine", "pymupdf"))
    if name not in TEXT_ENGINES:
        raise ValueError(
            f"未注册的文本层引擎 {name!r}（parsing.pdf_text_engine）：可选 {sorted(TEXT_ENGINES)}")
    return name


def table_engine(config: Dict) -> str:
    name = str(config["parsing"].get("table_engine", "pdfplumber"))
    if name not in TABLE_ENGINES:
        raise ValueError(
            f"未注册的表格引擎 {name!r}（parsing.table_engine）：可选 {sorted(TABLE_ENGINES)}")
    return name


# ---------------------------------------------------------------------------
# text layer
# ---------------------------------------------------------------------------

def text_layer_parts(page) -> List[Dict]:
    parts: List[Dict] = []
    raw = page.get_text("rawdict")
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.get("chars", [])
                if not chars:
                    continue
                text = "".join(ch.get("c", "") for ch in chars)
                if not text.strip():
                    continue
                boxes = [ch["bbox"] for ch in chars if ch.get("c", "").strip()]
                bbox = _union(boxes) if boxes else list(span.get("bbox", [0, 0, 0, 0]))
                parts.append({
                    "text": text,
                    "bbox": [round(v, 2) for v in bbox],
                    "confidence": 0.99,
                    "chars": [{"c": ch.get("c", ""), "bbox": [round(v, 2) for v in ch["bbox"]]}
                              for ch in chars],
                    "engine": "pymupdf_text",
                })
    return parts


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_parts(page, dpi: int, config: Optional[Dict] = None) -> Tuple[List[Dict], float, Dict]:
    """OCR one page. The third value records what the enhancement pass changed."""
    config = config or load_config()
    engine = ocr.get_engine(config)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image, enhancement = preprocess.ocr_input(image, config)
    result = engine.recognize(image)
    scale = 72.0 / dpi
    parts: List[Dict] = []
    scores: List[float] = []
    for item in result or []:
        box, text, score = item[0], str(item[1]), float(item[2])
        if not text.strip():
            continue
        xs = [float(point[0]) * scale for point in box]
        ys = [float(point[1]) * scale for point in box]
        bbox = [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]
        parts.append({
            "text": text,
            "bbox": bbox,
            "confidence": round(score, 4),
            "chars": [],
            "engine": engine.name,
        })
        scores.append(score)
    mean_score = round(float(statistics.mean(scores)), 4) if scores else 0.0
    return parts, mean_score, enhancement


def scan_codes(image: np.ndarray, scale: float,
               options: Optional[Dict] = None) -> List[Dict]:
    """Decode QR codes offline. Returns text plus a page-coordinate bbox.

    The invoice QR is a machine-readable copy of the printed number, date and
    total. Decoding it locally gives the extraction layer an independent check
    without any network call; a failed decode is simply an empty list.
    """
    options = options or {}
    if image is None or image.size == 0:
        return []
    limit = max(1, int(options.get("max_codes", 3)))
    detector = cv2.QRCodeDetector()
    found: List[Tuple[str, object]] = []
    try:
        text, points, _ = detector.detectAndDecode(image)
        if text and points is not None:
            found.append((text, points))
    except cv2.error:
        pass
    if not found:
        try:
            ok, texts, points, _ = detector.detectAndDecodeMulti(image)
            if ok and texts and points is not None:
                for text, corners in zip(texts, points):
                    if text:
                        found.append((text, corners))
        except (cv2.error, ValueError):
            pass
    codes: List[Dict] = []
    for text, points in found[:limit]:
        corners = np.asarray(points, dtype=float).reshape(-1, 2)
        xs = corners[:, 0] * scale
        ys = corners[:, 1] * scale
        bbox = [round(float(xs.min()), 2), round(float(ys.min()), 2),
                round(float(xs.max()), 2), round(float(ys.max()), 2)]
        codes.append({"text": str(text), "bbox": bbox, "engine": "opencv_qr"})
    return codes


# ---------------------------------------------------------------------------
# OCR gap recovery
# ---------------------------------------------------------------------------

def _recrop_options(config: Optional[Dict]) -> Dict:
    return ((config or {}).get("parsing") or {}).get("ocr_recrop") or {}


def gap_recrop_rect(line: Dict, page_width: float, page_height: float,
                    options: Dict) -> Optional[List[float]]:
    """Right-hand blank of a label-only line, or None when nothing can be recovered.

    The detector drops isolated glyphs on a full page but reads them at crop scale,
    so a label ending at "：" with an empty right side is the one place where a
    missing token is both visible and unambiguous. A line that already carries a
    value never qualifies: re-reading it could append a neighbour's text.
    """
    text = (line.get("text") or "").rstrip()
    separators = tuple(options.get("separator_chars") or ["：", ":"])
    if not text or not text.endswith(separators):
        return None
    bbox = line.get("bbox") or []
    if len(bbox) != 4:
        return None
    gap_pad = float(options.get("gap_pad", 2.0))
    max_gap = float(options.get("max_gap", 160.0))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    vertical_pad = height * float(options.get("vertical_pad", 0.25))
    left = float(bbox[2]) + gap_pad
    right = min(float(page_width) - 1.0, left + max_gap)
    if right - left < float(options.get("min_width", 4.0)):
        return None
    top = max(0.0, float(bbox[1]) - vertical_pad)
    bottom = min(float(page_height), float(bbox[3]) + vertical_pad)
    if bottom - top < 2.0:
        return None
    return [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)]


def accepts_recrop_text(text: str, score: float, options: Dict) -> bool:
    """A recovered token has to look like one short value, not a fragment of a line."""
    compact = re.sub(r"\s+", "", normalize_text(text or ""))
    if not compact or len(compact) > int(options.get("max_chars", 24)):
        return False
    if any(ch in compact for ch in options.get("separator_chars") or ["：", ":"]):
        return False
    if not _is_value_text(compact):
        return False
    return float(score) >= float(options.get("min_confidence", 0.55))


def _crop_ocr(page, rect: List[float], dpi: int, config: Dict) -> List[Tuple[List[float], str, float]]:
    pix = page.get_pixmap(dpi=dpi, clip=fitz.Rect(*rect), alpha=False)
    if pix.width < 4 or pix.height < 4:
        return []
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image, _ = preprocess.ocr_input(image, config)
    result = ocr.get_engine(config).recognize(image)
    parts: List[Tuple[List[float], str, float]] = []
    for item in result or []:
        text = str(item[1])
        if not text.strip():
            continue
        xs = [float(point[0]) for point in item[0]]
        ys = [float(point[1]) for point in item[0]]
        parts.append(([min(xs), min(ys), max(xs), max(ys)], text, float(item[2])))
    parts.sort(key=lambda item: item[0][0])
    return parts


def recover_gap_parts(page, lines: List[Dict], config: Dict,
                      crop_ocr=None) -> List[Dict]:
    """Re-OCR the blank right of label-only lines; return extra parts with page bboxes."""
    options = _recrop_options(config)
    if not options.get("enabled") or not lines:
        return []
    dpi = int(options.get("dpi", 400))
    scale = 72.0 / dpi
    max_parts = int(options.get("max_parts", 1))
    max_per_page = int(options.get("max_per_page", 8))
    crop = crop_ocr or _crop_ocr
    recovered: List[Dict] = []
    for line in lines:
        if len(recovered) >= max_per_page:
            break
        rect = gap_recrop_rect(line, page.rect.width, page.rect.height, options)
        if rect is None:
            continue
        try:
            candidates = crop(page, rect, dpi, config)
        except Exception:
            continue
        if not candidates or len(candidates) > max_parts:
            continue
        box, text, score = candidates[0]
        if not accepts_recrop_text(text, score, options):
            continue
        recovered.append({
            "text": text.strip(),
            "bbox": [round(rect[0] + float(box[0]) * scale, 2),
                     round(rect[1] + float(box[1]) * scale, 2),
                     round(rect[0] + float(box[2]) * scale, 2),
                     round(rect[1] + float(box[3]) * scale, 2)],
            "confidence": round(float(score), 4),
            "chars": [],
            "engine": f"{ocr.get_engine(config).name}_recrop",
            "recovered": True,
        })
    return recovered


# ---------------------------------------------------------------------------
# line grouping
# ---------------------------------------------------------------------------

def group_parts(parts: List[Dict]) -> List[Dict]:
    if not parts:
        return []
    heights = [max(1.0, p["bbox"][3] - p["bbox"][1]) for p in parts]
    tolerance = max(4.0, statistics.median(heights) * 0.55)
    ordered = sorted(parts, key=lambda p: ((p["bbox"][1] + p["bbox"][3]) / 2, p["bbox"][0]))
    clusters: List[List[Dict]] = []
    cluster_centers: List[float] = []
    for part in ordered:
        center = (part["bbox"][1] + part["bbox"][3]) / 2
        if clusters and abs(center - cluster_centers[-1]) <= tolerance:
            clusters[-1].append(part)
        else:
            clusters.append([part])
            cluster_centers.append(center)
    lines: List[Dict] = []
    for cluster in clusters:
        cluster = sorted(cluster, key=lambda p: p["bbox"][0])
        text = " ".join(p["text"].strip() for p in cluster if p["text"].strip())
        bbox = _union([p["bbox"] for p in cluster])
        confidence = round(float(statistics.mean([p["confidence"] for p in cluster])), 4)
        lines.append({
            "text": text,
            "bbox": bbox,
            "confidence": confidence,
            "parts": cluster,
            "engine": cluster[0].get("engine", "unknown"),
            "recovered": any(part.get("recovered") for part in cluster),
        })
    lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    return lines


def compact_line_map(line: Dict) -> Tuple[str, List[Tuple[str, List[float]]]]:
    """Return compact text and per-character bboxes, skipping whitespace."""
    chars: List[Tuple[str, List[float]]] = []
    for part in line.get("parts", []):
        raw_chars = part.get("chars") or []
        text = part.get("text", "")
        if raw_chars and len(raw_chars) == len(text):
            for ch in raw_chars:
                char = ch.get("c", "")
                if char.strip():
                    chars.append((char, ch["bbox"]))
        else:
            clean = [(idx, ch) for idx, ch in enumerate(text) if ch.strip()]
            for idx, ch in clean:
                box = part["bbox"]
                char_width = (box[2] - box[0]) / max(1, len(text))
                left = box[0] + idx * char_width
                chars.append((ch, [left, box[1], left + char_width, box[3]]))
    return "".join(ch for ch, _ in chars), chars


def _bbox_for_range(chars: List[Tuple[str, List[float]]], start: int, end: int) -> List[float]:
    boxes = [box for _, box in chars[start:end]]
    return _union(boxes)


# ---------------------------------------------------------------------------
# field-like extraction helpers (used both for parsing preview and extraction)
# ---------------------------------------------------------------------------

def _label_variants(label: str, aliases: Optional[List[str]] = None) -> List[str]:
    variants = [label] + list(aliases or [])
    return sorted({_compact(item) for item in variants if _compact(item)}, key=len, reverse=True)


def locate_label_value(lines: List[Dict], label: str, aliases: Optional[List[str]] = None,
                       stop_labels: Optional[List[str]] = None) -> Optional[Dict]:
    """Find a label-anchored value, returning its line, character span and bbox."""
    variants = _label_variants(label, aliases)
    stop_variants = sorted({_compact(item) for item in (stop_labels or []) if _compact(item)},
                           key=len, reverse=True)
    for line_index, line in enumerate(lines):
        compact, chars = compact_line_map(line)
        if not compact:
            continue
        best_match = None
        for variant in variants:
            pos = compact.find(variant)
            while pos != -1:
                end = pos + len(variant)
                # A field label is reliable when it is followed by a colon/end.
                # Longest-first ordering prevents "名称" from matching inside "企业名称".
                if end == len(compact) or compact[end] in ":：":
                    best_match = (variant, pos, end)
                    break
                pos = compact.find(variant, pos + 1)
            if best_match:
                break
        if not best_match:
            continue
        variant, start, end = best_match
        cursor = end
        while cursor < len(compact) and compact[cursor] in ":：":
            cursor += 1
        value_end = len(compact)
        for stop in stop_variants:
            if stop == variant:
                continue
            pos = compact.find(stop, cursor)
            if pos != -1 and pos < value_end:
                value_end = pos
        value = compact[cursor:value_end].strip(":：,，;； ")
        if not value:
            # value may be on the next visual line (rare, e.g. wrapped labels)
            if line_index + 1 < len(lines):
                next_line = lines[line_index + 1]
                next_compact, next_chars = compact_line_map(next_line)
                blocked = any(next_compact.startswith(stop) for stop in stop_variants
                              if stop != variant)
                if (next_line["bbox"][1] - line["bbox"][3] < 18 and next_compact
                        and not blocked):
                    return {
                        "value": next_compact,
                        "bbox": next_line["bbox"],
                        "confidence": next_line["confidence"],
                        "raw_text": next_line["text"],
                        "line_index": line_index + 1,
                        "char_span": [0, len(next_compact)],
                        "chars": next_chars,
                    }
            continue
        bbox = _bbox_for_range(chars, cursor, value_end)
        return {
            "value": value,
            "bbox": bbox,
            "confidence": line["confidence"],
            "raw_text": line["text"],
            "line_index": line_index,
            "char_span": [cursor, value_end],
            "chars": chars,
        }
    return None


def extract_label_value(lines: List[Dict], label: str, aliases: Optional[List[str]] = None,
                        stop_labels: Optional[List[str]] = None) -> Optional[Dict]:
    located = locate_label_value(lines, label, aliases, stop_labels)
    if not located:
        return None
    return {key: located[key] for key in
            ("value", "bbox", "confidence", "raw_text", "line_index", "char_span")}


def _part_text(part: Dict) -> str:
    return _compact(part.get("text", ""))


def _parts_text(parts: List[Dict]) -> str:
    return "".join(_part_text(part) for part in parts)


def _is_value_text(text: str) -> bool:
    """True when a compact string has something a human can read as a value."""
    return bool(re.search(r"[0-9A-Za-z\u4e00-\u9fff]", text or ""))


def _label_candidate_groups(lines: List[Dict], variants: List[str],
                            max_label_parts: int, join_gap: float,
                            require_delimiter: bool) -> List[Dict]:
    """Locate labels that may be split across adjacent OCR boxes.

    Real forms often render one label in two boxes (帐 | 号, 名 | 称) or in a
    single box followed by a value box. The text-layer helper only sees a
    flattened line, so it cannot tell where one table cell ends. Here each label
    match carries the boxes it was assembled from, which is also what makes a
    value to the right addressable without guessing.
    """
    matches: List[Dict] = []
    for line_index, line in enumerate(lines):
        parts = sorted(line.get("parts") or [], key=lambda item: item["bbox"][0])
        if not parts:
            continue
        used_starts = set()
        for start in range(len(parts)):
            if start in used_starts:
                continue
            for end in range(start, min(len(parts), start + max_label_parts)):
                if end > start:
                    gap = parts[end]["bbox"][0] - parts[end - 1]["bbox"][2]
                    if gap > join_gap:
                        break
                span = parts[start:end + 1]
                text = _parts_text(span)
                if not text:
                    continue
                hit = None
                for variant in variants:
                    pos = text.find(variant)
                    while pos != -1:
                        before_ok = pos == 0 or text[pos - 1] in "/／·:：,，;；(（[【"
                        after = pos + len(variant)
                        after_ok = (after == len(text)
                                    or text[after] in ":：,，;；/／")
                        # Without a delimiter we still require a word boundary
                        # before the label; otherwise 货物名称 would match 名称.
                        if before_ok and (after_ok or not require_delimiter):
                            hit = (variant, pos, after)
                            break
                        pos = text.find(variant, pos + 1)
                    if hit:
                        break
                if not hit:
                    continue
                variant, label_start, label_end = hit
                used_starts.add(start)
                matches.append({
                    "line_index": line_index,
                    "line": line,
                    "parts": span,
                    "variant": variant,
                    "label_start": label_start,
                    "label_end": label_end,
                    "text": text,
                    "bbox": _union([part["bbox"] for part in span]),
                })
                break
    matches.sort(key=lambda item: (item["line_index"], item["bbox"][0]))
    return matches


def _value_in_label_box(match: Dict, stop_variants: List[str],
                        strip_prefix: List[str]) -> Optional[Dict]:
    """Value that starts in the same OCR box (or box group) as the label."""
    text = match["text"]
    cursor = match["label_end"]
    while cursor < len(text) and text[cursor] in ":：":
        cursor += 1
    for prefix in strip_prefix:
        compact = _compact(prefix)
        if compact and text[cursor:].startswith(compact):
            cursor += len(compact)
            while cursor < len(text) and text[cursor] in " .:：":
                cursor += 1
            break
    value_end = len(text)
    for stop in stop_variants:
        if stop == match["variant"]:
            continue
        pos = text.find(stop, cursor)
        if pos != -1:
            value_end = min(value_end, pos)
    value = text[cursor:value_end].strip(":：,，;；()（）[]【】 ￥¥")
    if not _is_value_text(value):
        return None
    chars = compact_line_map({"parts": match["parts"]})[1]
    return {
        "value": value,
        "bbox": _bbox_for_range(chars, cursor, value_end),
        "confidence": round(float(statistics.mean(
            [part["confidence"] for part in match["parts"]])), 4),
        "raw_text": " ".join(part.get("text", "") for part in match["parts"]),
        "line_index": match["line_index"],
        "char_span": [cursor, value_end],
    }


def _right_value(lines: List[Dict], match: Dict, stop_variants: List[str],
                 strip_prefix: List[str], options: Dict) -> Optional[Dict]:
    """Nearest readable box to the right on the same visual row.

    This is what connects a label in one table cell to a value in the next cell
    when OCR puts them on different visual lines (e.g. a multi-row 缴款单位
    header next to 帐号). Vertical overlap, not line identity, is the row test.
    """
    label_box = match["bbox"]
    label_parts = {id(part) for part in match["parts"]}
    x_slack = float(options.get("x_slack", 4))
    row_overlap = float(options.get("row_overlap", 0.3))
    center_slack = float(options.get("center_slack", 0.6))
    max_parts = int(options.get("max_value_parts", 1))
    join_gap = float(options.get("value_join_gap", 12))
    label_height = max(1.0, label_box[3] - label_box[1])
    candidates: List[Dict] = []
    for line_index, line in enumerate(lines):
        for part in sorted(line.get("parts") or [], key=lambda item: item["bbox"][0]):
            if id(part) in label_parts:
                continue
            box = part["bbox"]
            if box[2] <= label_box[0] + x_slack or box[0] < label_box[2] - x_slack:
                continue
            height = max(1.0, box[3] - box[1])
            overlap = max(0.0, min(box[3], label_box[3]) - max(box[1], label_box[1]))
            ratio = overlap / min(height, label_height)
            center_diff = abs((box[1] + box[3]) / 2 - (label_box[1] + label_box[3]) / 2)
            if ratio < row_overlap and center_diff > center_slack * max(height, label_height):
                continue
            candidates.append({
                "line_index": line_index,
                "part": part,
                "ratio": ratio,
                "center_diff": center_diff,
                "x_gap": max(0.0, box[0] - label_box[2]),
            })
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["x_gap"], -item["ratio"], item["center_diff"]))
    value_index = int(options.get("value_index", 0))
    if value_index > 0:
        # A totals row can carry several values with no label of their own
        # (合计 | 金额 | 税额); the configured index picks the n-th one in
        # reading order on the same visual row.
        row = sorted(candidates, key=lambda item: item["part"]["bbox"][0])
        seen: List[Dict] = []
        last_right = None
        for item in row:
            if last_right is not None and item["part"]["bbox"][0] <= last_right:
                continue
            seen.append(item)
            last_right = item["part"]["bbox"][2]
        if value_index >= len(seen):
            return None
        first = seen[value_index]
    else:
        nearest_gap = candidates[0]["x_gap"]
        window = [item for item in candidates if item["x_gap"] <= nearest_gap + 2]
        first = min(window, key=lambda item: (item["center_diff"], -item["ratio"]))
    first_text = _part_text(first["part"])
    if any(first_text.startswith(stop) or first_text == stop for stop in stop_variants
           if stop != match["variant"]):
        return None
    chosen = [first]
    if max_parts > 1:
        ordered = sorted(candidates, key=lambda item: item["part"]["bbox"][0])
        for item in ordered:
            if item is first or item["part"]["bbox"][0] < first["part"]["bbox"][2]:
                continue
            if len(chosen) >= max_parts:
                break
            if item["part"]["bbox"][0] - chosen[-1]["part"]["bbox"][2] > join_gap:
                break
            if abs(item["center_diff"] - first["center_diff"]) > 4:
                break
            chosen.append(item)
    chosen.sort(key=lambda item: item["part"]["bbox"][0])
    if options.get("preserve_spaces"):
        text = " ".join(normalize_text(item["part"].get("text", "")) for item in chosen).strip()
    else:
        text = _parts_text([item["part"] for item in chosen])
    for prefix in strip_prefix:
        compact = _compact(prefix)
        if compact and text.startswith(compact):
            text = text[len(compact):].lstrip(" .:：")
            break
    value_end = len(text)
    for stop in stop_variants:
        if stop == match["variant"]:
            continue
        pos = text.find(stop)
        if pos != -1:
            value_end = min(value_end, pos)
    text = text[:value_end].strip(":：,，;；()（）[]【】 ￥¥")
    if not _is_value_text(text):
        return None
    pseudo_parts = [dict(item["part"], text=item["part"].get("text", "")) for item in chosen]
    if sum(len(_part_text(part)) for part in pseudo_parts) != len(text):
        # Prefix/stop trimming changed the text; keep the box but map characters
        # onto the whole value box so the highlight still covers the right cell.
        chars = []
        box = _union([item["part"]["bbox"] for item in chosen])
        width = (box[2] - box[0]) / max(1, len(text))
        chars = [(char, [box[0] + index * width, box[1],
                         box[0] + (index + 1) * width, box[3]])
                 for index, char in enumerate(text)]
    else:
        chars = compact_line_map({"parts": pseudo_parts})[1][:len(text)]
    return {
        "value": text,
        "bbox": _union([item["part"]["bbox"] for item in chosen]),
        "confidence": round(min(float(item["part"]["confidence"]) for item in chosen), 4),
        "raw_text": " ".join(item["part"].get("text", "") for item in chosen),
        "line_index": chosen[0]["line_index"],
        "char_span": [0, len(text)],
        "chars": chars,
    }


def locate_label_region(lines: List[Dict], label: str, aliases: Optional[List[str]] = None,
                        stop_labels: Optional[List[str]] = None, occurrence: int = 1,
                        strip_prefix: Optional[List[str]] = None,
                        require_delimiter: bool = False,
                        max_label_parts: int = 3,
                        label_join_gap: float = 40.0,
                        **options) -> Optional[Dict]:
    """Label-anchored value for table-like forms where OCR boxes are cells.

    Unlike :func:`locate_label_value`, the label may be split across adjacent
    boxes and the value may live in the box to the right, possibly on a
    different visual line of the same table row. ``occurrence`` selects the
    n-th label in reading order (1-based; -1 means the last), which is how
    buyer/seller repeated labels are told apart.
    """
    variants = _label_variants(label, aliases)
    stop_variants = sorted({_compact(item) for item in (stop_labels or []) if _compact(item)},
                           key=len, reverse=True)
    matches = _label_candidate_groups(lines, variants, max_label_parts, label_join_gap,
                                      require_delimiter)
    if not matches:
        return None
    if occurrence == -1:
        match = matches[-1]
    elif 1 <= occurrence <= len(matches):
        match = matches[occurrence - 1]
    else:
        return None
    located = _value_in_label_box(match, stop_variants, strip_prefix or [])
    if located is None:
        located = _right_value(lines, match, stop_variants, strip_prefix or [], options)
    if located is None and options.get("allow_below"):
        located = _below_value(lines, match, stop_variants, strip_prefix or [], options)
    if located is None:
        return None
    located["label_line_index"] = match["line_index"]
    located["label_bbox"] = match["bbox"]
    return located


def _below_value(lines: List[Dict], match: Dict, stop_variants: List[str],
                 strip_prefix: List[str], options: Dict) -> Optional[Dict]:
    """Value stacked under its label inside one table cell.

    Some cells put the caption on top and the value on the next visual line
    (填制单位 / 洋山海关). The box must sit mostly under the label and start
    within ``below_max_gap`` points, so unrelated text further down cannot be
    mistaken for the value.
    """
    label_box = match["bbox"]
    label_parts = {id(part) for part in match["parts"]}
    max_gap = float(options.get("below_max_gap", 26))
    min_overlap = float(options.get("below_min_overlap", 0.5))
    candidates: List[Dict] = []
    for line_index, line in enumerate(lines):
        for part in sorted(line.get("parts") or [], key=lambda item: item["bbox"][0]):
            if id(part) in label_parts:
                continue
            box = part["bbox"]
            gap = box[1] - label_box[3]
            if gap < -1 or gap > max_gap:
                continue
            overlap = max(0.0, min(box[2], label_box[2]) - max(box[0], label_box[0]))
            ratio = overlap / max(1.0, min(box[2] - box[0], label_box[2] - label_box[0]))
            if ratio < min_overlap:
                continue
            candidates.append({"line_index": line_index, "part": part,
                               "gap": max(0.0, gap), "ratio": ratio})
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["gap"], -item["ratio"], item["part"]["bbox"][0]))
    first = candidates[0]
    text = _part_text(first["part"])
    if any(text.startswith(stop) for stop in stop_variants if stop != match["variant"]):
        return None
    for prefix in strip_prefix:
        compact = _compact(prefix)
        if compact and text.startswith(compact):
            text = text[len(compact):].lstrip(" .:：")
            break
    text = text.strip(":：,，;；()（）[]【】 ￥¥")
    if not _is_value_text(text):
        return None
    chars = compact_line_map({"parts": [first["part"]]})[1][:len(text)]
    return {
        "value": text,
        "bbox": first["part"]["bbox"],
        "confidence": round(float(first["part"].get("confidence") or 0.0), 4),
        "raw_text": first["part"].get("text", ""),
        "line_index": first["line_index"],
        "char_span": [0, len(text)],
        "chars": chars,
    }


def extract_period_range(lines: List[Dict], label: str, aliases: Optional[List[str]] = None,
                         part: str = "start", stop_labels: Optional[List[str]] = None) -> Optional[Dict]:
    located = locate_label_value(lines, label, aliases, stop_labels)
    if not located:
        return None
    text = located["value"]
    spans = [(match.start(), match.end()) for match in
             re.finditer(r"\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?", text)]
    if len(spans) < 2:
        spans = [(match.start(), match.end()) for match in
                 re.finditer(r"(?<!\d)\d{8}(?!\d)", text)]
    if len(spans) < 2:
        return None
    start, end = spans[0] if part == "start" else spans[-1]
    offset = located["char_span"][0]
    return {
        "value": text[start:end],
        "bbox": _bbox_for_range(located["chars"], offset + start, offset + end),
        "confidence": located["confidence"],
        "raw_text": located["raw_text"],
        "line_index": located["line_index"],
        "char_span": [offset + start, offset + end],
    }


def extract_regex(lines: List[Dict], pattern: str, label: Optional[str] = None,
                  aliases: Optional[List[str]] = None, group: int = 0) -> Optional[Dict]:
    regex = re.compile(pattern)
    candidates = lines
    if label:
        variants = _label_variants(label, aliases)
        candidates = []
        for line in lines:
            compact, _ = compact_line_map(line)
            if any(variant in compact for variant in variants):
                candidates.append(line)
    for line in candidates:
        compact, chars = compact_line_map(line)
        match = regex.search(compact)
        if match:
            start, end = match.span(group)
            return {
                "value": match.group(group),
                "bbox": _bbox_for_range(chars, start, end),
                "confidence": line["confidence"],
                "raw_text": line["text"],
            }
    return None


def extract_unit(lines: List[Dict]) -> Optional[Dict]:
    for line in lines:
        compact, chars = compact_line_map(line)
        # 长的在前：否则「人民币百万元」会先命中「人民币元」的中间片段。
        # 除了「单位：万元」，也认报告说明式的「金额类数据项单位均为万元」。
        match = re.search(r"单位(?:[:：]|(?:均|一律)?(?:为|是))?\s*"
                          r"(人民币百万元|人民币万元|人民币元|百万元|万元|元)",
                          compact)
        if match:
            return {
                "value": match.group(1),
                "bbox": _bbox_for_range(chars, match.start(1), match.end(1)),
                "confidence": line["confidence"],
                "raw_text": line["text"],
            }
    return None


_NUMERIC_CELL = re.compile(r"^(?:[-−—–]{1,2}|\d[\d,]*(?:\.\d+)?)$")


def extract_header_matrix(lines: List[Dict], section: Optional[str] = None,
                          header_sequence: Optional[List[str]] = None,
                          column_labels: Optional[List[str]] = None,
                          row_label: Optional[str] = None, value_index: int = 0,
                          max_header_lines: int = 3,
                          max_row_gap: int = 8) -> Optional[Dict]:
    """Read one cell from a table whose sub-header repeats the same column labels.

    The evidence chain is structural, in this order: an optional section heading,
    a header line containing every ``header_sequence`` label in order, a sub-header
    line made of exactly ``len(header_sequence) * len(column_labels)`` parts that
    repeat ``column_labels``, and a ``row_label`` row with exactly that many numeric
    cells. A merged cell, a dropped number or an extra column breaks the count and
    returns None —— 位置对不上就不取，不按比例或邻近猜。
    """
    headers = [_compact(item) for item in (header_sequence or []) if _compact(item)]
    labels = [_compact(item) for item in (column_labels or []) if _compact(item)]
    row_key = _compact(row_label)
    if not headers or not labels or not row_key:
        return None
    width = len(headers) * len(labels)
    if value_index < 0 or value_index >= width:
        return None

    start = 0
    if section:
        section_key = _compact(section)
        start = -1
        for index, line in enumerate(lines):
            compact, _ = compact_line_map(line)
            if section_key and section_key in compact:
                start = index
                break
        if start < 0:
            return None

    header_at = None
    for index in range(start, len(lines)):
        compact, _ = compact_line_map(lines[index])
        cursor = -1
        for head in headers:
            cursor = compact.find(head, cursor + 1)
            if cursor < 0:
                break
        else:
            header_at = index
            break
    if header_at is None:
        return None

    sub_at = None
    for index in range(header_at + 1, min(header_at + 1 + max_header_lines, len(lines))):
        parts = [_compact(part.get("text", "")) for part in lines[index].get("parts", [])]
        parts = [part for part in parts if part]
        if len(parts) != width:
            continue
        if all(parts[i] == labels[i % len(labels)] for i in range(width)):
            sub_at = index
            break
    if sub_at is None:
        return None

    for index in range(sub_at + 1, min(sub_at + 1 + max_row_gap, len(lines))):
        line = lines[index]
        parts = [(part.get("text", ""), part.get("bbox"))
                 for part in line.get("parts", []) if _compact(part.get("text", ""))]
        if not parts or _compact(parts[0][0]) != row_key:
            continue
        cells = parts[1:]
        # 每个数值格必须独立成框；合并/缺列都会让数量对不上，整行放弃。
        if len(cells) != width or not all(_NUMERIC_CELL.match(_compact(text))
                                          for text, _ in cells):
            return None
        text, bbox = cells[value_index]
        return {
            "value": normalize_text(text).strip(),
            "bbox": bbox,
            "confidence": line["confidence"],
            "raw_text": line["text"],
            "line_index": index,
        }
    return None


# ---------------------------------------------------------------------------
# doc type detection
# ---------------------------------------------------------------------------

def detect_doc_type(lines: List[Dict], config: Optional[Dict] = None) -> Tuple[str, float]:
    config = config or load_config()
    options = (config.get("parsing") or {}).get("doc_type") or {}
    text = _compact(" ".join(line["text"] for line in lines))
    scores: Dict[str, float] = {}
    for doc_type, spec in doc_types(config).items():
        score = 0.0
        for keyword, weight in (spec.get("keywords") or {}).items():
            score += weight * text.count(_compact(keyword))
        scores[doc_type] = score
    if not scores:
        return "UNKNOWN", 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    # 两道门都在配置里：绝对分不够说明这份纸根本没印够关键词，差距不够说明两类长得像，
    # 两种情况下猜一个类型都会把错版式当对版式来抽，宁可整份 UNKNOWN。
    min_score = float(options.get("min_score", 3))
    min_margin = float(options.get("min_margin", 1))
    if best_score < min_score or (best_score - second_score) < min_margin:
        return "UNKNOWN", 0.0
    base = float(options.get("confidence_base", 0.55))
    per_point = float(options.get("confidence_per_point", 0.04))
    cap = float(options.get("confidence_cap", 0.99))
    confidence = min(cap, base + per_point * best_score)
    return best_type, round(confidence, 3)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _declared_headers(spec: Dict) -> List[str]:
    """Every header the parser may recognise, in schema order.

    ``optional_headers`` are alternate layouts (a bank statement that splits the
    amount into 借方发生额/贷方发生额 instead of 收支方向 + 交易金额). They must
    become real columns for cell assignment, but must not count toward
    ``min_columns`` nor toward ``table_score`` coverage: a document that uses the
    alternate layout would otherwise look incomplete.
    """
    extra = [header for header in (spec.get("optional_headers") or [])
             if header not in spec["headers"]]
    return list(spec["headers"]) + extra


def _header_alias_map(spec: Dict) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    aliases = spec.get("aliases") or {}
    for header in _declared_headers(spec):
        variants = [header] + list(aliases.get(header, []))
        mapping[header] = sorted({_compact(v) for v in variants}, key=len, reverse=True)
    return mapping


def _match_header(text: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    compact = _compact(text).strip(":：")
    for header, variants in alias_map.items():
        for variant in variants:
            if compact == variant or compact.startswith(variant):
                return header
    return None


def _cell(text: str, bbox: List[float], confidence: float) -> Dict:
    return {"text": normalize_text(text), "bbox": [round(v, 2) for v in bbox],
            "confidence": round(confidence, 4)}


def _table_columns(headers: List[str], centers: Dict[str, float]) -> List[Dict]:
    ordered = [centers[header] for header in headers if header in centers]
    gaps = [ordered[index] - ordered[index - 1] for index in range(1, len(ordered))
            if ordered[index] > ordered[index - 1]]
    radius = (statistics.median(gaps) * 0.62 + 18) if gaps else 120.0
    return [{"header": header, "center": round(centers[header], 2),
             "radius": round(radius, 2)}
            for header in headers if header in centers]


def row_accounting_options(config: Dict) -> Dict:
    return ((config.get("validation") or {}).get("row_accounting")) or {}


def _row_text(row: Dict, limit: int = 60) -> str:
    texts = [str(cell.get("text") or "").strip() for cell in row["cells"].values()]
    return " ".join(text for text in texts if text)[:limit]


def footer_rule(config: Dict) -> Dict:
    """How a dropped row proves itself a page footer rather than a lost transaction.

    ``footer_keywords`` are the words summary rows carry (合计/期末余额/第N页共M页);
    ``footer_max_cells`` bounds how many filled cells such a row may have — a footer is a
    sparse merged row, a real transaction is a full row. Without the bound, a transaction
    whose 摘要 happens to contain 合计 would be waved through as benign.
    """
    options = row_accounting_options(config)
    return {"keywords": [str(item) for item in (options.get("footer_keywords") or [])],
            "max_cells": int(options.get("footer_max_cells") or 0)}


def _rows_from_cells(cells: List[Dict], table_spec_dict: Dict, doc_type: str,
                     page_no: int, table_name: str,
                     skipped: Optional[List[Dict]] = None,
                     footer_options: Optional[Dict] = None) -> List[Dict]:
    """Keep data rows; every dropped row is named with a reason (SCENARIO §5.2 V-BS-07).

    ``skipped`` receives ``{"row_index", "reason"}`` for every row this filter does not
    keep, so the trust layer can tell "that was a repeated header / page footer" from
    "a transaction disappeared". Dropping silently would make the row count unauditable.
    """
    valid_rows: List[Dict] = []
    row_labels = {_compact(label) for label in (table_spec_dict.get("row_labels") or [])}
    required = list(table_spec_dict.get("row_required") or [])
    header_names = {_compact(name) for name in _declared_headers(table_spec_dict)}
    for variants in _header_alias_map(table_spec_dict).values():
        header_names.update(variants)
    rule = footer_options or {}
    words = [word for word in (rule.get("keywords") or []) if word]
    max_cells = int(rule.get("max_cells") or 0)
    for index, row in enumerate(cells):
        row_cells = row["cells"]
        texts = {label: str(cell.get("text") or "") for label, cell in row_cells.items()}
        nonempty = [text for text in texts.values() if text.strip()]
        reason: Optional[str] = None
        if not nonempty:
            reason = "blank"
        elif required and not all(texts.get(label) for label in required):
            reason = "missing_required"
        elif doc_type == "BANK_STATEMENT" and table_name == "transactions":
            date_text = _compact(texts.get("交易日期", ""))
            # Keep date-looking rows even when OCR mangles separators or digits;
            # whether the date is valid is an extraction-layer decision.
            if not re.match(r"^\d{4}", date_text):
                if {_compact(text) for text in nonempty} <= header_names:
                    reason = "header"
                elif max_cells and len(nonempty) <= max_cells \
                        and any(word in " ".join(nonempty) for word in words):
                    reason = "footer"
                else:
                    reason = "unaccounted"
        elif row_labels:
            label_cell = row_cells.get("项目")
            if not label_cell or _compact(label_cell["text"]) not in row_labels:
                reason = "label_unknown"
        if reason:
            if skipped is not None:
                skipped.append({"row_index": index, "reason": reason,
                                "text": _row_text(row)})
            continue
        valid_rows.append(row)
    return valid_rows


def layout_blocks(page, config: Dict) -> List[Dict]:
    """PyMuPDF bundled layout model (offline, ~0.3 s/page): labelled page blocks.

    Importing ``pymupdf.layout`` activates the model; ``get_layout`` fills
    ``page.layout_information`` with ``[x0, y0, x1, y1, label, ...]`` entries.
    Any failure degrades to ``[]`` so the whole-page fallback keeps working.
    """
    if not ((config.get("parsing") or {}).get("layout") or {}).get("enabled"):
        return []
    if _LAYOUT_STATE["ready"] is False:
        return []
    if _LAYOUT_STATE["ready"] is None:
        try:
            import pymupdf.layout  # noqa: F401  importing activates layout support
            _LAYOUT_STATE["ready"] = True
        except Exception:
            _LAYOUT_STATE["ready"] = False
            return []
    try:
        page.get_layout()
        raw = page.layout_information or []
    except Exception:
        _LAYOUT_STATE["ready"] = False
        return []
    return [{"label": str(item[4]), "bbox": [float(item[0]), float(item[1]),
                                             float(item[2]), float(item[3])]}
            for item in raw if len(item) >= 5]


def _table_clusters(blocks: List[Dict], gap: float) -> List[List[float]]:
    """Group table-labelled layout blocks into one bbox per physical table.

    The model may split a table into header/body sub-blocks; blocks within
    ``gap`` points of each other merge into one cluster, tables further apart
    stay separate (a one-page-two-tables page then yields two clusters).
    """
    table_blocks = [block for block in blocks
                    if "table" in str(block.get("label", "")).lower()]
    clusters: List[List[float]] = []
    for block in sorted(table_blocks, key=lambda item: item["bbox"][1]):
        bbox = block["bbox"]
        for cluster in clusters:
            if (bbox[0] <= cluster[2] + gap and bbox[2] >= cluster[0] - gap
                    and bbox[1] <= cluster[3] + gap and bbox[3] >= cluster[1] - gap):
                cluster[0] = min(cluster[0], bbox[0])
                cluster[1] = min(cluster[1], bbox[1])
                cluster[2] = max(cluster[2], bbox[2])
                cluster[3] = max(cluster[3], bbox[3])
                break
        else:
            clusters.append(list(bbox))
    return clusters


def _table_anomalies(data_segments: List[Dict], rejected: List[Dict],
                     filtered_rows: List[Dict], config: Dict,
                     headers: Optional[List[str]] = None,
                     column_centers: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Detect merged-cell / column-misalignment evidence, never repair it.

    Three signals, all attributed to the rows they touch (empty ``rows`` means
    "whole table"): segments the column matcher had to drop, segments tall
    enough to span several row bands, and cells that reach across a neighbouring
    declared column while that column is blank in the same row. Only the table's
    own vertical extent is scanned so footers, titles and summary blocks stay
    out of it.
    """
    anomaly_cfg = (config.get("parsing") or {}).get("table_anomaly") or {}
    if not anomaly_cfg.get("enabled", True):
        return []
    min_ratio = float(anomaly_cfg.get("min_height_ratio", 1.6))
    slack = float(anomaly_cfg.get("vertical_slack", 8))
    min_confidence = float(anomaly_cfg.get("min_confidence", 0.5))
    bands: List[List[float]] = []
    heights: List[float] = []
    for row in filtered_rows:
        boxes = [cell["bbox"] for cell in row["cells"].values()]
        band = [min(box[1] for box in boxes), max(box[3] for box in boxes)]
        bands.append(band)
        heights.append(max(1.0, band[1] - band[0]))
    if not bands:
        return []
    median_height = statistics.median(heights)
    top = bands[0][0] - slack
    bottom = bands[-1][1] + slack

    def affected(box: List[float]) -> List[int]:
        return [index for index, band in enumerate(bands)
                if box[3] > band[0] and box[1] < band[1]]

    anomalies: List[Dict] = []
    for seg in rejected:
        bbox = seg["bbox"]
        if bbox[3] <= top or bbox[1] >= bottom:
            continue
        if float(seg.get("confidence") or 0.0) < min_confidence:
            continue
        anomalies.append({"reason": "column_mismatch",
                          "text": normalize_text(seg["text"]),
                          "bbox": [round(value, 2) for value in bbox],
                          "rows": affected(bbox)})
    for seg in data_segments:
        bbox = seg["bbox"]
        if bbox[3] <= top or bbox[1] >= bottom:
            continue
        if float(seg.get("confidence") or 0.0) < min_confidence:
            continue
        if bbox[3] - bbox[1] < median_height * min_ratio:
            continue
        rows = affected(bbox)
        if len(rows) < 2:
            continue
        anomalies.append({"reason": "row_span",
                          "text": normalize_text(seg["text"]),
                          "bbox": [round(value, 2) for value in bbox],
                          "rows": rows})
    if (anomaly_cfg.get("detect_column_merge", True) and headers and column_centers):
        merge_slack = float(anomaly_cfg.get("column_merge_slack", 2.0))
        for index, row in enumerate(filtered_rows):
            cells = row["cells"]
            for header, cell in cells.items():
                box = cell["bbox"]
                swallowed = [other for other in headers
                             if other != header and not cells.get(other)
                             and box[0] - merge_slack <= column_centers[other] <= box[2] + merge_slack]
                if not swallowed:
                    continue
                anomalies.append({"reason": "column_merge",
                                  "text": normalize_text(cell["text"]),
                                  "bbox": [round(value, 2) for value in box],
                                  "columns": [header] + swallowed,
                                  "rows": [index]})
    return anomalies


def _split_merged_numeric(seg: Dict, headers: List[str],
                          column_bounds: Dict[str, List[float]],
                          spec: Dict) -> Optional[List[Tuple[str, Dict]]]:
    """Split one OCR box that crosses two numeric columns, without guessing.

    Some forms have no rule between 完税价格 and 税率, so OCR returns
    ``6,679,515.0813.0000`` as one box. The split is accepted only when both
    sides fully match the per-column patterns declared in the table schema;
    otherwise the original box stays a column-mismatch anomaly for review.
    The bbox split is proportional to the character count, which is the best
    available estimate because OCR did not return per-character boxes here.
    """
    patterns = spec.get("column_patterns") or {}
    if not patterns:
        return None
    text = _compact(seg.get("text", ""))
    if not text:
        return None
    box = seg["bbox"]
    width = box[2] - box[0]
    if width <= 0:
        return None
    candidates: List[Tuple[float, str]] = []
    for header in headers:
        if header not in patterns or header not in column_bounds:
            continue
        band = column_bounds[header]
        overlap = max(0.0, min(box[2], band[2]) - max(box[0], band[0]))
        if overlap <= 0:
            continue
        span = min(width, band[2] - band[0])
        if span <= 0 or overlap / span < 0.45:
            continue
        candidates.append((band[0], header))
    if len(candidates) < 2:
        return None
    candidates.sort()
    best = None
    for (_, left), (_, right) in zip(candidates, candidates[1:]):
        try:
            left_re = re.compile(patterns[left])
            right_re = re.compile(patterns[right])
        except re.error:
            continue
        for cut in range(1, len(text)):
            if not left_re.fullmatch(text[:cut]) or not right_re.fullmatch(text[cut:]):
                continue
            boundary = column_bounds[right][0]
            expected = (boundary - box[0]) / width
            error = abs(cut / len(text) - expected)
            if best is None or error < best[0]:
                best = (error, cut, left, right)
    if best is None:
        return None
    _, cut, left, right = best
    split_x = box[0] + width * cut / len(text)
    confidence = round(float(seg.get("confidence") or 0.0), 4)
    cells = []
    for header, value, left_x, right_x in (
            (left, text[:cut], box[0], split_x),
            (right, text[cut:], split_x, box[2])):
        cell = _cell(value, [left_x, box[1], right_x, box[3]], confidence)
        cell["merged_split"] = True
        cells.append((header, cell))
    return cells


def _segment_table_in_region(segments: List[Dict], doc_type: str, page_no: int,
                             config: Dict, table_name: str, spec: Dict,
                             region_id: str) -> Optional[Dict]:
    """Header-anchored table build restricted to one segment region."""
    alias_map = _header_alias_map(spec)
    declared = _declared_headers(spec)
    required = set(spec["headers"])
    min_columns = int(spec.get("min_columns", 2))
    # candidate header segments
    candidates: List[Tuple[Dict, str]] = []
    for seg in segments:
        header = _match_header(seg["text"], alias_map)
        if header:
            candidates.append((seg, header))
    if len({h for _, h in candidates if h in required}) < min_columns:
        return None
    # choose the densest y-band
    best_group: List[Tuple[Dict, str]] = []
    for seg, _ in candidates:
        yc = (seg["bbox"][1] + seg["bbox"][3]) / 2
        group = [(s, h) for s, h in candidates
                 if abs(((s["bbox"][1] + s["bbox"][3]) / 2) - yc) <= 7]
        group_headers = {h for _, h in group}
        if len(group_headers) > len({h for _, h in best_group}):
            best_group = group
    if len({h for _, h in best_group if h in required}) < min_columns:
        return None
    chosen: Dict[str, Dict] = {}
    for seg, header in best_group:
        if header not in chosen or seg["bbox"][0] < chosen[header]["bbox"][0]:
            chosen[header] = seg
    headers = [h for h in declared if h in chosen]
    if len([h for h in headers if h in required]) < min_columns:
        return None
    header_bottom = max(chosen[h]["bbox"][3] for h in headers)
    column_centers = {h: (chosen[h]["bbox"][0] + chosen[h]["bbox"][2]) / 2 for h in headers}
    column_bounds = {h: chosen[h]["bbox"] for h in headers}
    # First data row boxes often start a few points above the header bottom
    # because tall glyphs overlap the printed rule; the schema can widen this.
    row_slack = float(spec.get("header_row_slack", 2))
    data_segments = [seg for seg in segments
                     if seg["bbox"][1] >= header_bottom - row_slack
                     and seg not in [c[0] for c in candidates]]
    # A form can render totals, remarks and signature blocks under the item table
    # without a closing rule. The first configured stop label ends the data band.
    stop_labels = [_compact(label) for label in (spec.get("stop_labels") or [])]
    if stop_labels:
        stop_top = None
        for seg in data_segments:
            text = _compact(seg.get("text", ""))
            if any(text.startswith(label) or label in text for label in stop_labels):
                top = seg["bbox"][1]
                if stop_top is None or top < stop_top:
                    stop_top = top
        if stop_top is not None:
            data_segments = [seg for seg in data_segments if seg["bbox"][3] <= stop_top + 2]
    # group by row y
    heights = [max(1.0, s["bbox"][3] - s["bbox"][1]) for s in data_segments]
    row_tol = max(6.0, (statistics.median(heights) * 0.7) if heights else 8.0)
    ordered = sorted(data_segments, key=lambda s: ((s["bbox"][1] + s["bbox"][3]) / 2, s["bbox"][0]))
    rows_segments: List[List[Dict]] = []
    row_centers: List[float] = []
    for seg in ordered:
        center = (seg["bbox"][1] + seg["bbox"][3]) / 2
        if rows_segments and abs(center - row_centers[-1]) <= row_tol:
            rows_segments[-1].append(seg)
        else:
            rows_segments.append([seg])
            row_centers.append(center)
    raw_rows: List[Dict] = []
    rejected: List[Dict] = []
    for row_seg in rows_segments:
        cells: Dict[str, Dict] = {}
        for seg in row_seg:
            split_cells = _split_merged_numeric(seg, headers, column_bounds, spec)
            if split_cells:
                for header, cell in split_cells:
                    if header in cells:
                        cells[header]["text"] = normalize_text(cells[header]["text"] + " " + cell["text"])
                        cells[header]["bbox"] = _union([cells[header]["bbox"], cell["bbox"]])
                        cells[header]["confidence"] = min(cells[header]["confidence"], cell["confidence"])
                    else:
                        cells[header] = cell
                continue
            center = (seg["bbox"][0] + seg["bbox"][2]) / 2
            header = min(headers, key=lambda h: abs(column_centers[h] - center))
            # reject segments far outside the header column band
            distance = abs(column_centers[header] - center)
            widths = [abs(column_centers[h] - column_centers[headers[i - 1]])
                      for i, h in enumerate(headers) if i > 0]
            max_distance = (statistics.median(widths) * 0.75) if widths else 120
            if distance > max_distance + 30:
                rejected.append(seg)
                continue
            if header in cells:
                cells[header]["text"] = normalize_text(cells[header]["text"] + " " + seg["text"])
                cells[header]["bbox"] = _union([cells[header]["bbox"], seg["bbox"]])
                cells[header]["confidence"] = min(cells[header]["confidence"], seg["confidence"])
            else:
                cells[header] = _cell(seg["text"], seg["bbox"], seg["confidence"])
        if cells:
            raw_rows.append({"cells": cells})
    skipped_rows: List[Dict] = []
    filtered_rows = _rows_from_cells(raw_rows, spec, doc_type, page_no, table_name,
                                     skipped=skipped_rows,
                                     footer_options=footer_rule(config))
    if not filtered_rows:
        return None
    anomalies = _table_anomalies(data_segments, rejected, filtered_rows, config,
                                 headers, column_centers)
    all_boxes = [s["bbox"] for s in [chosen[h] for h in headers]] + \
                [cell["bbox"] for row in filtered_rows for cell in row["cells"].values()]
    table = {
        "table_id": (f"p{page_no}-{table_name}" if region_id == "page"
                     else f"p{page_no}-{table_name}-{region_id}"),
        "name": table_name,
        "page": page_no,
        "bbox": _union(all_boxes),
        "headers": headers,
        "columns": _table_columns(headers, column_centers),
        "rows": [{"row_index": index, "page": page_no, "cells": row["cells"]}
                 for index, row in enumerate(filtered_rows)],
        "skipped_rows": skipped_rows,
        "engine": "segment_table",
        "region": region_id,
    }
    if anomalies:
        table["anomalies"] = anomalies
    return table


def _segment_tables(segments: List[Dict], doc_type: str, page_no: int,
                    config: Dict, layout_blocks: Optional[List[Dict]] = None) -> List[Dict]:
    tables: List[Dict] = []
    doc_spec = doc_types(config).get(doc_type, {})
    layout_cfg = (config.get("parsing") or {}).get("layout") or {}
    if layout_cfg.get("enabled") and layout_blocks:
        overlap_ratio = float(layout_cfg.get("block_overlap_ratio", 0.35))
        clusters = _table_clusters(layout_blocks, float(layout_cfg.get("block_merge_gap", 6)))
        for index, cluster in enumerate(clusters):
            region_segments = [seg for seg in segments
                               if _overlap_ratio(seg["bbox"], cluster) >= overlap_ratio]
            if not region_segments:
                continue
            region_id = f"layout-{index}"
            for table_name, spec in (doc_spec.get("tables") or {}).items():
                table = _segment_table_in_region(region_segments, doc_type, page_no,
                                                 config, table_name, spec, region_id)
                if table:
                    tables.append(table)
        if tables:
            return tables
        # 版面模型没切出能用的表格块：退回整页行为，与旧版一致。
    for table_name, spec in (doc_spec.get("tables") or {}).items():
        table = _segment_table_in_region(segments, doc_type, page_no,
                                         config, table_name, spec, "page")
        if table:
            tables.append(table)
    return tables


def _pdfplumber_tables(plumber_page, doc_type: str, page_no: int, config: Dict) -> List[Dict]:
    """Try pdfplumber cells. Returns [] if the structure is not usable."""
    tables: List[Dict] = []
    doc_spec = doc_types(config).get(doc_type, {})
    try:
        found = plumber_page.find_tables()
    except Exception:
        return []
    for table in found:
        try:
            data = table.extract()
        except Exception:
            continue
        if not data or len(data) < 2:
            continue
        table_name = None
        spec = None
        header_index = None
        header_map: Dict[int, str] = {}
        for name, candidate in (doc_spec.get("tables") or {}).items():
            alias_map = _header_alias_map(candidate)
            required = set(candidate["headers"])
            min_columns = int(candidate.get("min_columns", 2))
            for r, row in enumerate(data):
                mapping: Dict[int, str] = {}
                for c, value in enumerate(row):
                    if value is None:
                        continue
                    header = _match_header(str(value), alias_map)
                    if header and header not in mapping.values():
                        mapping[c] = header
                if len([h for h in mapping.values() if h in required]) >= min_columns:
                    table_name, spec, header_index, header_map = name, candidate, r, mapping
                    break
            if table_name:
                break
        if not table_name or header_index is None:
            continue
        headers = [h for h in _declared_headers(spec) if h in header_map.values()]
        header_centers: Dict[str, float] = {}
        for column, header in header_map.items():
            try:
                bbox = table.rows[header_index].cells[column]
            except Exception:
                bbox = None
            if bbox:
                header_centers[header] = (float(bbox[0]) + float(bbox[2])) / 2
        raw_rows: List[Dict] = []
        for r in range(header_index + 1, len(data)):
            row_cells: Dict[str, Dict] = {}
            for c, header in header_map.items():
                if c >= len(data[r]):
                    continue
                text = data[r][c]
                if text is None or not str(text).strip():
                    continue
                bbox = None
                try:
                    bbox = table.rows[r].cells[c]
                except Exception:
                    bbox = None
                if not bbox:
                    bbox = table.bbox
                row_cells[header] = _cell(str(text), list(bbox), 0.99)
            if row_cells:
                raw_rows.append({"cells": row_cells})
        skipped_rows = []
        filtered_rows = _rows_from_cells(raw_rows, spec, doc_type, page_no, table_name,
                                         skipped=skipped_rows,
                                         footer_options=footer_rule(config))
        if not filtered_rows:
            continue
        all_boxes = [list(table.bbox)] + [cell["bbox"] for row in filtered_rows for cell in row["cells"].values()]
        tables.append({
            "table_id": f"p{page_no}-{table_name}",
            "name": table_name,
            "page": page_no,
            "bbox": _union(all_boxes),
            "headers": headers,
            "columns": _table_columns(headers, header_centers),
            "rows": [{"row_index": i, "page": page_no, "cells": row["cells"]}
                     for i, row in enumerate(filtered_rows)],
            "skipped_rows": skipped_rows,
            "engine": "pdfplumber",
        })
    return tables


def _continued_table(segments: List[Dict], previous: Dict, doc_type: str,
                     page_no: int, config: Dict, previous_height: float) -> Optional[Dict]:
    """Rebuild a table on a continuation page that repeats no header row."""
    spec = table_spec(doc_type, previous["name"], config)
    columns = previous.get("columns") or []
    min_columns = int((spec or {}).get("min_columns", 2))
    if not spec or len(columns) < min_columns:
        return None
    min_bottom = float(config.get("parsing", {}).get("continuation_min_bottom_ratio", 0.55))
    row_factor = float(config.get("parsing", {}).get("continuation_row_tolerance", 0.7))
    row_floor = float(config.get("parsing", {}).get("continuation_row_min_tolerance", 6.0))
    column_slack = float(config.get("parsing", {}).get("continuation_column_slack", 30))
    if previous_height and previous["bbox"][3] < previous_height * min_bottom:
        return None
    alias_map = _header_alias_map(spec)
    body = []
    for segment in segments:
        text = normalize_text(segment.get("text", ""))
        if not text.strip():
            continue
        if _match_header(text, alias_map):
            continue
        if re.fullmatch(r"第?\s*\d+\s*页(\s*共\s*\d+\s*页)?|\d+\s*/\s*\d+", text.strip()):
            continue
        body.append(segment)
    if not body:
        return None

    heights = [max(1.0, segment["bbox"][3] - segment["bbox"][1]) for segment in body]
    row_tolerance = max(row_floor, statistics.median(heights) * row_factor)
    ordered = sorted(body, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2,
                                             item["bbox"][0]))
    rows_segments: List[List[Dict]] = []
    row_centers: List[float] = []
    for segment in ordered:
        center = (segment["bbox"][1] + segment["bbox"][3]) / 2
        if rows_segments and abs(center - row_centers[-1]) <= row_tolerance:
            rows_segments[-1].append(segment)
        else:
            rows_segments.append([segment])
            row_centers.append(center)

    raw_rows: List[Dict] = []
    rejected: List[Dict] = []
    for row_segments in rows_segments:
        cells: Dict[str, Dict] = {}
        for segment in row_segments:
            center = (segment["bbox"][0] + segment["bbox"][2]) / 2
            column = min(columns, key=lambda item: abs(item["center"] - center))
            if abs(column["center"] - center) > column["radius"] + column_slack:
                rejected.append(segment)
                continue
            header = column["header"]
            if header in cells:
                cells[header]["text"] = normalize_text(
                    cells[header]["text"] + " " + segment["text"])
                cells[header]["bbox"] = _union([cells[header]["bbox"], segment["bbox"]])
                cells[header]["confidence"] = min(cells[header]["confidence"],
                                                  segment["confidence"])
            else:
                cells[header] = _cell(segment["text"], segment["bbox"],
                                      segment["confidence"])
        if cells:
            raw_rows.append({"cells": cells})
    skipped_rows = []
    filtered = _rows_from_cells(raw_rows, spec, doc_type, page_no, previous["name"],
                                skipped=skipped_rows,
                                footer_options=footer_rule(config))
    wide_rows = []
    for row in filtered:
        if len(row["cells"]) >= min_columns:
            wide_rows.append(row)
        else:
            skipped_rows.append({"row_index": None, "reason": "narrow",
                                 "text": _row_text(row)})
    filtered = wide_rows
    if not filtered:
        return None
    boxes = [cell["bbox"] for row in filtered for cell in row["cells"].values()]
    anomalies = _table_anomalies(body, rejected, filtered, config,
                                 previous["headers"],
                                 {column["header"]: column["center"] for column in columns})
    table = {
        "table_id": f"p{page_no}-{previous['name']}-cont",
        "name": previous["name"],
        "page": page_no,
        "bbox": _union(boxes),
        "headers": previous["headers"],
        "columns": columns,
        "rows": [{"row_index": index, "page": page_no, "cells": row["cells"]}
                 for index, row in enumerate(filtered)],
        "skipped_rows": skipped_rows,
        "engine": "continuation",
        "continued_from": previous["table_id"],
    }
    if anomalies:
        table["anomalies"] = anomalies
    return table


# ---------------------------------------------------------------------------
# document parsing
# ---------------------------------------------------------------------------

def _page_quality_options(config: Dict) -> Dict:
    return (config.get("parsing") or {}).get("page_quality") or {}


def _raster_profile(pixels: List[int], dpi: float, coverage: float, config: Dict) -> Dict:
    minimum = float(_page_quality_options(config).get("min_effective_dpi", 150))
    return {"pixels": [int(pixels[0]), int(pixels[1])], "coverage": round(coverage, 3),
            "effective_dpi": round(dpi, 1), "low_resolution": dpi < minimum}


def _a4_prior_dpi(pixels: List[int], config: Dict) -> float:
    """Sampling density of a raster whose physical paper size is unknown.

    Photo and scan imports are normalized with one point per source pixel (or
    opened straight as an image), so their page geometry says nothing about the
    sheet. The documents in scope are A4-ish, so the pixel count is read back as
    "how many dots per inch would this be on A4".
    """
    options = _page_quality_options(config)
    short_in = float(options.get("a4_short_inches", 8.27))
    long_in = float(options.get("a4_long_inches", 11.69))
    if pixels[1] >= pixels[0]:
        width_in, height_in = short_in, long_in
    else:
        width_in, height_in = long_in, short_in
    return min(pixels[0] / width_in, pixels[1] / height_in)


def _imported_raster(pixels: Optional[List[int]], config: Dict) -> Optional[Dict]:
    """Raster profile for a page normalized from an imported image."""
    options = _page_quality_options(config)
    if not options.get("enabled", True) or not pixels or min(int(p) for p in pixels) <= 0:
        return None
    return _raster_profile(pixels, _a4_prior_dpi(pixels, config), 1.0, config)


def _page_raster(page, config: Dict) -> Optional[Dict]:
    """Sampling density of the scan behind this page, or ``None`` for vector pages.

    A scanned page is only as legible as the pixels it carries: rendering it at
    300 dpi for OCR upsamples blur rather than adding information. PDF points are
    1/72 inch, so an embedded image's physical density is exact; a page drawn one
    point per source pixel has no usable paper size and falls back to the A4 prior.
    """
    options = _page_quality_options(config)
    if not options.get("enabled", True):
        return None
    rect = page.rect
    if rect.is_empty:
        return None
    best: Optional[Dict] = None
    for image in page.get_images(full=True):
        pixels_w, pixels_h = int(image[2]), int(image[3])
        if pixels_w <= 0 or pixels_h <= 0:
            continue
        for shown in page.get_image_rects(image[0]):
            if shown.is_empty:
                continue
            clip = shown & rect
            coverage = (clip.width * clip.height) / max(1.0, rect.width * rect.height)
            if best is None or coverage > best["coverage"]:
                best = {"pixels": [pixels_w, pixels_h], "coverage": coverage,
                        "shown": [shown.width, shown.height]}
    if best is None or best["coverage"] < float(options.get("min_coverage", 0.5)):
        return None
    pixels_w, pixels_h = best["pixels"]
    shown_w, shown_h = best["shown"]
    one_pixel_per_point = (abs(shown_w - pixels_w) <= 1.5
                           and abs(shown_h - pixels_h) <= 1.5
                           and max(rect.width, rect.height) > 900)
    if one_pixel_per_point:
        dpi = _a4_prior_dpi(best["pixels"], config)
    else:
        dpi = min(pixels_w / max(shown_w / 72.0, 1e-6),
                  pixels_h / max(shown_h / 72.0, 1e-6))
    return _raster_profile(best["pixels"], dpi, best["coverage"], config)


GROUNDING_CODE = "GROUNDING_MISSING"
# SCENARIO line 340 的四类失效，拆成机器可比的词；没有第五种。
GROUNDING_REASONS = ("page_missing", "bbox_missing", "bbox_shape",
                     "bbox_degenerate", "bbox_outside")


def grounding_tolerance(config: Dict) -> float:
    """Page-edge slack in points; the one number both layers read for bbox bounds."""
    return float(((config.get("parsing") or {}).get("grounding") or {}).get("tolerance_pt", 1.0))


def grounding_problem(width: float, height: float, bbox, tolerance: float) -> Optional[str]:
    """Why this bbox cannot be traced back to its page, or ``None`` when it can.

    SCENARIO line 340 names the four ways a coordinate fails: no page, no bbox,
    non-positive width or height, bbox outside the page. Pure function so the
    acceptance gate can ask the same question of a planted bbox.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "bbox_missing"
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return "bbox_shape"
    if not all(math.isfinite(value) for value in values):
        return "bbox_shape"
    x0, y0, x1, y1 = values
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return "bbox_degenerate"
    if width > 0 and height > 0 and (x0 < -tolerance or y0 < -tolerance
                                     or x1 > width + tolerance or y1 > height + tolerance):
        return "bbox_outside"
    return None


def _grounding_record(page_no: int, kind: str, reason: str, text: str, bbox) -> Dict:
    return {"code": GROUNDING_CODE, "page": page_no, "kind": kind, "reason": reason,
            "text": (text or "")[:40], "bbox": bbox}


def _reason_counts(pages: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for page in pages:
        for record in page.get("grounding_dropped") or []:
            reason = str(record.get("reason") or "unknown")
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def enforce_part_grounding(page_no: int, width: float, height: float,
                           parts: List[Dict], config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """SCENARIO §7.1: a text block without a page bbox is dropped, not guessed at."""
    tolerance = grounding_tolerance(config)
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for part in parts:
        problem = grounding_problem(width, height, part.get("bbox"), tolerance)
        if problem:
            dropped.append(_grounding_record(page_no, "text", problem,
                                             part.get("text") or "", part.get("bbox")))
        else:
            kept.append(part)
    return kept, dropped


def enforce_table_grounding(page_no: int, width: float, height: float,
                            tables: List[Dict], config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Same rule for tables: the table, its rows and every cell carry a page bbox."""
    tolerance = grounding_tolerance(config)
    kept: List[Dict] = []
    dropped: List[Dict] = []
    for table in tables:
        problem = ("page_missing" if table.get("page") != page_no
                   else grounding_problem(width, height, table.get("bbox"), tolerance))
        if problem:
            dropped.append(_grounding_record(page_no, "table", problem,
                                             table.get("name") or "", table.get("bbox")))
            continue
        rows: List[Dict] = []
        for row in table.get("rows") or []:
            cells: Dict[str, Dict] = {}
            for header, cell in (row.get("cells") or {}).items():
                cell_problem = ("page_missing" if row.get("page") != page_no
                                else grounding_problem(width, height, cell.get("bbox"), tolerance))
                if cell_problem:
                    dropped.append(_grounding_record(page_no, "cell", cell_problem,
                                                     cell.get("text") or "", cell.get("bbox")))
                else:
                    cells[header] = cell
            rows.append(dict(row, cells=cells))
        kept.append(dict(table, rows=rows))
    return kept, dropped


def parse_document(path: Path, doc_id: str, dpi: int = 300, force: bool = False,
                   config: Optional[Dict] = None, out_dir: Optional[Path] = None) -> Dict:
    config = config or load_config()
    out_dir = Path(out_dir) if out_dir else ROOT / config["paths"]["work"] / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.json"
    source_hash = sha256_file(path)
    if out_path.exists() and not force:
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            if cached.get("parser_version") == PARSER_VERSION and cached.get("sha256") == source_hash:
                cached["cached"] = True
                return cached
        except Exception:
            pass

    doc = open_document(path, config)
    prep = page_preprocessing(path, config)
    engine = ocr.get_engine(config)
    text_engine_name = text_engine(config)
    table_engine_name = table_engine(config)
    pages: List[Dict] = []
    all_lines: List[Dict] = []
    engine_names = set()
    uses_ocr = False
    layout_by_page: Dict[int, List[Dict]] = {}
    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        layout_by_page[page_no] = layout_blocks(page, config)
        text_len = len(page.get_text("text").strip())
        enhancement: Dict = {}
        page_used_ocr = False
        if text_len >= int(config["parsing"].get("text_density_threshold", 40)):
            parts = text_layer_parts(page)
            page_engine = f"{text_engine_name}_text"
            ocr_score = None
        else:
            parts, ocr_score, enhancement = ocr_parts(page, dpi, config)
            page_engine = engine.name
            page_used_ocr = engine.is_ocr
            uses_ocr = uses_ocr or page_used_ocr
        parts, dropped_parts = enforce_part_grounding(
            page_no, page.rect.width, page.rect.height, parts, config)
        lines = group_parts(parts)
        recrop_parts = 0
        if page_used_ocr:
            extra = recover_gap_parts(page, lines, config)
            if extra:
                extra, dropped_extra = enforce_part_grounding(
                    page_no, page.rect.width, page.rect.height, extra, config)
                parts.extend(extra)
                dropped_parts.extend(dropped_extra)
                lines = group_parts(parts)
                recrop_parts = len(extra)
        all_lines.extend(lines)
        engine_names.add(page_engine)
        entry = dict(prep[page_index]) if page_index < len(prep) else {}
        flags = list(entry.get("flags") or [])
        photo_flag = preprocess.flag_from_ocr(ocr_score, config) if page_used_ocr else None
        if photo_flag and photo_flag not in flags:
            flags.append(photo_flag)
        raster = _page_raster(page, config) or _imported_raster(entry.get("pixels"), config)
        if raster and raster["low_resolution"] and "LOW_RESOLUTION" not in flags:
            flags.append("LOW_RESOLUTION")
        pages.append({
            "page": page_no,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "engine": page_engine,
            "text_length": text_len,
            "ocr_mean_confidence": ocr_score,
            "quality_flags": flags,
            "raster": raster,
            "preprocess": {
                "geometry": entry.get("geometry") or {},
                "enhancement": enhancement,
                "rectified": bool((entry.get("geometry") or {}).get("warped"))
                or abs(float((entry.get("geometry") or {}).get("deskew_degrees") or 0.0)) > 0,
            },
            "parts": parts,
            "lines": lines,
            "recrop_parts": recrop_parts,
            "grounding_dropped": dropped_parts,
            "tables": [],
            "codes": [],
        })
    doc_type, type_confidence = detect_doc_type(all_lines, config)

    qr_cfg = (config.get("parsing") or {}).get("qr") or {}
    doc_qr = (doc_types(config).get(doc_type) or {}).get("qr") or {}
    if qr_cfg.get("enabled") and doc_qr:
        qr_dpi = int(qr_cfg.get("dpi", 200))
        for page_data in pages:
            try:
                page = doc[page_data["page"] - 1]
                pix = page.get_pixmap(dpi=qr_dpi, alpha=False)
                array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                page_data["codes"] = scan_codes(image, 72.0 / qr_dpi, qr_cfg)
            except Exception:
                page_data["codes"] = []

    # tables: pdfplumber for text-layer pages, segment fallback for all
    plumber = None
    text_layer_engine = f"{text_engine_name}_text"
    if table_engine_name == "pdfplumber" and any(
            page["engine"] == text_layer_engine for page in pages):
        try:
            import pdfplumber
            plumber = pdfplumber.open(path)
        except Exception:
            plumber = None
    previous_tables: Dict[str, Dict] = {}
    previous_height = 0.0
    for page_data in pages:
        page_no = page_data["page"]
        segments = page_data["parts"]
        tables: List[Dict] = []
        if plumber is not None and page_data["engine"] == text_layer_engine:
            try:
                tables = _pdfplumber_tables(plumber.pages[page_no - 1], doc_type, page_no, config)
            except Exception:
                tables = []
        if not tables and doc_type != "UNKNOWN":
            tables = _segment_tables(segments, doc_type, page_no, config,
                                     layout_by_page.get(page_no))
        if doc_type != "UNKNOWN" and previous_tables:
            present = {table["name"] for table in tables}
            for name, previous in previous_tables.items():
                if name in present:
                    continue
                continued = _continued_table(segments, previous, doc_type, page_no,
                                             config, previous_height)
                if continued:
                    tables.append(continued)
        tables, dropped_tables = enforce_table_grounding(
            page_no, page_data["width"], page_data["height"], tables, config)
        page_data["grounding_dropped"].extend(dropped_tables)
        page_data["tables"] = tables
        for table in tables:
            previous_tables[table["name"]] = table
        previous_height = page_data["height"]
        # keep line-level parts for debugging but avoid duplicating entire page twice
        page_data.pop("parts", None)
    if plumber is not None:
        plumber.close()
    doc.close()

    confidences = [line["confidence"] for line in all_lines]
    mean_conf = round(float(statistics.mean(confidences)), 4) if confidences else 0.0
    parse_score = mean_conf if uses_ocr else 0.99
    photo_flags: List[str] = []
    for page_data in pages:
        for flag in page_data["quality_flags"]:
            if flag not in photo_flags:
                photo_flags.append(flag)
    result = {
        "parser_version": PARSER_VERSION,
        "document_id": doc_id,
        "source": path.name,
        "sha256": source_hash,
        "page_count": len(pages),
        "doc_type": doc_type,
        "doc_type_confidence": type_confidence,
        "engine": "+".join(sorted(engine_names)),
        "quality": {
            "parse_score": parse_score,
            "mean_confidence": mean_conf,
            "line_count": len(all_lines),
            "ocr_used": uses_ocr,
            "photo_flags": photo_flags,
            "low_resolution_pages": [item["page"] for item in pages
                                     if "LOW_RESOLUTION" in item["quality_flags"]],
            "rectified_pages": sum(1 for item in pages if item["preprocess"]["rectified"]),
            "recrop_parts": sum(item.get("recrop_parts", 0) for item in pages),
            "grounding_dropped": sum(len(item.get("grounding_dropped") or []) for item in pages),
            "grounding_dropped_by_reason": _reason_counts(pages),
            "grounding_tolerance_pt": grounding_tolerance(config),
            "enhanced_pages": sum(1 for item in pages
                                  if item["preprocess"]["enhancement"].get("seal_suppressed")
                                  or item["preprocess"]["enhancement"].get("highlight_suppressed")),
        },
        "pages": pages,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["cached"] = False
    return result


def _cache_tag(config: Dict) -> str:
    """Digest of everything that changes the normalized copy, so config edits rebuild."""
    ingest = config.get("ingest", {})
    relevant = {
        "parser_version": PARSER_VERSION,
        "image_max_side": ingest.get("image_max_side"),
        "multiframe_suffixes": sorted(str(item).lower().lstrip(".")
                                      for item in ingest.get("multiframe_suffixes", [])),
        "preprocess": config.get("preprocess"),
    }
    blob = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def _image_frames(path: Path, config: Dict) -> List[Dict]:
    """Decode one entry per page: EXIF orientation, downscale, rectify, quality flags."""
    from PIL import Image, ImageOps

    ingest = config.get("ingest", {})
    max_side = int(ingest.get("image_max_side", 2600))
    multiframe = {str(item).lower().lstrip(".")
                  for item in ingest.get("multiframe_suffixes", ["tif", "tiff"])}
    pages: List[Dict] = []
    with Image.open(path) as source:
        wanted = source.n_frames if path.suffix.lower().lstrip(".") in multiframe else 1
        for index in range(max(1, wanted)):
            source.seek(index)
            frame = ImageOps.exif_transpose(source)
            if max_side > 0 and max(frame.size) > max_side:
                ratio = max_side / max(frame.size)
                size = (max(1, round(frame.width * ratio)), max(1, round(frame.height * ratio)))
                frame = frame.resize(size, Image.LANCZOS)
            array = np.asarray(frame.convert("RGB"))
            array, record = preprocess.rectify(array, config)
            pages.append({"image": Image.fromarray(array), "geometry": record,
                          "pixels": [int(array.shape[1]), int(array.shape[0])],
                          "flags": preprocess.quality_flags(array, record, config)})
    return pages


def normalized_source(path: Path, config: Optional[Dict] = None) -> Path:
    """Pipeline-ready copy of an input: the PDF itself, or a normalized image build.

    Images go through Pillow first: EXIF orientation is applied here so the result
    never depends on the viewer, oversized phone photos are downscaled, skewed or
    perspective-shot pages are rectified, and formats MuPDF cannot read (WEBP)
    become PNG. Multi-frame formats listed in ``ingest.multiframe_suffixes``
    (scanner TIFF) keep one page per frame and are written as PDF. The copy is
    cached under work/render/images keyed by the source SHA-256 plus a digest of
    the normalization settings, and the original in the inbox is never rewritten.
    """
    config = config or load_config()
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return path
    cache = resolve_path(config["paths"]["work"], config) / "render" / "images"
    cache.mkdir(parents=True, exist_ok=True)
    key = f"{sha256_file(path)}-{_cache_tag(config)}"
    sidecar = cache / f"{key}.prep.json"
    for kind in ("png", "pdf"):
        target = cache / f"{key}.{kind}"
        if target.exists() and sidecar.exists():
            return target
    pages = _image_frames(path, config)
    kind = "pdf" if len(pages) > 1 else "png"
    target = cache / f"{key}.{kind}"
    temp = cache / f"{key}.part.{kind}"
    sidecar_temp = cache / f"{key}.part.json"
    try:
        images = [item["image"] for item in pages]
        if kind == "pdf":
            images[0].save(temp, "PDF", save_all=True, append_images=images[1:], resolution=72.0)
        else:
            images[0].save(temp, "PNG")
        payload = [{"page": index + 1, "geometry": item["geometry"], "flags": item["flags"],
                    "pixels": item["pixels"]}
                   for index, item in enumerate(pages)]
        # 两份文件都写成「先临时再改名」：中途断电留下的半截 sidecar 会让以后每次
        # 命中这个缓存都读坏 JSON，而直接覆盖写做不到这一点。
        sidecar_temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        sidecar_temp.replace(sidecar)
        temp.replace(target)
    finally:
        for scratch in (temp, sidecar_temp):
            if scratch.exists():
                scratch.unlink()
    return target


def page_preprocessing(path: Path, config: Optional[Dict] = None) -> List[Dict]:
    """Per-page rectify records and photo flags recorded when the image was normalized."""
    config = config or load_config()
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return []
    key = f"{sha256_file(path)}-{_cache_tag(config)}"
    sidecar = resolve_path(config["paths"]["work"], config) / "render" / "images" / f"{key}.prep.json"
    if not sidecar.exists():
        return []
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def open_document(path: Path, config: Optional[Dict] = None) -> "fitz.Document":
    """Open a document through normalized_source, so every layer shares one geometry."""
    config = config or load_config()
    return fitz.open(normalized_source(path, config))


def iter_inputs(input_dir: Path, config: Optional[Dict] = None) -> List[Path]:
    """List importable files (PDF and configured image types) under a directory."""
    config = config or load_config()
    allowed = {str(item).lower().lstrip(".")
               for item in config.get("ingest", {}).get("allowed_suffixes", ["pdf"])}
    return sorted(path for path in input_dir.rglob("*")
                  if path.is_file() and path.suffix.lower().lstrip(".") in allowed)


def document_id_for(path: Path, manifest: Optional[Dict] = None) -> str:
    if manifest:
        for entry in manifest.get("documents", []):
            manifest_path = (ROOT / entry["path"]).resolve()
            if manifest_path == path.resolve():
                return entry["document_id"]
    name = path.stem
    if name.endswith(".scan"):
        name = name[:-5]
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description="Parse PDFs into lines and table cells with bboxes")
    parser.add_argument("--input", default=str(ROOT / "samples" / "dev"))
    parser.add_argument("--out", default=str(ROOT / "work" / "parsed"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    input_dir = Path(args.input)
    config = load_config()
    out_dir = Path(args.out)
    manifest = None
    manifest_path = input_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs = iter_inputs(input_dir, config)
    if args.limit:
        pdfs = pdfs[:args.limit]
    for path in pdfs:
        doc_id = document_id_for(path, manifest)
        result = parse_document(path, doc_id, dpi=args.dpi, force=args.force,
                                config=config, out_dir=out_dir)
        tables = sum(len(page["tables"]) for page in result["pages"])
        print(f"{doc_id:<5} {result['doc_type']:<24} pages={result['page_count']} "
              f"engine={result['engine']:<12} tables={tables} cached={result.get('cached')}")


if __name__ == "__main__":
    main()
