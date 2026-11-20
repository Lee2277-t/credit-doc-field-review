"""Unit tests for the review/export/evaluation contracts.

Run from the project root: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import random
import re
import shutil
import tempfile
import time
import unittest
import zipfile
from unittest import mock
from datetime import date
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

from docpipe import (calibrate, clispec, corpusaudit, csvaudit, docaudit, doctypeaudit, hardset, ocr,
                     labelaudit, gatecontract, modelaudit, parseaudit, preprocess, riskcontract, schemapaths, summaryaudit, synth,
                     tablecross,
                     tablescore)
from docpipe.config import (ROOT, evaluation_only_splits, field_spec, field_specs,
                            load_config, table_spec)
from docpipe.evaluate import (_decision_stats, claim_targets, decision_claim, error_category,
                              grounding_ok, transaction_comparison, values_match)
from docpipe.export import export, export_zip, load_final_documents
from docpipe.extract import (_extract_line_cell, _normalize, _qr_crosscheck, _raw_text_audit,
                             _statement_header_anchors, extract_document)
from docpipe.importer import (process_inbox, prune_trash, remove_document,
                              safe_upload_name, store_upload)
from docpipe.parse import (_continued_table, _page_raster, _segment_tables,
                           _split_merged_numeric, accepts_recrop_text, detect_doc_type,
                           extract_header_matrix, extract_unit, gap_recrop_rect,
                           group_parts, iter_inputs, locate_label_region, ocr_parts,
                           open_document, parse_document, recover_gap_parts)
from docpipe.review_actions import (field_actions, review_action_table, review_actions)
from docpipe.reviews import (ACTION_DEFAULTS, REVIEWED_ACCEPT, REVIEWED_CORRECTED,
                             action_vocab, payload_vocab, read_review_ledger,
                             read_review_records, review_status, status_from_vocab, summarize)
from docpipe.report import build_report
from docpipe.trust import (Calibrator, _percent, assess, decide_field,
                           feature_names, validate_customs_payment,
                           validate_document_fields, validate_income_statement,
                           validate_vat_invoice)

def requires(*paths):
    """仓库不含合成语料与运行产物（见 .gitignore）：缺件时显式跳过，而不是报一堆文件不存在。"""
    return unittest.skipUnless(
        all((ROOT / item).exists() for item in paths),
        "缺少生成物：先跑 python -m docpipe.synth、docpipe.run、docpipe.calibrate 与 docpipe.export")



def _png_bytes(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, "PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buffer, "JPEG")
    return buffer.getvalue()


def _noisy_png_bytes(size: int = 2600) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((size, size), 120).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def _text_page(width: int = 900, height: int = 1200, rows: int = 18) -> np.ndarray:
    """White page with dark horizontal bars: the row profile deskew keys on."""
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    step = max(12, height // rows)
    for index in range(1, rows):
        start = index * step
        array[start:start + 6, int(width * 0.12):int(width * 0.88)] = 20
    return array


def _rotate_page(array: np.ndarray, degrees: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((array.shape[1] / 2.0, array.shape[0] / 2.0), degrees, 1.0)
    return cv2.warpAffine(array, matrix, (array.shape[1], array.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _perspective_shot(corners: list, frame: int = 90) -> np.ndarray:
    """A page photographed as a quadrilateral inside a darker frame."""
    source = np.array([[0, 0], [700, 0], [700, 950], [0, 950]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, np.array(corners, dtype=np.float32))
    return cv2.warpPerspective(_text_page(700, 950, 16), matrix, (1000, 1300),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(frame, frame, frame))


def _uneven_page(page: np.ndarray, low: float = 0.80, high: float = 0.95) -> np.ndarray:
    """A photographed page is never evenly lit; the lamp slops across it sideways."""
    ramp = np.linspace(low, high, page.shape[1], dtype=np.float32)
    return np.clip(page.astype(np.float32) * ramp[None, :, None], 0, 255).astype(np.uint8)


def _glare_shot(page: np.ndarray, alpha: float = 0.90) -> np.ndarray:
    """The uneven page with a specular band laid over its left third."""
    array = _uneven_page(page)
    height, width = array.shape[:2]
    band = np.array([[int(width * 0.30), 0], [int(width * 0.55), 0],
                     [int(width * 0.20), height], [0, height]], dtype=np.int32)
    shaded = np.clip(array.astype(np.float32) * (1 - alpha) + 255.0 * alpha, 0, 255).astype(np.uint8)
    return np.where(cv2.fillPoly(np.zeros((height, width), np.uint8), [band], 255)[..., None],
                    shaded, array)


def _in_band(shape) -> np.ndarray:
    """The same band as a pixel mask, at whatever resolution the caller is measuring."""
    height, width = shape
    band = np.array([[int(width * 0.30), 0], [int(width * 0.55), 0],
                     [int(width * 0.20), height], [0, height]], dtype=np.int32)
    return cv2.fillPoly(np.zeros((height, width), np.uint8), [band], 255) > 127


def _analysis_binary(array: np.ndarray, options: dict) -> np.ndarray:
    small = preprocess._shrunk(preprocess._gray(array), int(options["analyze_max_side"]))
    return cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _pdf_bytes(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((50, 80), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def _finger_shot(page: np.ndarray, color=(223, 177, 148)) -> np.ndarray:
    """A diagonal skin-coloured bar running off the bottom-left of the frame."""
    array = _uneven_page(page)
    height, width = array.shape[:2]
    cv2.line(array, (-int(width * 0.05), height + 20),
             (int(width * 0.34), int(height * 0.42)), color, max(6, int(width * 0.085)))
    return array


class ValueMatchTests(unittest.TestCase):
    def test_identity_ignores_separators(self):
        spec = {"type": "string", "risk": "identity"}
        self.assertTrue(values_match(spec, "9144 0101 MA5X 8K2L", "91440101MA5X8K2L"))

    def test_decimal_is_exact_after_normalization(self):
        spec = {"type": "decimal"}
        self.assertTrue(values_match(spec, "100.00", "100"))
        self.assertFalse(values_match(spec, "100.00", "100.01"))

    def test_date_normalization(self):
        spec = {"type": "date"}
        self.assertTrue(values_match(spec, "2025年7月1日", "2025-07-01"))
        self.assertFalse(values_match(spec, "2025-07-01", "2025-07-02"))


class TransactionMatchTests(unittest.TestCase):
    def test_rows_match_by_content_not_order(self):
        config = load_config()
        labels = {"doc_type": "BANK_STATEMENT", "fields": {}}
        pred = {}
        rows = [
            ("2025-07-01", "100.00", "1100.00", "income"),
            ("2025-07-02", "50.00", "1050.00", "expense"),
        ]
        for index, (day, amount, balance, direction) in enumerate(rows):
            for column, value in (("txn_date", day), ("amount", amount),
                                  ("balance", balance), ("direction", direction)):
                path = f"bs.transactions[{index}].{column}"
                labels["fields"][path] = {"value": value}
        # Reverse row order in the prediction.
        for index, source in ((0, 1), (1, 0)):
            day, amount, balance, direction = rows[source]
            for column, value in (("txn_date", day), ("amount", amount),
                                  ("balance", balance), ("direction", direction)):
                path = f"bs.transactions[{index}].{column}"
                pred[path] = {"value": value}

        result = transaction_comparison(labels, pred, "BANK_STATEMENT", config)
        self.assertEqual(result["gt_rows"], 2)
        self.assertEqual(result["order_sensitive_correct"], 0)
        self.assertEqual(result["content_matched"], 2)
        self.assertEqual(result["content_match_rate"], 1.0)

    def test_spec_lookup_for_transaction_column(self):
        spec = field_spec("BANK_STATEMENT", "bs.transactions[i].amount", load_config())
        self.assertEqual(spec["risk"], "monetary")


class ReviewHelpersTests(unittest.TestCase):
    def test_summary_counts(self):
        fields = {
            "a": {"decision": "review"},
            "b": {"decision": "review"},
            "c": {"decision": "accept"},
        }
        records = {
            "a": {"action": "accept"},
            "b": {"action": "correct", "corrected_value": "x"},
        }
        summary = summarize(fields, records)
        self.assertEqual(summary["reviewed_count"], 2)
        self.assertEqual(summary["reviewed_accept_count"], 1)
        self.assertEqual(summary["reviewed_corrected_count"], 1)
        self.assertEqual(summary["pending_review_count"], 0)
        self.assertEqual(summary["correction_rate"], 0.5)

    def test_last_record_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "review").mkdir()
            (out / "review" / "T1.review.jsonl").write_text(
                '{"field_path":"a","action":"accept"}\n'
                '{"field_path":"a","action":"correct","corrected_value":"x"}\n',
                encoding="utf-8")
            records = read_review_records(out, "T1")
            self.assertEqual(records["a"]["action"], "correct")


class ReviewVocabularyTests(unittest.TestCase):
    """复核判决的动作词汇表只许有一个主人：配置。改名时写入侧、成品、汇总、审计、界面一起翻面。"""

    def _renamed(self, action="amend", state="reviewed_amended"):
        config = copy.deepcopy(load_config())
        delivery = config["validation"]["delivery"]
        delivery["review_state_pairs"] = {"accept": "reviewed_accept", action: state}
        delivery["value_changing_actions"] = [action]
        # 计数的**键名**是产物契约（CSV / summary / 界面筛选都靠它），不跟着改名；
        # 它数的是哪个状态由 `review_count_keys` 说，完整改名必须把这一条也一起改。
        delivery["review_count_keys"] = {"reviewed_count": None,
                                         "reviewed_corrected_count": state}
        return config

    def _out_fixture(self, root: Path, config: Dict, action: str) -> Path:
        changing = action_vocab(config)["value_changing"]
        out = root / "out"
        (out / "T1").mkdir(parents=True)
        (out / "review").mkdir()
        summary = {
            "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
            "overall": {},
            "by_doc_type": [],
            "by_field_path": [],
            "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                           "path": "T1/T1.json", "review_count": 1}],
        }
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        document = {
            "document_id": "T1",
            "doc_type": "BANK_STATEMENT",
            "fields": {
                "bs.account_name": {
                    "path": "bs.account_name", "value": "旧值", "value_type": "string",
                    "confidence": 0.9, "decision": "review",
                    "grounding": {"page": 1, "bbox": [1, 2, 3, 4]},
                    "extractor_id": "label_value:户名", "review_reasons": ["LOW_OCR"],
                }
            },
        }
        (out / "T1" / "T1.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8")
        (out / "review" / "T1.review.jsonl").write_text(
            json.dumps({"document_id": "T1", "field_path": "bs.account_name",
                        "original_value": "旧值", "action": action,
                        "corrected_value": "新值" if action in changing else None,
                        "reviewer": "local", "timestamp": "2026-10-24T12:01:00+08:00",
                        "auto_decision": "review"},
                       ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def test_the_shipped_vocab_has_exactly_one_master(self):
        vocab = action_vocab(load_config())
        from docpipe import deliveryaudit
        # 审计自带一份字面量（审计员不该被被审的代码牵着走），所以这里三方逐字对齐，
        # 改名只改配置一处，另外两份必须跟着红——这条断言就是那盏灯。
        for action, state in ACTION_DEFAULTS["review_state_pairs"].items():
            self.assertEqual(vocab["pairs"][action], state)
            self.assertEqual(deliveryaudit.DEFAULTS["review_state_pairs"][action], state)
        self.assertEqual(vocab["changing"], list(ACTION_DEFAULTS["value_changing_actions"]))
        self.assertEqual(vocab["changing"], list(deliveryaudit.DEFAULTS["value_changing_actions"]))
        self.assertEqual(vocab["corrected_state"], REVIEWED_CORRECTED)

    def test_a_vocab_that_accuses_itself_fails_closed_at_the_producer(self):
        config = copy.deepcopy(load_config())
        config["validation"]["delivery"]["value_changing_actions"] = ["amend"]
        with self.assertRaises(ValueError) as ctx:
            action_vocab(config)
        self.assertIn("amend", str(ctx.exception))
        from docpipe import deliveryaudit
        # 审计侧不崩、但要把它点名：假指控的责任在词汇表，不在签字的人。
        with tempfile.TemporaryDirectory() as temp:
            result = deliveryaudit.reconcile(Path(temp), deliveryaudit.options(config))
        self.assertIn("value_changing_action_unpaired",
                      {item["problem"] for item in result["problems"]})

    def test_renaming_the_action_moves_the_whole_chain(self):
        from docpipe.serve import ReviewStore
        config = self._renamed()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = self._out_fixture(root, config, "amend")
            store = ReviewStore(out, config)
            store.append_review("T1", {"field_path": "bs.account_name", "action": "amend",
                                       "corrected_value": "新值", "reviewer": "local"})
            with self.assertRaises(ValueError) as stale:
                store.append_review("T1", {"field_path": "bs.account_name", "action": "correct",
                                           "corrected_value": "旧值", "reviewer": "local"})
            self.assertIn("accept / amend", str(stale.exception))
            result = export(out, out / "final", config)
            final = json.loads((out / "final" / "T1" / "T1.json").read_text(encoding="utf-8"))
            field = final["fields"]["bs.account_name"]
            self.assertEqual(field["effective_value"], "新值")
            self.assertEqual(field["review_status"], "reviewed_amended")
            self.assertEqual(result["overall"]["reviewed_corrected_count"], 1)
            from docpipe import deliveryaudit
            recon = deliveryaudit.reconcile(out, deliveryaudit.options(config))
            self.assertEqual(recon["problems"], [])
            self.assertEqual(recon["mismatches"], [])

    def test_no_value_changing_action_counts_no_corrections(self):
        config = copy.deepcopy(load_config())
        config["validation"]["delivery"]["value_changing_actions"] = []
        vocab = action_vocab(config)
        self.assertIsNone(vocab["corrected_state"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = self._out_fixture(root, config, "correct")
            result = export(out, out / "final", config)
            # 空表时没有任何状态算「人工修正」——拿 None 去比会把所有未复核的格子数成修正。
            self.assertEqual(result["overall"]["reviewed_corrected_count"], 0)
            self.assertEqual(result["overall"]["reviewed_count"], 1)

    def test_the_payload_names_both_actions_in_singular_for_the_display_layers(self):
        vocab = payload_vocab(self._renamed())
        self.assertEqual(vocab["correct_action"], "amend")
        self.assertEqual(vocab["confirm_action"], "accept")
        self.assertEqual(vocab["corrected_state"], "reviewed_amended")
        json.dumps(vocab)  # 进得去 JSON 载荷才算给界面用
        self.assertEqual(payload_vocab()["correct_action"], "correct")

    def test_summarize_and_review_status_follow_the_config(self):
        config = self._renamed()
        fields = {"a": {"decision": "review"}, "b": {"decision": "review"}}
        records = {"a": {"action": "accept"}, "b": {"action": "amend", "corrected_value": "x"}}
        summary = summarize(fields, records, config)
        self.assertEqual(summary["reviewed_corrected_count"], 1)
        self.assertEqual(summary["reviewed_accept_count"], 1)
        self.assertEqual(review_status(records["b"], config), "reviewed_amended")
        self.assertEqual(review_status(records["a"], config), REVIEWED_ACCEPT)
        self.assertEqual(status_from_vocab(action_vocab(config), {"action": "unknown"}),
                         REVIEWED_ACCEPT)

    def test_the_console_takes_its_words_from_the_payload(self):
        root = Path(__file__).resolve().parents[1] / "docpipe"
        script = (root / "web" / "app.js").read_text(encoding="utf-8")
        vocab = action_vocab(load_config())
        # 界面拿不到 YAML，只能靠载荷；它自带的兜底必须逐字写着出厂动作，且只写这一处。
        fallback = '{{ confirm: "{}", correct: "{}" }}'.format(
            vocab["confirming"][0], vocab["changing"][0])
        self.assertEqual(script.count("const SHIPPED_REVIEW_VOCAB = " + fallback + ";"), 1)
        self.assertIn("payload.review_vocab", script)
        for word in ("confirm", "correct"):
            self.assertIn('data-action="${reviewWords().%s}"' % word, script)
        # 台账动作一律走 reviewWords；界面里剩下的 "accept" 是机器判决词汇，不是复核动作。
        for action in (vocab["changing"][0], vocab["confirming"][0]):
            self.assertNotIn('action === "{}"'.format(action), script)
            self.assertNotIn('postReview(path, "{}"'.format(action), script)


class ExportTests(unittest.TestCase):
    def test_export_applies_last_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            (out / "T1").mkdir(parents=True)
            (out / "review").mkdir()
            summary = {
                "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
                "overall": {},
                "by_doc_type": [],
                "by_field_path": [],
                "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                               "path": "T1/T1.json", "review_count": 1}],
            }
            (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            document = {
                "document_id": "T1",
                "doc_type": "BANK_STATEMENT",
                "fields": {
                    "bs.account_name": {
                        "path": "bs.account_name", "value": "旧值", "value_type": "string",
                        "confidence": 0.9, "decision": "review",
                        "grounding": {"page": 1, "bbox": [1, 2, 3, 4]},
                        "extractor_id": "label_value:户名", "review_reasons": ["LOW_OCR"],
                    }
                },
            }
            (out / "T1" / "T1.json").write_text(
                json.dumps(document, ensure_ascii=False), encoding="utf-8")
            (out / "review" / "T1.review.jsonl").write_text(
                json.dumps({"field_path": "bs.account_name", "action": "correct",
                            "original_value": "旧值", "corrected_value": "=1+2",
                            "reviewer": "local", "timestamp": "2026-10-24T12:01:00+08:00"},
                           ensure_ascii=False) + "\n", encoding="utf-8")

            result = export(out, out / "final", load_config())
            final = json.loads((out / "final" / "T1" / "T1.json").read_text(encoding="utf-8"))
            field = final["fields"]["bs.account_name"]
            self.assertEqual(field["value"], "旧值")
            self.assertEqual(field["effective_value"], "=1+2")
            self.assertEqual(field["review_status"], "reviewed_corrected")
            self.assertEqual(result["overall"]["reviewed_corrected_count"], 1)
            self.assertEqual(result["overall"]["pending_review_count"], 0)
            with (out / "final" / "final_fields.csv").open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["effective_value"], "'=1+2")
            self.assertEqual(row["review_status"], "reviewed_corrected")


class ExportZipTests(unittest.TestCase):
    def test_zip_contains_final_and_review_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "out" / "user"
            (out / "T1").mkdir(parents=True)
            (out / "review").mkdir()
            (out / "summary.json").write_text(json.dumps({
                "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
                "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                               "path": "T1/T1.json", "review_count": 0}],
            }), encoding="utf-8")
            (out / "T1" / "T1.json").write_text(json.dumps({
                "document_id": "T1", "doc_type": "BANK_STATEMENT",
                "fields": {"bs.account_name": {
                    "path": "bs.account_name", "value": "示例公司",
                    "value_type": "string", "decision": "accept",
                    "confidence": 0.99, "grounding": {"page": 1, "bbox": [1, 2, 3, 4]},
                }},
            }, ensure_ascii=False), encoding="utf-8")
            (out / "review" / "T1.review.jsonl").write_text("", encoding="utf-8")
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")

            result = export_zip(out, config=config)
            self.assertTrue(Path(result["path"]).exists())
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
            self.assertIn("final/final_summary.json", names)
            self.assertIn("final/final_fields.csv", names)
            self.assertIn("final/T1/T1.json", names)
            self.assertIn("review/T1.review.jsonl", names)
            self.assertIn("export_manifest.json", names)
            self.assertEqual(result["documents"], 1)


class ReportTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        out = root / "out" / "user"
        (out / "T1").mkdir(parents=True)
        (out / "review").mkdir()
        (out / "summary.json").write_text(json.dumps({
            "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
            "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                           "path": "T1/T1.json", "review_count": 1}],
        }), encoding="utf-8")
        fields = {
            "bs.account_name": {
                "path": "bs.account_name", "label": "户名", "risk": "party_name",
                "value": "旧户名", "value_type": "string", "confidence": 0.61,
                "decision": "review", "review_reasons": ["LOW_OCR"],
                "grounding": {"page": 1, "bbox": [10, 20, 90, 32],
                              "bbox_norm": [0.02, 0.05, 0.2, 0.08]},
            },
            "bs.period_start": {
                "path": "bs.period_start", "label": "起始日期", "risk": "date",
                "value": "2026-01-01", "value_type": "date", "confidence": 0.97,
                "decision": "accept", "grounding": {"page": 1},
            },
            "bs.total_in": {
                "path": "bs.total_in", "label": "收入合计", "risk": "monetary",
                "value": "1234.50", "value_type": "decimal", "confidence": 0.88,
                "decision": "review", "grounding": {"page": 2},
            },
            "bs.transactions[0].txn_date": {
                "path": "bs.transactions[0].txn_date", "label": "交易日期", "risk": "date",
                "value": "2026-01-05", "value_type": "date", "confidence": 0.95,
                "decision": "accept", "grounding": {"page": 1},
            },
            "bs.transactions[0].amount": {
                "path": "bs.transactions[0].amount", "label": "交易金额", "risk": "monetary",
                "value": "88.00", "value_type": "decimal", "confidence": 0.9,
                "decision": "accept", "grounding": {"page": 1},
            },
        }
        (out / "T1" / "T1.json").write_text(json.dumps({
            "document_id": "T1", "doc_type": "BANK_STATEMENT",
            "source": {"path": "", "file_name": "示例流水.pdf", "page_count": 2},
            "fields": fields,
        }, ensure_ascii=False), encoding="utf-8")
        (out / "review" / "T1.review.jsonl").write_text(
            json.dumps({"field_path": "bs.account_name", "action": "correct",
                        "original_value": "旧户名", "corrected_value": "新户名",
                        "reviewer": "local", "timestamp": "2026-10-24T12:01:00+08:00"},
                       ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def test_report_is_offline_and_grouped_in_chinese(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self._workspace(Path(temp))
            result = build_report(out, config=load_config())
            html = Path(result["path"]).read_text(encoding="utf-8")
            self.assertEqual(result["name"], "查看结果.html")
            self.assertEqual(result["documents"], 1)
            self.assertEqual(result["embedded_pages"], 0)
            self.assertIn("信贷资料结构化结果", html)
            self.assertIn("银行流水", html)
            self.assertIn("主体与证件", html)
            self.assertIn("金额与统计", html)
            self.assertIn("交易明细", html)
            self.assertIn("新户名", html)
            self.assertIn("待复核", html)
            self.assertIn("自动接受", html)
            self.assertNotIn("未复核", html)
            for marker in ('src="http', 'href="http', "<link", "cdn."):
                self.assertNotIn(marker, html)

    def test_report_is_the_fourth_exit_for_the_rejection_contract(self):
        """复核台、两份 CSV、两份 summary 都带「怎么办」，单文件网页版没有。

        拿到成品的人不在复核台前，同样得知道每个被拦字段该去核对什么；文案只能从
        `validation.review_actions` 来，模板里写死一句就等于开了第二个口径。
        表格里被拦的那一格在网页上没有独立行，指引挂整行下面——holdout 唯一被拦的
        就是交易金额一格，漏掉它这个出口在 holdout 上就是空转。
        """
        with tempfile.TemporaryDirectory() as temp:
            out = self._workspace(Path(temp))
            config = load_config()
            action = config["validation"]["review_actions"]["LOW_OCR"]
            record = json.loads((out / "T1" / "T1.json").read_text(encoding="utf-8"))
            cell = record["fields"]["bs.transactions[0].amount"]
            cell["decision"], cell["review_reasons"] = "review", ["TABLE_ANOMALY"]
            (out / "T1" / "T1.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            html = Path(build_report(out, config=config)["path"]).read_text(encoding="utf-8")
            self.assertIn('class="hint"', html)
            self.assertIn("怎么办", html)
            self.assertIn(action, html)
            payload = json.loads(re.search(
                r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S
                ).group(1).replace("<\\/", "</"))
            table = payload["documents"][0]["tables"][0]
            self.assertEqual(table["rows"][0]["actions"],
                             [config["validation"]["review_actions"]["TABLE_ANOMALY"]])
            self.assertEqual(len(table["rows"][0]["cells"]), len(table["columns"]))
            config["validation"]["review_actions"]["LOW_OCR"] = "换一句只写在配置里的核对指引"
            rebuilt = Path(build_report(out, config=config)["path"]).read_text(encoding="utf-8")
            self.assertIn("换一句只写在配置里的核对指引", rebuilt)
            self.assertNotIn(action, rebuilt)

    def test_report_table_columns_follow_the_schema_not_the_printed_headers(self):
        """表头是印在纸上的中文（「交易金额」），字段键是英文（`amount`）：拿表头当键，一格都取不到。

        列序和中文名都从 schema 的数组字段路径来，并且只出这份资料里真有值的列。
        """
        with tempfile.TemporaryDirectory() as temp:
            out = self._workspace(Path(temp))
            html = Path(build_report(out, config=load_config())["path"]).read_text(encoding="utf-8")
            table = json.loads(re.search(
                r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S
                ).group(1).replace("<\\/", "</"))["documents"][0]["tables"][0]
            self.assertEqual(table["columns"], ["交易日期", "交易金额"])
            self.assertEqual([cell["value"] for cell in table["rows"][0]["cells"]],
                             ["2026-01-05", "88.00"])

    def test_export_zip_puts_report_at_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = self._workspace(root)
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            result = export_zip(out, config=config)
            self.assertTrue(Path(result["report"]["path"]).exists())
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
                guide = archive.read("怎么看.txt").decode("utf-8")
            self.assertIn("查看结果.html", names)
            self.assertIn("怎么看.txt", names)
            self.assertIn("查看结果.html", guide)
            self.assertTrue(all(not name.startswith("./") for name in names))


    def test_report_degrades_pages_to_fit_size_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = self._workspace(root)
            image = root / "扫描页.png"
            image.write_bytes(_noisy_png_bytes(2000))
            record = json.loads((out / "T1" / "T1.json").read_text(encoding="utf-8"))
            record["source"] = {"path": str(image), "file_name": "扫描页.png", "page_count": 1}
            (out / "T1" / "T1.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            full = build_report(out, dest=root / "full", config=config)
            self.assertEqual(full["page_level"], 0)
            self.assertEqual(full["embedded_pages"], 1)
            config["report"]["max_size_mb"] = 1
            tight = build_report(out, dest=root / "tight", config=config)
            self.assertGreater(tight["page_level"], 0)
            self.assertLessEqual(tight["size"], 1024 * 1024)
            self.assertLess(tight["size"], full["size"])
            html = Path(tight["path"]).read_text(encoding="utf-8")
            self.assertIn("单文件体积", html)
            self.assertIn("银行流水", html)


class ImportTests(unittest.TestCase):
    def _config(self, root: Path):
        config = copy.deepcopy(load_config())
        config["paths"]["inbox"] = str(root / "inbox")
        config["paths"]["user_out"] = str(root / "out" / "user")
        config["paths"]["work"] = str(root / "work")
        config["ui"]["max_upload_mb"] = 1
        return config

    def test_safe_name_blocks_path_traversal(self):
        self.assertEqual(safe_upload_name("../../evil.pdf", "pdf"), "evil.pdf")
        self.assertEqual(safe_upload_name('a:b*c.png', "png"), "a_b_c.png")
        self.assertEqual(safe_upload_name("", "jpg"), "document.jpg")

    def test_deduplicate_and_rename(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(Path(temp))
            first = b"%PDF-1.7\none"
            second = b"%PDF-1.7\ntwo"
            saved = store_upload(io.BytesIO(first), "资料.pdf", len(first), config)
            self.assertFalse(saved["duplicate"])
            duplicate = store_upload(io.BytesIO(first), "资料.pdf", len(first), config)
            self.assertTrue(duplicate["duplicate"])
            renamed = store_upload(io.BytesIO(second), "资料.pdf", len(second), config)
            self.assertFalse(renamed["duplicate"])
            self.assertEqual(renamed["name"], "资料-2.pdf")

    def test_image_signature_wins_over_claimed_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(Path(temp))
            png = _png_bytes()
            saved = store_upload(io.BytesIO(png), "照片.pdf", len(png), config)
            self.assertEqual(saved["suffix"], "png")
            self.assertEqual(saved["name"], "照片.png")
            self.assertTrue((Path(temp) / "inbox" / "照片.png").exists())
            jpeg = _jpeg_bytes()
            photo = store_upload(io.BytesIO(jpeg), "流水照片.jpeg", len(jpeg), config)
            self.assertEqual(photo["suffix"], "jpg")
            self.assertEqual(photo["name"], "流水照片.jpg")

    def test_reject_unknown_type_and_oversize(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(Path(temp))
            with self.assertRaises(ValueError):
                store_upload(io.BytesIO(b"not a document"), "x.pdf", 14, config)
            with self.assertRaises(ValueError):
                store_upload(io.BytesIO(b"%PDF-" + b"x" * (2 * 1024 * 1024)),
                             "big.pdf", 2 * 1024 * 1024 + 5, config)

    def test_iter_inputs_covers_images_and_config_whitelist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            (root / "a.pdf").write_bytes(b"%PDF-1.7\n")
            (root / "b.png").write_bytes(_png_bytes())
            (root / "c.txt").write_text("note", encoding="utf-8")
            found = {path.name for path in iter_inputs(root, config)}
            self.assertEqual(found, {"a.pdf", "b.png"})
            config["ingest"]["allowed_suffixes"] = ["pdf"]
            self.assertEqual([p.name for p in iter_inputs(root, config)], ["a.pdf"])

    def test_open_document_normalizes_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            image = root / "页面.png"
            image.write_bytes(_png_bytes(width=4000, height=300))
            document = open_document(image, config)
            try:
                self.assertEqual(document.page_count, 1)
            finally:
                document.close()
            cached = list((Path(config["paths"]["work"]) / "render" / "images").glob("*.png"))
            self.assertEqual(len(cached), 1)
            with Image.open(cached[0]) as normalized:
                self.assertEqual(max(normalized.size), config["ingest"]["image_max_side"])


    def test_multiframe_tiff_keeps_every_page_but_animation_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            frames = [Image.new("RGB", (320, 200), color) for color in
                      ((250, 250, 250), (200, 210, 220))]
            tiff = root / "扫描件.tif"
            frames[0].save(tiff, save_all=True, append_images=frames[1:], format="TIFF")
            document = open_document(tiff, config)
            try:
                self.assertEqual(document.page_count, 2)
            finally:
                document.close()
            self.assertEqual(len(list((Path(config["paths"]["work"]) / "render"
                                       / "images").glob("*.pdf"))), 1)
            gif = root / "动图.gif"
            frames[0].save(gif, save_all=True, append_images=frames[1:], format="GIF")
            document = open_document(gif, config)
            try:
                self.assertEqual(document.page_count, 1)
            finally:
                document.close()
            config["ingest"]["multiframe_suffixes"] = []
            document = open_document(tiff, config)
            try:
                self.assertEqual(document.page_count, 1)
            finally:
                document.close()


class ZipImportTests(unittest.TestCase):
    def _config(self, root: Path):
        config = copy.deepcopy(load_config())
        config["paths"]["inbox"] = str(root / "inbox")
        config["paths"]["user_out"] = str(root / "out" / "user")
        config["paths"]["work"] = str(root / "work")
        config["ui"]["max_upload_mb"] = 4
        return config

    @staticmethod
    def _zip(entries, compress: int = zipfile.ZIP_STORED) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compress) as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
        return buffer.getvalue()

    def test_unpacks_only_supported_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            first, second = _png_bytes(), _pdf_bytes("压缩包内页")
            blob = self._zip([("a/甲.png", first), ("b/乙.pdf", second),
                              ("说明.txt", b"hello"), ("empty.png", b"")])
            saved = store_upload(io.BytesIO(blob), "资料包.zip", len(blob), config)
            self.assertEqual(saved["suffix"], "zip")
            self.assertEqual(sorted(saved["extracted"]), ["乙.pdf", "甲.png"])
            self.assertEqual(saved["extracted_count"], 2)
            self.assertEqual(saved["skipped"], 2)
            self.assertEqual(sorted(p.name for p in (root / "inbox").iterdir()),
                             ["乙.pdf", "甲.png"])

    def test_member_paths_cannot_escape_the_inbox(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            payload = _png_bytes()
            blob = self._zip([("../../escape.png", payload),
                              ("..\\windows.png", payload),
                              ("/abs.png", _pdf_bytes("绝对路径"))])
            saved = store_upload(io.BytesIO(blob), "越界.zip", len(blob), config)
            self.assertEqual(sorted(saved["extracted"]), ["abs.pdf", "escape.png", "windows.png"])
            self.assertEqual(sorted(p.name for p in (root / "inbox").iterdir()),
                             ["abs.pdf", "escape.png", "windows.png"])
            self.assertFalse(list(root.glob("escape.png")))
            self.assertFalse(list(root.parent.glob("escape.png")))
            self.assertFalse(list(root.parent.glob("windows.png")))

    def test_archive_itself_never_stays_and_headers_are_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            blob = self._zip([("伪装.pdf.png", b"not a document at all"),
                              ("内层.zip", self._zip([("真.png", _png_bytes())]))])
            with self.assertRaises(ValueError) as caught:
                store_upload(io.BytesIO(blob), "假资料.zip", len(blob), config)
            self.assertIn("没有可导入的资料", str(caught.exception))
            self.assertFalse(list((root / "inbox").iterdir()))

    def test_ratio_bomb_and_entry_cap_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            bomb = self._zip([("炸弹.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * (3 * 1024 * 1024))],
                             zipfile.ZIP_DEFLATED)
            with self.assertRaises(ValueError) as caught:
                store_upload(io.BytesIO(bomb), "炸弹.zip", len(bomb), config)
            self.assertIn("疑似异常", str(caught.exception))
            self.assertFalse(list((root / "inbox").iterdir()))

            many = self._zip([(f"页{i}.png", _png_bytes()) for i in range(6)])
            config["ingest"]["zip_max_entries"] = 5
            with self.assertRaises(ValueError) as caught:
                store_upload(io.BytesIO(many), "太多.zip", len(many), config)
            self.assertIn("文件太多", str(caught.exception))

    def test_zip_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            config["ingest"]["zip_allowed"] = False
            blob = self._zip([("甲.png", _png_bytes())])
            with self.assertRaises(ValueError) as caught:
                store_upload(io.BytesIO(blob), "关掉.zip", len(blob), config)
            self.assertIn("zip_allowed", str(caught.exception))
            self.assertFalse(list((root / "inbox").iterdir()))

    def test_duplicate_members_reported_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            png = _png_bytes()
            store_upload(io.BytesIO(png), "甲.png", len(png), config)
            blob = self._zip([("甲.png", png), ("乙.png", _png_bytes(60, 40))])
            saved = store_upload(io.BytesIO(blob), "重复.zip", len(blob), config)
            self.assertEqual(saved["extracted"], ["乙.png"])
            self.assertEqual(saved["duplicates"], 1)
            self.assertEqual(len(list((root / "inbox").iterdir())), 2)

    def test_stray_archive_in_inbox_is_not_a_document(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            (root / "inbox").mkdir(parents=True)
            (root / "inbox" / "随手放的.zip").write_bytes(self._zip([("甲.png", _png_bytes())]))
            (root / "inbox" / "甲.png").write_bytes(_png_bytes())
            self.assertEqual([p.name for p in iter_inputs(root / "inbox", config)], ["甲.png"])


class IsolatedImportRegressionTests(unittest.TestCase):
    """只读配置与代码；原件、缓存、校准、台账和输出全部留在临时目录。"""

    COMPANY_FIELD = "bl.company_name"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="docpipe-import-regression-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = copy.deepcopy(load_config())
        for key in ("samples", "labels", "work", "out", "inbox", "user_out"):
            self.config["paths"][key] = str(self.root / key)
        self.inbox = Path(self.config["paths"]["inbox"])
        self.out = Path(self.config["paths"]["user_out"])
        self.work = Path(self.config["paths"]["work"])
        self.config["calibration"]["artifact"] = str(self.work / "calibration.json")
        self.config["calibration"]["review_dirs"] = [str(self.out)]
        self.config["ui"]["workspaces"] = [
            {"id": "isolated", "label": "隔离测试", "out": str(self.out)}]
        self.config["ui"]["trash_prune"] = False
        # 文件名含 .scan 也是真实文本层 PDF；不需要启动 OCR 或加载版面模型。
        self.config["parsing"]["ocr_engine"] = "null"
        self.config["parsing"]["layout"]["enabled"] = False
        self.inbox.mkdir()

    @staticmethod
    def _license_bytes(number):
        company = synth.make_company(random.Random(4100 + number), number)
        with fitz.open() as document:
            synth.build_business_license(document, company)
            return document.tobytes(garbage=4, deflate=True), company

    def _upload_license(self, name, number):
        data, company = self._license_bytes(number)
        saved = store_upload(io.BytesIO(data), name, len(data), self.config)
        self.assertEqual(saved["name"], name)
        self.assertFalse(saved["duplicate"])
        return company

    def _replace_license(self, name, number):
        data, company = self._license_bytes(number)
        (self.inbox / name).write_bytes(data)
        return company

    def _assert_result(self, result, success_names, failed_names=()):
        summary = result["summary"]
        self.assertEqual(summary["run"]["input_count"], len(success_names) + len(failed_names))
        self.assertEqual(summary["run"]["document_count"], len(success_names))
        self.assertEqual(summary["run"]["failed_count"], len(failed_names))
        self.assertEqual(result["failures"], summary["failures"])
        self.assertCountEqual([item["file_name"] for item in result["failures"]], failed_names)
        self.assertTrue(all(item["error"] for item in result["failures"]))
        self.assertEqual(json.loads((self.out / "summary.json").read_text(encoding="utf-8")),
                         summary)
        self.assertEqual(Path(result["out_dir"]).resolve(), self.out)
        self.assertCountEqual([item["source"] for item in summary["documents"]], success_names)
        documents, _, _ = load_final_documents(self.out, self.config)
        self.assertEqual(len(documents), len(success_names))
        self.assertEqual(len({doc["document_id"] for doc in documents}), len(success_names))
        self.assertCountEqual(result["document_ids"], [doc["document_id"] for doc in documents])
        self.assertTrue(set(result["document_ids"]).isdisjoint(
            item["document_id"] for item in result["failures"]))
        by_name = {doc["source"]["file_name"]: doc for doc in documents}
        self.assertCountEqual(by_name, success_names)
        for name, document in by_name.items():
            source = self.inbox / name
            self.assertEqual(Path(document["source"]["path"]).resolve(), source)
            self.assertEqual(document["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        with (self.out / "fields.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["document_id"] for row in rows}, set(result["document_ids"]))
        return by_name

    def _process(self, success_names, failed_names=()):
        return self._assert_result(process_inbox(self.config), success_names, failed_names)

    def _correct(self, document, value):
        from docpipe.serve import ReviewStore
        saved = ReviewStore(self.out, self.config).append_review(document["document_id"], {
            "field_path": self.COMPANY_FIELD, "action": "correct",
            "corrected_value": value, "reviewer": "isolated-test"})
        self.assertEqual(saved["record"]["source_sha256"], document["source"]["sha256"])
        self.assertEqual(saved["ledger_unread"], 0)
        return saved["record"]

    def _append_records(self, doc_id, records):
        with (self.out / "review" / f"{doc_id}.review.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _assert_company(self, document, company, correction=None):
        self.assertEqual(document["doc_type"], "BUSINESS_LICENSE")
        field = document["fields"][self.COMPANY_FIELD]
        self.assertEqual(field["value"], company["company_name"])
        self.assertEqual(field["effective_value"], correction or company["company_name"])
        self.assertEqual(field["review_status"], REVIEWED_CORRECTED if correction else None)

    def test_reviewed_scan_keeps_its_id_when_collisions_are_added_and_removed(self):
        scan_company = self._upload_license("a.scan.pdf", 1)
        first = self._process(["a.scan.pdf"])["a.scan.pdf"]
        self.assertEqual(first["document_id"], "a")
        self._assert_company(first, scan_company)
        correction = "仅属于扫描文件的人工修正"
        record = self._correct(first, correction)

        plain_company = self._upload_license("a.pdf", 2)
        documents = self._process(["a.pdf", "a.scan.pdf"])
        self.assertEqual(documents["a.scan.pdf"]["document_id"], first["document_id"])
        self._assert_company(documents["a.scan.pdf"], scan_company, correction)
        self._assert_company(documents["a.pdf"], plain_company)
        plain_id = documents["a.pdf"]["document_id"]

        suffix_company = self._upload_license("a_2.pdf", 3)
        documents = self._process(["a.pdf", "a.scan.pdf", "a_2.pdf"])
        self.assertEqual(documents["a.pdf"]["document_id"], plain_id)
        self.assertEqual(documents["a.scan.pdf"]["document_id"], first["document_id"])
        self._assert_company(documents["a.scan.pdf"], scan_company, correction)
        self._assert_company(documents["a.pdf"], plain_company)
        self._assert_company(documents["a_2.pdf"], suffix_company)
        remaining_ids = {name: documents[name]["document_id"] for name in ("a.scan.pdf", "a_2.pdf")}

        removed = remove_document(plain_id, self.config)
        remaining = self._assert_result(removed, ["a.scan.pdf", "a_2.pdf"])
        self.assertEqual(removed["remaining"], 2)
        self.assertEqual(removed["pruned"], [])
        self.assertTrue(Path(removed["trash"]).is_relative_to(self.work))
        self.assertTrue((Path(removed["trash"]) / "a.pdf").is_file())
        self.assertFalse((self.inbox / "a.pdf").exists())
        self.assertEqual({name: doc["document_id"] for name, doc in remaining.items()}, remaining_ids)
        self._assert_company(remaining["a.scan.pdf"], scan_company, correction)
        self._assert_company(remaining["a_2.pdf"], suffix_company)
        self.assertEqual(read_review_records(self.out, first["document_id"])[self.COMPANY_FIELD], record)

    def test_fresh_a_a_scan_and_a_2_do_not_collide(self):
        companies = {name: self._upload_license(name, number)
                     for number, name in enumerate(("a.pdf", "a.scan.pdf", "a_2.pdf"), 1)}
        documents = self._process(list(companies))
        self.assertEqual(documents["a_2.pdf"]["document_id"], "a_2")
        for name, company in companies.items():
            self._assert_company(documents[name], company)
        repeated = self._process(list(companies))
        self.assertEqual({name: doc["document_id"] for name, doc in repeated.items()},
                         {name: doc["document_id"] for name, doc in documents.items()})

    def test_legacy_review_follows_old_source_but_not_its_replacement(self):
        company = self._upload_license("a.scan.pdf", 1)
        old = self._process(["a.scan.pdf"])["a.scan.pdf"]
        correction = "无源哈希的历史人工修正"
        legacy = {"document_id": old["document_id"], "field_path": self.COMPANY_FIELD,
                  "action": "correct", "corrected_value": correction, "reviewer": "legacy"}
        self.assertNotIn("source_sha256", legacy)
        self._append_records(old["document_id"], [legacy])
        plain_company = self._upload_license("a.pdf", 2)
        documents = self._process(["a.pdf", "a.scan.pdf"])
        self.assertEqual(documents["a.scan.pdf"]["document_id"], old["document_id"])
        self._assert_company(documents["a.scan.pdf"], company, correction)
        self._assert_company(documents["a.pdf"], plain_company)
        plain_id = documents["a.pdf"]["document_id"]

        replacement = self._replace_license("a.scan.pdf", 3)
        documents = self._process(["a.pdf", "a.scan.pdf"])
        self.assertNotEqual(documents["a.scan.pdf"]["document_id"], old["document_id"])
        self.assertEqual(documents["a.pdf"]["document_id"], plain_id)
        self._assert_company(documents["a.scan.pdf"], replacement)
        self._assert_company(documents["a.pdf"], plain_company)
        ledger = read_review_ledger(self.out, old["document_id"])
        self.assertEqual(ledger["records"][self.COMPANY_FIELD], legacy)
        self.assertEqual(ledger["stale"], [])

    def test_reviewed_content_change_gets_a_new_stable_id_without_old_correction(self):
        self._upload_license("reviewed.pdf", 1)
        old = self._process(["reviewed.pdf"])["reviewed.pdf"]
        record = self._correct(old, "只对旧内容生效的修正")
        replacement = self._replace_license("reviewed.pdf", 2)
        current = self._process(["reviewed.pdf"])["reviewed.pdf"]
        self.assertNotEqual(current["source"]["sha256"], old["source"]["sha256"])
        self.assertNotEqual(current["document_id"], old["document_id"])
        self._assert_company(current, replacement)
        self.assertEqual(read_review_records(self.out, current["document_id"]), {})
        self.assertEqual(read_review_records(self.out, old["document_id"])[self.COMPANY_FIELD], record)
        repeated = self._process(["reviewed.pdf"])["reviewed.pdf"]
        self.assertEqual(repeated["document_id"], current["document_id"])
        self._assert_company(repeated, replacement)

    def test_hash_mismatch_is_visible_as_stale_and_never_changes_final_values(self):
        from docpipe.serve import ReviewStore
        company = self._upload_license("hash.pdf", 1)
        document = self._process(["hash.pdf"])["hash.pdf"]
        doc_id = document["document_id"]
        correction = "当前哈希的有效修正"
        valid = self._correct(document, correction)
        stale = [{**valid, "source_sha256": "0" * 64, "corrected_value": "过期企业名称"},
                 {**valid, "source_sha256": "0" * 64,
                  "field_path": "bl.legal_representative", "corrected_value": "过期法人"}]
        self.assertNotEqual(document["source"]["sha256"], "0" * 64)
        self._append_records(doc_id, stale)
        ledger = read_review_ledger(self.out, doc_id)
        self.assertEqual(ledger["records"], {self.COMPANY_FIELD: valid})
        self.assertEqual(ledger["all"], [valid])
        self.assertEqual(ledger["stale"], stale)
        self.assertEqual(ledger["unread"], [])
        self.assertEqual(ledger["lines"], 3)
        store = ReviewStore(self.out, self.config)
        self.assertEqual(store.ledger(doc_id)["stale"], stale)
        self.assertEqual(store.reviews(doc_id), [valid])
        final = self._process(["hash.pdf"])["hash.pdf"]
        self._assert_company(final, company, correction)
        representative = final["fields"]["bl.legal_representative"]
        self.assertEqual(representative["effective_value"], company["legal_rep"])
        self.assertIsNone(representative["review_status"])
        self.assertNotIn("review", representative)
        self.assertEqual(final["review_summary"]["reviewed_count"], 1)

    def test_bad_and_good_pdfs_can_repeat_then_clear_failures_after_repair(self):
        broken = b"%PDF-1.7\nisolated truncated document\n"
        store_upload(io.BytesIO(broken), "a_bad.pdf", len(broken), self.config)
        company = self._upload_license("z_good.pdf", 1)
        first = self._process(["z_good.pdf"], ["a_bad.pdf"])["z_good.pdf"]
        self._assert_company(first, company)
        repeated = self._process(["z_good.pdf"], ["a_bad.pdf"])["z_good.pdf"]
        self.assertEqual(repeated["document_id"], first["document_id"])
        self._assert_company(repeated, company)
        self.assertEqual((self.inbox / "a_bad.pdf").read_bytes(), broken)
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        failed_id = summary["failures"][0]["document_id"]
        self.assertFalse((self.out / failed_id / f"{failed_id}.json").exists())
        package = export_zip(self.out, config=self.config)
        self.assertEqual(package["failed_count"], 1)
        with zipfile.ZipFile(package["path"]) as archive:
            final = json.loads(archive.read("final/final_summary.json"))
            manifest = json.loads(archive.read("export_manifest.json"))
            self.assertEqual(final["export"]["failed_count"], 1)
            self.assertEqual(final["failures"], summary["failures"])
            self.assertEqual(manifest["failures"], summary["failures"])
            self.assertEqual(manifest["failed_count"], 1)
            self.assertIn("解析失败 1 份", archive.read("怎么看.txt").decode("utf-8-sig"))

        repaired_company = self._replace_license("a_bad.pdf", 2)
        repaired = self._process(["a_bad.pdf", "z_good.pdf"])
        self.assertEqual(repaired["z_good.pdf"]["document_id"], first["document_id"])
        self._assert_company(repaired["z_good.pdf"], company)
        self._assert_company(repaired["a_bad.pdf"], repaired_company)

    def test_all_bad_pdfs_return_zero_successes_and_keep_originals(self):
        originals = {"broken.pdf": b"%PDF-1.7\ntruncated one\n",
                     "other.pdf": b"%PDF-1.4\ntruncated two\n"}
        for name, data in originals.items():
            store_upload(io.BytesIO(data), name, len(data), self.config)
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                self.assertEqual(self._process([], list(originals)), {})
                for name, data in originals.items():
                    self.assertEqual((self.inbox / name).read_bytes(), data)
                self.assertEqual(list(self.out.glob("*/*.json")), [])

    def test_recoverable_parse_and_extract_errors_are_per_file_failures(self):
        from docpipe import run as pipeline
        self._upload_license("a_bad.pdf", 1)
        company = self._upload_license("z_good.pdf", 2)
        for stage in ("parse_document", "extract_document"):
            original = getattr(pipeline, stage)
            for error_type in (OSError, ValueError, RuntimeError):
                with self.subTest(stage=stage, error=error_type.__name__):
                    error = error_type("isolated per-file failure")

                    def fail_one(*args, **kwargs):
                        identity = args[0].name if stage == "parse_document" else args[0]["document_id"]
                        if identity in {"a_bad.pdf", "a_bad"}:
                            raise error
                        return original(*args, **kwargs)

                    with mock.patch.object(pipeline, stage, side_effect=fail_one):
                        result = process_inbox(self.config)
                    documents = self._assert_result(result, ["z_good.pdf"], ["a_bad.pdf"])
                    self._assert_company(documents["z_good.pdf"], company)
                    self.assertEqual(result["failures"][0]["error"], f"{error_type.__name__}: {error}")

    def test_type_error_and_keyboard_interrupt_are_not_swallowed(self):
        from docpipe import run as pipeline
        self._upload_license("programming-error.pdf", 1)
        for stage in ("parse_document", "extract_document"):
            for error_type in (TypeError, KeyboardInterrupt):
                with self.subTest(stage=stage, error=error_type.__name__):
                    error = error_type("must escape the batch")
                    with mock.patch.object(pipeline, stage, side_effect=error):
                        with self.assertRaises(error_type) as caught:
                            pipeline.run(self.inbox, self.out, config=self.config)
                    self.assertIs(caught.exception, error)
                    self.assertFalse((self.out / "summary.json").exists())
                    self.assertTrue((self.inbox / "programming-error.pdf").is_file())

    def test_cli_partial_failure_exits_one_and_successful_retry_does_not(self):
        from docpipe import run as pipeline
        (self.inbox / "a_bad.pdf").write_bytes(b"%PDF-1.7\nCLI broken fixture\n")
        company = self._upload_license("z_good.pdf", 1)
        argv = ["--input", str(self.inbox), "--out", str(self.out)]
        output = io.StringIO()
        # 只替换配置入口，CLI 的 run/解析/提取/写出保持真实，绝不走默认工作区。
        with mock.patch.object(pipeline, "load_config", return_value=self.config):
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    pipeline.main(argv)
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("a_bad.pdf", output.getvalue())
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run"]["input_count"], 2)
        self.assertEqual(summary["run"]["document_count"], 1)
        self.assertEqual(summary["run"]["failed_count"], 1)
        self.assertEqual([item["file_name"] for item in summary["failures"]], ["a_bad.pdf"])
        documents, _, _ = load_final_documents(self.out, self.config)
        self.assertEqual(len(documents), 1)
        self._assert_company(documents[0], company)

        self._replace_license("a_bad.pdf", 2)
        with mock.patch.object(pipeline, "load_config", return_value=self.config):
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.main(argv)
        summary = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run"]["document_count"], 2)
        self.assertEqual(summary["run"]["failed_count"], 0)
        self.assertEqual(summary["failures"], [])

    def test_zip_stored_names_include_duplicate_members_and_all_duplicate_retries(self):
        existing = _pdf_bytes("isolated existing ZIP member")
        new = _pdf_bytes("isolated new ZIP member")
        store_upload(io.BytesIO(existing), "existing.pdf", len(existing), self.config)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for name, data in (("existing.pdf", existing), ("nested/existing.pdf", existing),
                               ("new.pdf", new), ("ignored.txt", b"not a document")):
                archive.writestr(name, data)
        blob = buffer.getvalue()
        saved = store_upload(io.BytesIO(blob), "batch.zip", len(blob), self.config)
        self.assertEqual(saved["stored_names"], ["existing.pdf", "existing.pdf", "new.pdf"])
        self.assertEqual(saved["extracted"], ["new.pdf"])
        self.assertEqual(saved["extracted_count"], 1)
        self.assertEqual(saved["duplicates"], 2)
        self.assertEqual(saved["skipped"], 1)
        self.assertFalse(saved["duplicate"])
        repeated = store_upload(io.BytesIO(blob), "batch.zip", len(blob), self.config)
        self.assertEqual(repeated["stored_names"], saved["stored_names"])
        self.assertEqual(repeated["extracted"], [])
        self.assertEqual(repeated["extracted_count"], 0)
        self.assertEqual(repeated["duplicates"], 3)
        self.assertTrue(repeated["duplicate"])
        self.assertEqual({path.name: path.read_bytes() for path in self.inbox.iterdir()},
                         {"existing.pdf": existing, "new.pdf": new})

    def test_unreviewed_manifest_document_keeps_explicit_id_after_content_change(self):
        self._upload_license("generated.pdf", 1)
        manifest_id = "SYNTH_KEEP_001"
        (self.inbox.parent / "manifest.json").write_text(json.dumps({"documents": [
            {"document_id": manifest_id, "path": str(self.inbox / "generated.pdf")}
        ]}), encoding="utf-8")
        first = self._process(["generated.pdf"])["generated.pdf"]
        self.assertEqual(first["document_id"], manifest_id)
        # run 自动创建的空台账不等于已复核；不能靠删空文件才保住 manifest 编号。
        self.assertEqual(read_review_ledger(self.out, manifest_id)["lines"], 0)
        replacement = self._replace_license("generated.pdf", 2)
        changed = self._process(["generated.pdf"])["generated.pdf"]
        self.assertNotEqual(changed["source"]["sha256"], first["source"]["sha256"])
        self._assert_company(changed, replacement)
        self.assertEqual(changed["document_id"], manifest_id,
                         "未复核的显式 manifest 文档重生成后必须保留原 ID")


class ContinuationTableTests(unittest.TestCase):
    def test_headerless_continuation_inherits_columns(self):
        config = load_config()
        headers = ["交易日期", "摘要", "对方户名", "对方账号", "收支方向",
                   "交易金额", "余额", "交易流水号"]
        centers = [68, 129, 196, 278, 338, 388, 453, 521]
        previous = {
            "table_id": "p1-transactions",
            "name": "transactions",
            "page": 1,
            "bbox": [30, 200, 560, 810],
            "headers": headers,
            "columns": [{"header": header, "center": center, "radius": 55}
                        for header, center in zip(headers, centers)],
            "rows": [],
        }

        def segment(text, left, right, top):
            return {"text": text, "bbox": [left, top, right, top + 10],
                    "confidence": 0.99}

        segments = [
            segment("企业对公账户交易明细（续）", 150, 450, 40),
            segment("2025-07-03", 35, 101, 120),
            segment("货款", 101, 157, 120),
            segment("示例供应商", 157, 235, 120),
            segment("6222000011112222", 235, 321, 120),
            segment("收入", 321, 355, 120),
            segment("1,000.00", 355, 421, 120),
            segment("11,000.00", 421, 485, 120),
            segment("SN0003", 485, 557, 120),
        ]
        table = _continued_table(segments, previous, "BANK_STATEMENT", 2, config, 842)
        self.assertIsNotNone(table)
        self.assertEqual(table["continued_from"], "p1-transactions")
        self.assertEqual(table["engine"], "continuation")
        self.assertEqual(len(table["rows"]), 1)
        cells = table["rows"][0]["cells"]
        self.assertEqual(cells["交易日期"]["text"], "2025-07-03")
        self.assertEqual(cells["交易金额"]["text"], "1,000.00")
        self.assertEqual(cells["交易流水号"]["text"], "SN0003")

    def test_summary_page_does_not_become_rows(self):
        config = load_config()
        headers = ["交易日期", "摘要", "对方户名", "对方账号", "收支方向",
                   "交易金额", "余额", "交易流水号"]
        centers = [68, 129, 196, 278, 338, 388, 453, 521]
        previous = {
            "table_id": "p1-transactions", "name": "transactions", "page": 1,
            "bbox": [30, 200, 560, 810], "headers": headers,
            "columns": [{"header": header, "center": center, "radius": 55}
                        for header, center in zip(headers, centers)],
            "rows": [],
        }
        segments = [{"text": "期末余额", "bbox": [50, 120, 150, 130], "confidence": 0.99},
                    {"text": "11,000.00", "bbox": [160, 120, 260, 130], "confidence": 0.99}]
        self.assertIsNone(_continued_table(segments, previous, "BANK_STATEMENT", 3,
                                           config, 842))


class CalibrationGroupingTests(unittest.TestCase):
    def test_field_auto_falls_back_to_doc_type(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "calibration.json"
            artifact.write_text(json.dumps({
                "coef": [0.0], "intercept": 0.0,
                "sample_grouping": "field_auto",
                "field_samples": {"bs.transactions[i].amount": 50,
                                  "bs.account_name": 5},
                "group_samples": {"BANK_STATEMENT": 500},
            }), encoding="utf-8")
            config = copy.deepcopy(load_config())
            config["calibration"]["artifact"] = str(artifact)
            calibrator = Calibrator(config)
            self.assertEqual(calibrator.samples_for(
                {"path": "bs.transactions[3].amount"}, "BANK_STATEMENT"), 50)
            self.assertEqual(calibrator.samples_for(
                {"path": "bs.account_name"}, "BANK_STATEMENT"), 500)


class DecisionTests(unittest.TestCase):
    def _inputs(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        field = {
            "path": "bs.account_name", "label": "户名", "value": "示例公司",
            "value_type": "string", "required": True, "risk": "party_name",
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "grounding_present": 1},
            "meta": {"sign_source": "direction"},
            "decision": "review", "validators": [], "review_reasons": [],
        }
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 200, "height": 200}]}
        doc = {"document_id": "T1", "doc_type": "BANK_STATEMENT"}
        return config, field, parsed, doc

    def test_sign_inferred_forces_review(self):
        config, field, parsed, doc = self._inputs()
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertEqual(result["decision"], "review")
        self.assertIn("SIGN_INFERRED", result["review_reasons"])

    def test_reason_can_be_disabled_by_config(self):
        config, field, parsed, doc = self._inputs()
        config["hard_abstain_reasons"] = [
            reason for reason in config["hard_abstain_reasons"] if reason != "SIGN_INFERRED"
        ]
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["review_reasons"], [])

    def test_every_hard_reason_has_a_ui_label(self):
        """写进 hard_abstain_reasons 的原因码都必须有界面中文文案。

        P0-8 漏过一次：MATRIX_FALLBACK 进了硬性原因，界面只显示裸代码。原因码是给
        复核的人读的，不是给机器读的，所以这条不靠人记得住。
        """
        config = load_config()
        viewer = (Path(__file__).resolve().parents[1] / "docpipe" / "web" / "app.js"
                  ).read_text(encoding="utf-8")
        block = re.search(r"const REASON_LABELS = \{(.*?)\n\};", viewer, re.S)
        self.assertIsNotNone(block, "REASON_LABELS 结构变了，同步改这条检查")
        labelled = set(re.findall(r"^\s*([A-Z][A-Z0-9_]+):", block.group(1), re.M))
        for reason in config["hard_abstain_reasons"]:
            self.assertIn(reason, labelled, reason)

    def test_seal_overlap_forces_review(self):
        config, field, parsed, doc = self._inputs()
        field["meta"] = {}
        parsed["pages"][0]["preprocess"] = {
            "enhancement": {"seal_regions": [[0.04, 0.04, 0.5, 0.2]]}}
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertEqual(result["decision"], "review")
        self.assertIn("SEAL_OVERLAP", result["review_reasons"])
        # A seal elsewhere on the page must not drag every field into review.
        parsed["pages"][0]["preprocess"]["enhancement"] = {"seal_regions": [[0.7, 0.7, 0.95, 0.95]]}
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertNotIn("SEAL_OVERLAP", result["review_reasons"])
        parsed["pages"][0]["preprocess"]["enhancement"] = {"seal_regions": [[0.04, 0.04, 0.5, 0.2]]}
        config["validation"]["review_on_seal_overlap"] = False
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertNotIn("SEAL_OVERLAP", result["review_reasons"])

    def test_confidence_records_where_it_came_from(self):
        config, field, parsed, doc = self._inputs()
        result = decide_field(field, doc, parsed, config, [], Calibrator(config))
        self.assertEqual(result["confidence_basis"], "heuristic")
        self.assertEqual(result["calibration_samples"], 0)
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "calibration.json"
            artifact.write_text(json.dumps({
                "coef": [0.0], "intercept": 4.0, "sample_grouping": "field_auto",
                "field_samples": {"bs.account_name": 3}, "group_samples": {"BANK_STATEMENT": 3},
            }), encoding="utf-8")
            thin = copy.deepcopy(config)
            thin["calibration"]["artifact"] = str(artifact)
            result = decide_field(field, doc, parsed, thin, [], Calibrator(thin))
            self.assertEqual(result["confidence_basis"], "calibrated_low_samples")
            self.assertLessEqual(result["confidence"],
                                 float(thin["calibration"]["low_sample_confidence_cap"]))
            rich = copy.deepcopy(config)
            rich["calibration"]["artifact"] = str(artifact)
            rich["calibration"]["min_samples_per_field"] = 1
            artifact.write_text(json.dumps({
                "coef": [0.0], "intercept": 4.0, "sample_grouping": "field_auto",
                "field_samples": {"bs.account_name": 500}, "group_samples": {"BANK_STATEMENT": 500},
            }), encoding="utf-8")
            result = decide_field(field, doc, parsed, rich, [], Calibrator(rich))
            self.assertEqual(result["confidence_basis"], "calibrated")
            self.assertEqual(result["calibration_samples"], 500)

    def test_error_categories(self):
        self.assertEqual(error_category({"pred": None, "reasons": []}), "MISSING")
        self.assertEqual(error_category({"pred": "x", "reasons": ["LOW_OCR"],
                                         "signals": {"ocr_conf": 0.4}}), "OCR")
        self.assertEqual(error_category({"pred": "x", "reasons": [],
                                         "extractor_id": "table_cell:金额",
                                         "signals": {"ocr_conf": 0.99,
                                                     "table_score": 0.2}}), "TABLE_ALIGNMENT")
        self.assertEqual(error_category({"pred": "x", "reasons": [],
                                         "signals": {"ocr_conf": 0.99,
                                                     "rule_exact": 0.5}}), "RULE_MISMATCH")
        self.assertEqual(error_category({"pred": "x", "reasons": [],
                                         "extractor_id": "table_cell:金额",
                                         "signals": {"table_score": 0.2}}), "OCR")


class PreprocessTests(unittest.TestCase):
    def test_deskew_straightens_a_rotated_photo(self):
        config = load_config()
        options = preprocess.geometry_options(config)
        fixed, record = preprocess.rectify(_rotate_page(_text_page(), 7), config)
        self.assertAlmostEqual(record["deskew_degrees"], -7.0, delta=0.6)
        self.assertFalse(record["uncorrected_skew"])
        self.assertNotIn("uncertain_skew", record)
        residual, _contrast, _gain = preprocess.projection_skew(_analysis_binary(fixed, options), options)
        self.assertAlmostEqual(residual or 0.0, 0.0, delta=0.6)

    def test_skew_beyond_the_cap_is_marked_not_half_fixed(self):
        config = load_config()
        _fixed, record = preprocess.rectify(_rotate_page(_text_page(), 20), config)
        self.assertEqual(record["deskew_degrees"], 0.0)
        self.assertTrue(record["uncorrected_skew"])
        self.assertIn("PHOTO_SKEW_UNCORRECTED", preprocess.quality_flags(_fixed, record, config))
        # The cap is a knob, not a constant in the code.
        tighter = copy.deepcopy(config)
        tighter["preprocess"]["geometry"]["max_deskew_degrees"] = 5
        _fixed2, record2 = preprocess.rectify(_rotate_page(_text_page(), 7), tighter)
        self.assertTrue(record2["uncorrected_skew"])
        self.assertEqual(record2["deskew_degrees"], 0.0)

    def test_weak_projection_gain_blocks_rotation(self):
        config = copy.deepcopy(load_config())
        config["preprocess"]["geometry"]["min_skew_gain"] = 1000.0
        _fixed, record = preprocess.rectify(_rotate_page(_text_page(), 7), config)
        self.assertEqual(record["deskew_degrees"], 0.0)
        self.assertTrue(record["uncertain_skew"])

    def test_perspective_page_is_flattened_without_being_rotated(self):
        config = load_config()
        shot = _perspective_shot([[300, 100], [700, 100], [900, 1200], [100, 1200]])
        fixed, record = preprocess.rectify(shot, config)
        self.assertTrue(record["warped"])
        self.assertFalse(record["partial"])
        self.assertEqual(record["deskew_degrees"], 0.0)
        smaller = copy.deepcopy(config)
        smaller["preprocess"]["geometry"]["warp_output_scale"] = 1.0
        small_fixed, small_record = preprocess.rectify(shot, smaller)
        self.assertTrue(small_record["warped"])
        # Scaling the output up keeps the small glyphs readable after resampling.
        self.assertGreater(fixed.shape[1], small_fixed.shape[1])

    def test_page_cut_off_by_the_frame_is_flagged(self):
        config = load_config()
        shot = _perspective_shot([[6, 6], [760, 20], [900, 1280], [100, 1150]])
        _fixed, record = preprocess.rectify(shot, config)
        self.assertTrue(record["partial"])
        self.assertFalse(record["warped"])
        self.assertIn("PHOTO_PARTIAL", preprocess.quality_flags(_fixed, record, config))

    def test_quality_flags_only_fire_on_bad_photos(self):
        config = load_config()
        page = _text_page()
        self.assertEqual(preprocess.quality_flags(page, {}, config), [])
        self.assertIn("PHOTO_BLUR",
                      preprocess.quality_flags(cv2.GaussianBlur(page, (0, 0), 6), {}, config))
        self.assertEqual(preprocess.quality_flags((page * 0.45).astype(np.uint8), {}, config),
                         ["PHOTO_DARK"])
        washed = np.clip(page.astype(np.int16) - 40, 170, 215).astype(np.uint8)
        self.assertEqual(preprocess.quality_flags(washed, {}, config), ["PHOTO_WASHED"])

    def test_ocr_confidence_floor_flags_low_quality_ocr(self):
        config = load_config()
        self.assertEqual(preprocess.flag_from_ocr(0.5, config), "PHOTO_LOW_OCR")
        self.assertIsNone(preprocess.flag_from_ocr(0.9, config))
        self.assertIsNone(preprocess.flag_from_ocr(None, config))
        config["preprocess"]["quality"]["min_ocr_mean_confidence"] = 0
        self.assertIsNone(preprocess.flag_from_ocr(0.1, config))

    def test_seal_suppression_clears_the_stamp_but_keeps_the_ink(self):
        config = load_config()
        for name, color in (("red", (0, 0, 200)), ("blue", (200, 0, 0))):
            bgr = cv2.cvtColor(_text_page(), cv2.COLOR_RGB2BGR).copy()
            cv2.circle(bgr, (650, 800), 150, color, 14)
            self.assertEqual(bgr[800, 800].tolist(), list(color))
            out, info = preprocess.ocr_input(bgr, config)
            self.assertTrue(info["seal_suppressed"], name)
            # The stamp itself reads as paper, the printed ink underneath stays ink.
            self.assertTrue(all(channel >= 200 for channel in out[800, 800]), name)
            self.assertEqual(out[597, 450].tolist(), [20, 20, 20], name)
            self.assertEqual(len(info["seal_regions"]), 1, name)
            box = info["seal_regions"][0]
            self.assertEqual(len(box), 4, name)
            self.assertTrue(0.0 <= min(box) and max(box) <= 1.0, name)
            self.assertFalse(info["seal_edge_fragment"], name)

    def test_edge_clipped_seal_fragment_is_detected(self):
        # 骑缝章只有四分之一入画，整页彩色占比落到主门之下：光线从主门挡掉，
        # 残片必须靠「贴边」这一条被认出来，否则跨页件的章整份静默失守。
        config = load_config()
        options = preprocess.enhance_options(config)
        bgr = cv2.cvtColor(_text_page(), cv2.COLOR_RGB2BGR).copy()
        cv2.circle(bgr, (0, 0), 140, (0, 0, 200), 16)
        out, info = preprocess.ocr_input(bgr, config)
        self.assertLess(info["colored_ratio"], options["min_colored_ratio"])
        self.assertGreaterEqual(info["colored_ratio"], options["edge_seal_min_colored_ratio"])
        self.assertTrue(info["seal_suppressed"])
        self.assertTrue(info["seal_edge_fragment"])
        self.assertEqual(len(info["seal_regions"]), 1)
        x0, y0, x1, y1 = info["seal_regions"][0]
        self.assertLessEqual(x0, 0.01)
        self.assertLessEqual(y0, 0.01)
        self.assertGreater(x1, 0.01)
        # 残片本身读成纸白，离章足够远的墨一个像素都不许动。
        self.assertTrue(all(channel >= 200 for channel in out[99, 99]))
        self.assertEqual(out[597, 450].tolist(), [20, 20, 20])
        # 「贴边门」也是配置项：抬到残片占比之上，同一张图就不该再动手。
        stricter = copy.deepcopy(config)
        stricter["preprocess"]["enhance"]["edge_seal_min_colored_ratio"] = 0.05
        _out, info2 = preprocess.ocr_input(bgr, stricter)
        self.assertFalse(info2["seal_suppressed"])
        self.assertFalse(info2["seal_edge_fragment"])

    def test_edge_gate_does_not_promote_interior_blobs_below_main_gate(self):
        # 主门之下的内部彩色小块（图标、logo）不是章：新门只认贴边色块，
        # 否则等于把主门悄悄降到了 0.0015。
        config = load_config()
        options = preprocess.enhance_options(config)
        bgr = cv2.cvtColor(_text_page(), cv2.COLOR_RGB2BGR).copy()
        cv2.circle(bgr, (450, 600), 70, (0, 0, 200), 6)
        out, info = preprocess.ocr_input(bgr, config)
        self.assertLess(info["colored_ratio"], options["min_colored_ratio"])
        self.assertGreaterEqual(info["colored_ratio"], options["edge_seal_min_colored_ratio"])
        # 这枚小块单看形状/面积完全够格当章——把它拒掉的只能是不贴边这条新判据。
        analysis = preprocess._shrunk(bgr, int(preprocess.geometry_options(config)["analyze_max_side"]))
        _mask, without_edge_rule = preprocess._colored_regions(analysis, options)
        self.assertEqual(len(without_edge_rule), 1)
        self.assertFalse(info["seal_suppressed"])
        self.assertFalse(info["seal_edge_fragment"])
        self.assertNotIn("seal_regions", info)
        self.assertTrue(np.array_equal(out, bgr))

    def test_edge_only_colored_regions_keeps_only_edge_touching_blobs(self):
        config = load_config()
        options = preprocess.enhance_options(config)
        page = np.full((900, 900, 3), 255, dtype=np.uint8)
        cv2.circle(page, (0, 900), 120, (0, 0, 200), 12)    # 贴左下角的残片
        cv2.circle(page, (450, 450), 120, (0, 0, 200), 12)  # 页内一枚完整章
        _mask, full_regions = preprocess._colored_regions(page, options)
        edge_mask, edge_regions = preprocess._colored_regions(page, options, edge_only=True)
        self.assertEqual(len(full_regions), 2)
        self.assertEqual(len(edge_regions), 1)
        x0, y0, x1, y1 = edge_regions[0]
        self.assertGreaterEqual(y1, 0.99)
        self.assertLess(x1 - x0, 0.5)
        # 抑制掩码只盖贴边那枚，页内完整章不被牵连。
        self.assertGreater(int(edge_mask[780, 0]), 0)
        self.assertEqual(int(edge_mask[570, 450]), 0)

    def test_glare_band_is_stretched_back_into_readable_ink(self):
        config = load_config()
        bgr = cv2.cvtColor(_glare_shot(_text_page()), cv2.COLOR_RGB2BGR)
        before = preprocess._gray(bgr)
        out, info = preprocess.ocr_input(bgr, config)
        self.assertTrue(info["highlight_suppressed"])
        self.assertGreaterEqual(info["highlight_area"], 0.01)
        self.assertLessEqual(info["highlight_area"], 0.55)
        self.assertGreaterEqual(info["highlight_gain"],
                                preprocess.enhance_options(config)["highlight_min_gain"])
        band = _in_band(before.shape)
        # 带子里原来只剩一层浅灰，压制后必须重新有「墨」——这是这个功能唯一的存在理由。
        washed = float(np.percentile(before[band], 1))
        rescued = float(np.percentile(preprocess._gray(out)[band], 1))
        self.assertGreater(washed, 180.0)
        self.assertLess(rescued, washed - 60.0)
        # 带外不许被牵连：离反光足够远的字，一个像素都不该改。
        far = np.zeros(before.shape, dtype=bool)
        far[:, int(before.shape[1] * 0.62):] = True
        self.assertTrue(np.array_equal(out[far], bgr[far]))

    def test_glare_suppression_leaves_an_evenly_lit_page_untouched(self):
        config = load_config()
        bgr = cv2.cvtColor(_uneven_page(_text_page()), cv2.COLOR_RGB2BGR)
        out, info = preprocess.ocr_input(bgr, config)
        self.assertFalse(info["highlight_suppressed"])
        self.assertNotIn("highlight_area", info)
        self.assertTrue(np.array_equal(out, bgr))

    def test_evenly_washed_page_is_a_retake_not_a_band(self):
        config = load_config()
        washed = np.clip(_text_page().astype(np.int16) - 40, 175, 215).astype(np.uint8)
        _out, info = preprocess.ocr_input(cv2.cvtColor(washed, cv2.COLOR_RGB2BGR), config)
        self.assertFalse(info["highlight_suppressed"])
        self.assertIn("PHOTO_WASHED", preprocess.quality_flags(washed, {}, config))

    def test_glare_suppression_knobs_are_honoured(self):
        bgr = cv2.cvtColor(_glare_shot(_text_page()), cv2.COLOR_RGB2BGR)
        off = copy.deepcopy(load_config())
        off["preprocess"]["enhance"]["highlight_suppression"] = False
        untouched, info = preprocess.ocr_input(bgr, off)
        self.assertFalse(info["highlight_suppressed"])
        self.assertTrue(np.array_equal(untouched, bgr))
        # 把「值得救」的门槛抬高，同一张图就不该再动手：阈值在配置里，不是写死的。
        stricter = copy.deepcopy(load_config())
        stricter["preprocess"]["enhance"]["highlight_min_gain"] = 50.0
        self.assertFalse(preprocess.ocr_input(bgr, stricter)[1]["highlight_suppressed"])

    def test_finger_over_the_corner_is_marked_not_repainted(self):
        config = load_config()
        shot = _finger_shot(_text_page())
        before = shot.copy()
        flags = preprocess.quality_flags(shot, {}, config)
        self.assertIn("PHOTO_OCCLUSION", flags)
        # 遮挡只标记：压住的那块像素本来就没有信息，猜出来比留白更危险，也不许动原图。
        self.assertTrue(np.array_equal(shot, before))
        # 圆章不是手指：同样是彩色块，长条这条不过。
        stamp = _uneven_page(_text_page()).copy()
        cv2.circle(stamp, (650, 800), 150, (200, 0, 0), 14)
        self.assertNotIn("PHOTO_OCCLUSION", preprocess.quality_flags(stamp, {}, config))
        quieter = copy.deepcopy(config)
        quieter["preprocess"]["quality"]["occlusion_check"] = False
        self.assertNotIn("PHOTO_OCCLUSION", preprocess.quality_flags(shot, {}, quieter))

    def test_photo_labels_are_not_free_to_drift_between_places(self):
        root = Path(__file__).resolve().parents[1]
        viewer = (root / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
        for flag in preprocess.FLAG_LABELS:
            self.assertIn(flag, viewer, flag)
        # 报表那边不再抄一份：标签由 FLAG_LABELS 注入，模板里只留占位符。
        from docpipe.report import TEMPLATE

        self.assertIn("const PHOTO_LABELS = __PHOTO_LABELS__;", TEMPLATE)

    def test_dark_page_is_lifted_and_says_which_lift_ran(self):
        config = load_config()
        bgr = cv2.cvtColor(np.clip(_text_page().astype(np.float32) * 0.4 + 20, 0, 255)
                           .astype(np.uint8), cv2.COLOR_RGB2BGR)
        out, info = preprocess.ocr_input(bgr, config)
        # 两道门都该过：整页压暗既让中位亮度掉出门槛，也让纸墨差压到线下。
        self.assertTrue(info["brightness_lifted"])
        self.assertTrue(info["contrast_lifted"])
        # 旗标说动了手，像素就得真的变；只挂旗标不改写、或改写却不挂旗标，都是复核台的假提示。
        self.assertFalse(np.array_equal(out, bgr))
        # 一道门各自关掉，只有它自己的旗标落回 False，另一道照旧——提示跟着实际动作走。
        for key, flag, other in (("brightness_lift", "brightness_lifted", "contrast_lifted"),
                                 ("contrast_lift", "contrast_lifted", "brightness_lifted")):
            quieter = copy.deepcopy(config)
            quieter["preprocess"]["enhance"][key] = False
            lifted = preprocess.ocr_input(bgr.copy(), quieter)[1]
            self.assertFalse(lifted[flag], key)
            self.assertTrue(lifted[other], key)

    def test_enhancement_is_skipped_without_seals_and_when_disabled(self):
        config = load_config()
        _out, info = preprocess.ocr_input(cv2.cvtColor(_text_page(), cv2.COLOR_RGB2BGR), config)
        self.assertFalse(info["seal_suppressed"])
        self.assertNotIn("seal_regions", info)
        self.assertFalse(info["contrast_lifted"])
        self.assertFalse(info["brightness_lifted"])
        off = copy.deepcopy(config)
        off["preprocess"]["enhance"]["seal_suppression"] = False
        bgr = cv2.cvtColor(_text_page(), cv2.COLOR_RGB2BGR).copy()
        cv2.circle(bgr, (650, 800), 150, (0, 0, 200), 14)
        untouched, info = preprocess.ocr_input(bgr, off)
        self.assertFalse(info["seal_suppressed"])
        self.assertEqual(untouched[800, 800].tolist(), [0, 0, 200])
        disabled = copy.deepcopy(config)
        disabled["preprocess"]["enabled"] = False
        self.assertEqual(preprocess.rectify(bgr, disabled)[1]["deskew_degrees"], 0.0)
        self.assertFalse(preprocess.ocr_input(bgr, disabled)[1]["seal_suppressed"])


class RemoveDocumentTests(unittest.TestCase):
    def _config(self, root: Path):
        config = copy.deepcopy(load_config())
        config["paths"]["inbox"] = str(root / "inbox")
        config["paths"]["user_out"] = str(root / "out" / "user")
        config["paths"]["work"] = str(root / "work")
        return config

    def _import(self, root: Path, names: list) -> dict:
        config = self._config(root)
        inbox = root / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        for name in names:
            (inbox / f"{name}.pdf").write_bytes(
                _pdf_bytes(f"DOCUMENT {name} " + "content words for the text layer " * 3))
        return config, process_inbox(config)

    def test_removal_retires_files_and_rebuilds_the_rest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self._import(root, ["one", "two"])
            out = root / "out" / "user"
            (out / "review" / "one.review.jsonl").write_text(
                json.dumps({"field_path": "x.y", "action": "accept"}) + "\n", encoding="utf-8")
            self.assertTrue((out / "one" / "one.json").exists())

            result = remove_document("one", config)
            self.assertEqual(result["remaining"], 1)
            self.assertEqual(result["document_ids"], ["two"])
            trash = Path(result["trash"])
            self.assertTrue((root / "work" / "trash").is_dir())
            self.assertTrue((trash / "one.pdf").exists())
            self.assertTrue((trash / "one.review.jsonl").exists())
            self.assertTrue((trash / "one" / "one.json").exists())
            self.assertTrue((root / "work" / "parsed" / "one.json").exists() is False)
            self.assertFalse((out / "one").exists())
            self.assertFalse((root / "inbox" / "one.pdf").exists())
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([item["document_id"] for item in summary["documents"]], ["two"])

    def test_removing_the_last_document_empties_the_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self._import(root, ["only"])
            out = root / "out" / "user"
            (out / "final").mkdir(parents=True, exist_ok=True)
            (out / "final" / "查看结果.html").write_text("<html>old</html>", encoding="utf-8")
            result = remove_document("only", config)
            self.assertEqual(result["remaining"], 0)
            self.assertFalse((out / "summary.json").exists())
            self.assertFalse((out / "final" / "查看结果.html").exists())
            self.assertTrue((Path(result["trash"]) / "final" / "查看结果.html").exists())
            self.assertTrue((root / "inbox" / "only.pdf").exists() is False)

    def test_source_outside_the_inbox_is_left_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            (root / "inbox").mkdir(parents=True)
            outside = root / "elsewhere"
            outside.mkdir()
            source = outside / "keepme.pdf"
            source.write_bytes(_pdf_bytes("NOT IMPORTED " + "text layer content here " * 3))
            document = root / "out" / "user" / "keepme"
            document.mkdir(parents=True)
            (document / "keepme.json").write_text(json.dumps(
                {"source": {"path": str(source), "sha256": "0" * 64}}), encoding="utf-8")
            result = remove_document("keepme", config)
            self.assertEqual(result["remaining"], 0)
            self.assertTrue(source.exists())
            self.assertFalse(document.exists())
            self.assertTrue(any(Path(result["trash"]).iterdir()))

    def test_rejects_unsafe_document_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self._config(Path(temp))
            for bad in ("", "../one", "a/b", "a\\b", "..", "x\x00y"):
                with self.assertRaises(ValueError):
                    remove_document(bad, config)


class TrashRetentionTests(unittest.TestCase):
    def _config(self, root: Path):
        config = copy.deepcopy(load_config())
        config["paths"]["inbox"] = str(root / "inbox")
        config["paths"]["user_out"] = str(root / "out" / "user")
        config["paths"]["work"] = str(root / "work")
        return config

    @staticmethod
    def _stamp(days_ago: float) -> str:
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() - days_ago * 86400))

    def _runs(self, root: Path, names) -> Path:
        trash = root / "work" / "trash"
        for name in names:
            (trash / name).mkdir(parents=True, exist_ok=True)
            (trash / name / "原件.png").write_bytes(b"x")
        return trash

    def test_keeps_newest_runs_until_the_ttl_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            fresh = [self._stamp(0), self._stamp(0.01)]
            old = [self._stamp(31), self._stamp(40), self._stamp(60)]
            trash = self._runs(root, sorted(fresh + old))
            config["ui"]["trash_keep_runs"] = 5
            config["ui"]["trash_keep_days"] = 30
            purged = prune_trash(config)
            self.assertEqual(sorted(purged), sorted(old))
            self.assertEqual(sorted(p.name for p in trash.iterdir()), sorted(fresh))

    def test_run_cap_purges_even_recent_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            stamps = sorted(self._stamp(index * 0.001) for index in range(5))
            trash = self._runs(root, stamps)
            config["ui"]["trash_keep_runs"] = 2
            config["ui"]["trash_keep_days"] = 365
            purged = prune_trash(config)
            self.assertEqual(sorted(purged), sorted(stamps[:-2]))
            self.assertEqual(sorted(p.name for p in trash.iterdir()), sorted(stamps[-2:]))

    def test_only_stamp_directories_are_touched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            stale = self._stamp(99)
            trash = self._runs(root, [stale])
            (trash / "我自己放的").mkdir()
            (trash / "我自己放的" / "留着.txt").write_text("keep", encoding="utf-8")
            (trash / "随手.txt").write_text("keep", encoding="utf-8")
            config["ui"]["trash_keep_runs"] = 0
            config["ui"]["trash_keep_days"] = 0
            purged = prune_trash(config)
            self.assertEqual(purged, [stale])
            self.assertTrue((trash / "我自己放的" / "留着.txt").exists())
            self.assertTrue((trash / "随手.txt").exists())

    def test_pruning_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            stale = self._stamp(900)
            trash = self._runs(root, [stale])
            config["ui"]["trash_prune"] = False
            self.assertEqual(prune_trash(config), [])
            self.assertTrue((trash / stale / "原件.png").exists())

    def test_removal_reports_and_applies_the_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self._import_two(root)
            stale = self._stamp(45)
            trash = self._runs(root, [stale])
            config["ui"]["trash_keep_runs"] = 10
            config["ui"]["trash_keep_days"] = 30
            result = remove_document("one", config)
            self.assertEqual(result["pruned"], [stale])
            self.assertFalse((trash / stale).exists())
            self.assertTrue(Path(result["trash"]).is_dir())

    def _import_two(self, root: Path):
        config = self._config(root)
        inbox = root / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        for name in ("one", "two"):
            (inbox / f"{name}.pdf").write_bytes(
                _pdf_bytes(f"DOCUMENT {name} " + "content words for the text layer " * 3))
        return config, process_inbox(config)


@requires("samples/dev/D001.pdf", "labels/dev/D001.json")
class CalibrationCacheTests(unittest.TestCase):
    """校准必须按源文件 sha 取解析结果，不能凭文档名信任旧缓存。"""

    def test_stale_parse_cache_is_rebuilt(self):
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            samples = root / "samples" / "dev"
            labels = root / "labels" / "dev"
            parsed = root / "work" / "parsed"
            for directory in (samples, labels, parsed):
                directory.mkdir(parents=True)
            shutil.copy(ROOT / "samples" / "dev" / "D001.pdf", samples / "D001.pdf")
            shutil.copy(ROOT / "labels" / "dev" / "D001.json", labels / "D001.json")
            (parsed / "D001.json").write_text(json.dumps({
                "document_id": "D001", "doc_type": "UNKNOWN", "sha256": "stale",
                "parser_version": "old", "pages": [],
            }), encoding="utf-8")
            config["paths"] = dict(config["paths"], samples=str(root / "samples"))

            with mock.patch.object(calibrate, "ROOT", root):
                docs, parsed_by_id = calibrate.load_split("dev", config)

        self.assertEqual([doc["document_id"] for doc in docs], ["D001"])
        self.assertGreater(len(docs[0]["fields"]), 0)
        self.assertNotEqual(parsed_by_id["D001"]["sha256"], "stale")


class CalibrationAugmentationTests(unittest.TestCase):
    """P0-2：合成负例只补校准样本，不改变业务数据。"""

    SPECS = {
        "bl.registered_capital": {"type": "decimal", "risk": "monetary"},
        "bl.establishment_date": {"type": "date", "risk": "date"},
        "bl.legal_representative": {"type": "string", "risk": "party_name"},
        "bl.unified_social_credit_code": {"type": "string", "risk": "identity"},
        "bl.entity_type": {"type": "enum", "enum": "flag", "risk": "enum_flag"},
        "bl.registration_authority": {"type": "string", "risk": "optional_low_risk"},
    }
    VALUES = {
        "bl.registered_capital": "100.00",
        "bl.establishment_date": "2019-05-20",
        "bl.legal_representative": "张三",
        "bl.unified_social_credit_code": "91110000MA01ABCDEF",
        "bl.entity_type": "yes",
        "bl.registration_authority": "示例市市场监督管理局",
    }

    def _entry(self, path: str) -> Dict:
        spec = dict(self.SPECS[path])
        value = self.VALUES[path]
        return {
            "doc": {"document_id": "D900", "doc_type": "BUSINESS_LICENSE"},
            "path": path,
            "gt": {"value": value, "page": 1, "bbox": [40.0, 100.0, 140.0, 112.0]},
            "pred": {
                "path": path, "value": value, "risk": spec.get("risk", ""),
                "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                            "table_score": 1.0, "cross_field_score": 1.0,
                            "grounding_present": 1},
                "grounding": {"page": 1, "bbox": [40.0, 100.0, 140.0, 112.0]},
            },
            "spec": spec,
        }

    def _config(self, **overrides) -> Dict:
        config = copy.deepcopy(load_config())
        block = {"enabled": True, "seed": 11, "target": 10,
                 "near_miss_share": 0.45, "misgrounded_share": 0.35,
                 "wrong_value_share": 0.20, "max_per_field": 2}
        block.update(overrides)
        config["calibration"]["synthetic_negatives"] = block
        return config

    def test_synthetic_negatives_target_cap_and_determinism(self):
        entries = [self._entry(path) for path in self.SPECS]
        config = self._config()
        first = calibrate._synthetic_negatives(entries, config)
        again = calibrate._synthetic_negatives(entries, config)
        self.assertEqual(len(first), 10)
        self.assertEqual([item["synthetic"] for item in first],
                         [item["synthetic"] for item in again])
        self.assertEqual([item["features"] for item in first],
                         [item["features"] for item in again])
        self.assertTrue(all(item["y"] == 0 for item in first))
        self.assertTrue(all(item["document_id"] == "D900" for item in first))
        per_field: Dict[str, int] = {}
        for item in first:
            per_field[item["path"]] = per_field.get(item["path"], 0) + 1
        self.assertLessEqual(max(per_field.values()), 2)
        self.assertGreaterEqual(len({item["synthetic"] for item in first}), 2)
        disabled = calibrate._synthetic_negatives(entries, self._config(enabled=False))
        self.assertEqual(disabled, [])

    def test_near_miss_and_wrong_value_stay_format_shaped(self):
        spec_decimal = {"type": "decimal", "risk": "monetary"}
        near = calibrate._near_miss("100.00", spec_decimal, load_config(), random.Random(1))
        self.assertNotEqual(near, "100.00")
        self.assertRegex(near, r"^-?\d+\.\d{2}$")
        wrong = calibrate._wrong_value("100.00", spec_decimal, load_config(), random.Random(1))
        self.assertNotEqual(wrong, "100.00")
        self.assertRegex(wrong, r"^-?\d+\.\d{2}$")
        spec_date = {"type": "date", "risk": "date"}
        shifted = calibrate._near_miss("2019-05-20", spec_date, load_config(), random.Random(1))
        self.assertNotEqual(shifted, "2019-05-20")
        spec_identity = {"type": "string", "risk": "identity"}
        flipped = calibrate._near_miss("91110000MA01ABCDEF", spec_identity,
                                       load_config(), random.Random(1))
        self.assertEqual(len(flipped), len("91110000MA01ABCDEF"))
        self.assertNotEqual(flipped, "91110000MA01ABCDEF")
        spec_enum = {"type": "enum", "enum": "flag", "risk": "enum_flag"}
        other = calibrate._near_miss("yes", spec_enum, load_config(), random.Random(1))
        self.assertNotEqual(other, "yes")

    def test_misgrounded_moves_bbox_but_keeps_value(self):
        entry = self._entry("bl.registered_capital")
        shifted = calibrate._misgrounded(entry["pred"], random.Random(3))
        self.assertEqual(shifted["value"], entry["pred"]["value"])
        self.assertEqual(shifted["signals"], entry["pred"]["signals"])
        self.assertTrue(values_match(entry["spec"], entry["gt"]["value"], shifted["value"]))
        self.assertFalse(grounding_ok(entry["gt"], shifted["grounding"]))

    def test_reliability_block_math(self):
        probabilities = np.array([0.95, 0.96, 0.55, 0.99])
        labels = np.array([1, 0, 0, 1])
        config = copy.deepcopy(load_config())
        config["calibration"]["reliability_bins"] = [0.0, 0.6, 0.9, 0.95, 0.98, 1.01]
        block = calibrate._reliability_block(probabilities, labels, config)
        self.assertEqual([point["count"] for point in block["points"]], [1, 2, 1])
        self.assertAlmostEqual(block["ece"], 0.3675, places=4)
        self.assertAlmostEqual(block["brier"], 0.3067, places=4)


class TableRobustnessTests(unittest.TestCase):
    """P0 骨架：版面块辅助表格定位 + 合并单元格/列错位检测。"""

    HEADERS = [("交易日期", 35, 101, 100), ("摘要", 101, 185, 100),
               ("对方户名", 185, 275, 100), ("收支", 275, 309, 100),
               ("交易金额", 309, 375, 100), ("余额", 375, 439, 100)]

    @staticmethod
    def _seg(x0: float, y0: float, x1: float, y1: float, text: str) -> Dict:
        return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": 0.99,
                "chars": [], "engine": "pymupdf_text"}

    def _two_table_segments(self) -> List[Dict]:
        segments = [self._seg(x0, y0, x1, y0 + 12, text)
                    for text, x0, x1, y0 in self.HEADERS]
        # 大表三行：交易日期 / 摘要 / 对方户名 / 收支 / 交易金额 / 余额
        for row, y in enumerate((120, 140, 160)):
            segments += [
                self._seg(35, y, 101, y + 12, f"2025-07-0{row + 1}"),
                self._seg(101, y, 185, y + 12, "货款"),
                self._seg(185, y, 275, y + 12, "甲公司"),
                self._seg(275, y, 309, y + 12, "收入"),
                self._seg(309, y, 375, y + 12, "1000.00"),
                self._seg(375, y, 439, y + 12, "101000.00"),
            ]
        # 同页第二张小表：交易日期 / 摘要 / 利息金额，x 起点与大表一致
        for row, y in enumerate((220, 240)):
            segments += [
                self._seg(35, y, 101, y + 12, f"2025-08-1{row + 5}"),
                self._seg(101, y, 185, y + 12, "利息"),
                self._seg(185, y, 275, y + 12, "123.45"),
            ]
        return segments

    def test_layout_regions_split_two_tables(self):
        segments = self._two_table_segments()
        config = copy.deepcopy(load_config())
        config["parsing"]["layout"]["enabled"] = True
        blocks = [
            {"label": "table", "bbox": [30, 90, 450, 175]},
            {"label": "table", "bbox": [30, 210, 300, 260]},
        ]
        tables = _segment_tables(segments, "BANK_STATEMENT", 1, config, blocks)
        self.assertEqual(sum(len(t["rows"]) for t in tables), 3,
                         f"layout regions must isolate the big table, got {tables}")
        self.assertEqual(tables[0]["region"], "layout-0")
        disabled = copy.deepcopy(config)
        disabled["parsing"]["layout"]["enabled"] = False
        tables_off = _segment_tables(segments, "BANK_STATEMENT", 1, disabled, blocks)
        self.assertGreater(sum(len(t["rows"]) for t in tables_off), 3,
                           "without layout assist the small table rows pollute")

    def test_layout_miss_falls_back_to_page_behavior(self):
        segments = self._two_table_segments()
        config = copy.deepcopy(load_config())
        config["parsing"]["layout"]["enabled"] = True
        with_blocks = _segment_tables(segments, "BANK_STATEMENT", 1, config, [])
        without = _segment_tables(segments, "BANK_STATEMENT", 1, config, None)
        self.assertEqual(len(with_blocks), len(without))
        self.assertEqual(sum(len(t["rows"]) for t in with_blocks),
                         sum(len(t["rows"]) for t in without))

    def test_row_span_segment_marks_affected_rows(self):
        segments = [self._seg(x0, y0, x1, y0 + 12, text)
                    for text, x0, x1, y0 in self.HEADERS]
        for row, y in enumerate((120, 140, 160, 180, 200)):
            segments += [
                self._seg(35, y, 101, y + 12, f"2025-07-0{row + 1}"),
                self._seg(185, y, 275, y + 12, "甲公司"),
                self._seg(275, y, 309, y + 12, "收入"),
                self._seg(309, y, 375, y + 12, "1000.00"),
                self._seg(375, y, 439, y + 12, "101000.00"),
            ]
        segments += [
            # 跨行合并的摘要单元格：一条片段横跨第 2、3 行。
            self._seg(101, 140, 185, 180, "批量代发工资"),
        ]
        config = copy.deepcopy(load_config())
        tables = _segment_tables(segments, "BANK_STATEMENT", 1, config, None)
        self.assertEqual(len(tables), 1)
        anomalies = tables[0].get("anomalies", [])
        self.assertEqual([a["reason"] for a in anomalies], ["row_span"])
        self.assertEqual(anomalies[0]["rows"], [1, 2])
        # 关闭检测开关后不再记录。
        off = copy.deepcopy(config)
        off["parsing"]["table_anomaly"]["enabled"] = False
        tables_off = _segment_tables(segments, "BANK_STATEMENT", 1, off, None)
        self.assertNotIn("anomalies", tables_off[0])

    def test_row_span_forces_review_through_extract_and_decide(self):
        segments = [self._seg(x0, y0, x1, y0 + 12, text)
                    for text, x0, x1, y0 in self.HEADERS]
        for row, y in enumerate((120, 140, 160, 180, 200)):
            segments += [
                self._seg(35, y, 101, y + 12, f"2025-07-0{row + 1}"),
                self._seg(185, y, 275, y + 12, "甲公司"),
                self._seg(275, y, 309, y + 12, "收入"),
                self._seg(309, y, 375, y + 12, "1000.00"),
                self._seg(375, y, 439, y + 12, "101000.00"),
            ]
        segments += [self._seg(101, 140, 185, 180, "批量代发工资")]
        config = copy.deepcopy(load_config())
        tables = _segment_tables(segments, "BANK_STATEMENT", 1, config, None)
        parsed = {
            "document_id": "SEG", "doc_type": "BANK_STATEMENT",
            "page_count": 1, "engine": "pymupdf_text",
            "pages": [{"page": 1, "width": 595, "height": 842, "engine": "pymupdf_text",
                       "preprocess": {}, "quality_flags": [], "lines": [],
                       "tables": tables}],
        }
        extracted = extract_document(parsed, config)
        flag_row = extracted["fields"]["bs.transactions[1].summary"]
        clean_row = extracted["fields"]["bs.transactions[0].summary"]
        self.assertEqual(flag_row["meta"]["table_anomaly"], ["row_span"])
        self.assertNotIn("table_anomaly_columns", flag_row["meta"])
        self.assertNotIn("table_anomaly", clean_row["meta"])
        decided = decide_field(copy.deepcopy(flag_row), extracted, parsed, config,
                               [], Calibrator(config))
        self.assertIn("TABLE_ANOMALY", decided["review_reasons"])
        self.assertEqual(decided["decision"], "review")

    def _merge_segments(self, summary_right: float) -> List[Dict]:
        """五行流水，第 2 行的「摘要」框越到给定右边界。

        对方户名列在该行留空——真实数电发票上「项目名称/规格型号」被并成一框就是这个形状：
        框越进了右边那一列的中心，而那一列本行没有自己的框。
        """
        segments = [self._seg(x0, y0, x1, y0 + 12, text)
                    for text, x0, x1, y0 in self.HEADERS]
        for row, y in enumerate((120, 140, 160, 180, 200)):
            segments += [
                self._seg(35, y, 101, y + 12, f"2025-07-0{row + 1}"),
                self._seg(275, y, 309, y + 12, "支出"),
                self._seg(309, y, 375, y + 12, "500.00"),
                self._seg(375, y, 439, y + 12, "100500.00"),
            ]
            if row == 1:
                segments.append(self._seg(101, y, summary_right, y + 12, "代付甲公司货款"))
            else:
                segments += [
                    self._seg(101, y, 185, y + 12, "货款"),
                    self._seg(185, y, 275, y + 12, "甲公司"),
                ]
        return segments

    def test_wide_cell_covering_a_blank_column_is_flagged_as_merge(self):
        config = copy.deepcopy(load_config())
        tables = _segment_tables(self._merge_segments(240.0), "BANK_STATEMENT", 1, config, None)
        merges = [a for a in tables[0]["anomalies"] if a["reason"] == "column_merge"]
        self.assertEqual(len(merges), 1, f"expected one merge, got {tables[0]['anomalies']}")
        self.assertEqual(merges[0]["rows"], [1])
        self.assertEqual(merges[0]["columns"], ["摘要", "对方户名"])
        off = copy.deepcopy(config)
        off["parsing"]["table_anomaly"]["detect_column_merge"] = False
        tables_off = _segment_tables(self._merge_segments(240.0), "BANK_STATEMENT", 1, off, None)
        self.assertNotIn("column_merge", [a["reason"] for a in tables_off[0].get("anomalies", [])])

    def test_blank_column_next_to_a_narrow_cell_is_not_a_merge(self):
        """对方户名真的空着、摘要框没越线：不是并框，不该增加复核。"""
        config = copy.deepcopy(load_config())
        tables = _segment_tables(self._merge_segments(185.0), "BANK_STATEMENT", 1, config, None)
        self.assertNotIn("column_merge",
                         [a["reason"] for a in tables[0].get("anomalies", [])])

    def test_column_merge_forces_review_without_splitting_the_value(self):
        config = copy.deepcopy(load_config())
        tables = _segment_tables(self._merge_segments(240.0), "BANK_STATEMENT", 1, config, None)
        parsed = {
            "document_id": "MERGE", "doc_type": "BANK_STATEMENT",
            "page_count": 1, "engine": "pymupdf_text",
            "pages": [{"page": 1, "width": 595, "height": 842, "engine": "pymupdf_text",
                       "preprocess": {}, "quality_flags": [], "lines": [],
                       "tables": tables}],
        }
        extracted = extract_document(parsed, config)
        merged = extracted["fields"]["bs.transactions[1].summary"]
        neighbour = extracted["fields"]["bs.transactions[0].summary"]
        # 值原样保留（不按比例猜拆），只把这一行标成异常。
        self.assertEqual(merged["value"], "代付甲公司货款")
        self.assertEqual(merged["meta"]["table_anomaly"], ["column_merge"])
        # 复核界面要能直接说出被并的是哪两列，不用回解析 JSON 里查。
        self.assertEqual(merged["meta"]["table_anomaly_columns"], [["摘要", "对方户名"]])
        self.assertNotIn("table_anomaly", neighbour.get("meta") or {})
        decided = decide_field(copy.deepcopy(merged), extracted, parsed, config,
                               [], Calibrator(config))
        self.assertIn("TABLE_ANOMALY", decided["review_reasons"])
        self.assertEqual(decided["decision"], "review")
        switched = copy.deepcopy(config)
        switched["validation"]["review_on_table_anomaly"] = False
        relaxed = decide_field(copy.deepcopy(merged), extracted, parsed, switched,
                               [], Calibrator(switched))
        self.assertNotIn("TABLE_ANOMALY", relaxed["review_reasons"])

    @staticmethod
    def _fixture_data() -> Dict:
        rng = random.Random(7)
        txns = synth.make_bank_transactions(rng, date(2025, 7, 1), date(2025, 12, 31),
                                            Decimal("100000.00"), 5)
        return {
            "bank_name": "示例银行",
            "account_number": "6222020200112233445",
            "period_start": date(2025, 7, 1),
            "period_end": date(2025, 12, 31),
            "opening_balance": Decimal("100000.00"),
            "transactions": txns,
        }

    @staticmethod
    def _fixture_pdf(out_dir: Path, extra: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        company = synth.make_company(random.Random(7), 0)
        data = TableRobustnessTests._fixture_data()
        doc = fitz.open()
        synth.build_bank_statement_fixture(doc, company, data, extra=extra)
        path = out_dir / f"fixture-{extra}.pdf"
        doc.save(path, garbage=4, deflate=True)
        doc.close()
        return path

    @staticmethod
    def _txn_count(extracted: Dict) -> int:
        return sum(1 for key in extracted["fields"]
                   if key.startswith("bs.transactions[") and key.endswith(".txn_date"))

    def test_pdf_two_tables_layout_assist_isolates_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = self._fixture_pdf(root / "src", "two_tables")
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed_off = parse_document(pdf, "OFF", force=True, config=config,
                                        out_dir=root / "parsed-off")
            extracted_off = extract_document(parsed_off, config)
            self.assertGreater(self._txn_count(extracted_off), 5,
                               "the small table must pollute without layout assist")
            config_on = copy.deepcopy(config)
            config_on["parsing"]["layout"]["enabled"] = True
            parsed_on = parse_document(pdf, "ON", force=True, config=config_on,
                                       out_dir=root / "parsed-on")
            extracted_on = extract_document(parsed_on, config_on)
            self.assertEqual(self._txn_count(extracted_on), 5)

    def test_pdf_stray_note_flags_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = self._fixture_pdf(root / "src", "stray_note")
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(pdf, "NOTE", force=True, config=config,
                                    out_dir=root / "parsed")
            anomalies = [a for page in parsed["pages"] for table in page["tables"]
                         for a in table.get("anomalies", [])]
            mismatches = [a for a in anomalies if a["reason"] == "column_mismatch"]
            self.assertTrue(mismatches, f"expected column_mismatch, got {anomalies}")
            self.assertEqual(mismatches[0]["rows"], [3])
            # 这份夹具没有竖线，OCR 之前文本层就把「对方户名」和「收支方向」并进了同一个框
            # （row0 的 counterparty 抽不到、direction 为空就是这个原因）。
            # 现在这类横向并框也会被记下来，不再只盯越界片段。
            merges = [a for a in anomalies if a["reason"] == "column_merge"]
            self.assertTrue(merges, f"expected column_merge too, got {anomalies}")
            self.assertEqual(merges[0]["columns"], ["对方户名", "收支方向"])
            extracted = extract_document(parsed, config)
            docs = assess([extracted], {extracted["document_id"]: parsed}, config,
                          cross_document=False)
            row3 = docs[0]["fields"]["bs.transactions[3].amount"]
            row0 = docs[0]["fields"]["bs.transactions[0].amount"]
            self.assertIn("TABLE_ANOMALY", row3.get("review_reasons", []))
            self.assertIn("TABLE_ANOMALY", row0.get("review_reasons", []))
            off = copy.deepcopy(config)
            off["validation"]["review_on_table_anomaly"] = False
            docs_off = assess([extracted], {extracted["document_id"]: parsed}, off,
                              cross_document=False)
            self.assertNotIn("TABLE_ANOMALY",
                             docs_off[0]["fields"]["bs.transactions[3].amount"]
                             .get("review_reasons", []))
            self.assertNotIn("TABLE_ANOMALY",
                             docs_off[0]["fields"]["bs.transactions[0].amount"]
                             .get("review_reasons", []))


class LabelRegionTests(unittest.TestCase):
    """label_region: 一行多值、分框带空格、标签下方堆叠取值。"""

    @staticmethod
    def _line(text: str, x0: float, y0: float, x1: float, y1: float,
              confidence: float = 0.98) -> Dict:
        return {"text": text, "confidence": confidence,
                "parts": [{"text": text, "bbox": [x0, y0, x1, y1],
                           "confidence": confidence}]}

    def test_value_index_reads_the_configured_column(self):
        lines = [
            self._line("合计", 100, 100, 140, 112),
            self._line("72566.37", 380, 100, 450, 112),
            self._line("9433.63", 520, 100, 600, 112),
        ]
        first = locate_label_region(lines, "合计", value_index=0)
        second = locate_label_region(lines, "合计", value_index=1)
        self.assertEqual(first["value"], "72566.37")
        self.assertEqual(second["value"], "9433.63")
        # 越界的列号不能悄悄退回第 0 列。
        self.assertIsNone(locate_label_region(lines, "合计", value_index=2))

    def test_preserve_spaces_keeps_separate_boxes_apart(self):
        lines = [
            self._line("运输工具", 100, 100, 180, 112),
            self._line("YM", 200, 100, 225, 112),
            self._line("TUTORIAL", 235, 100, 300, 112),
        ]
        compact = locate_label_region(lines, "运输工具", max_value_parts=2)
        spaced = locate_label_region(lines, "运输工具", max_value_parts=2,
                                     preserve_spaces=True)
        self.assertEqual(compact["value"], "YMTUTORIAL")
        self.assertEqual(spaced["value"], "YM TUTORIAL")

    def test_value_below_label_needs_the_switch_and_a_short_gap(self):
        lines = [
            self._line("填制单位", 100, 100, 180, 112),
            self._line("洋山海关", 100, 120, 180, 132),
        ]
        self.assertIsNone(locate_label_region(lines, "填制单位"))
        stacked = locate_label_region(lines, "填制单位", allow_below=True)
        self.assertEqual(stacked["value"], "洋山海关")
        self.assertEqual(stacked["bbox"], [100.0, 120.0, 180.0, 132.0])
        far = list(lines)
        far[1] = self._line("洋山海关", 100, 160, 180, 172)
        self.assertIsNone(locate_label_region(far, "填制单位", allow_below=True))


class RegexAnchorTests(unittest.TestCase):
    """配了 label 的 regex 只在标签行上取数，缺标签就留空。"""

    @staticmethod
    def _parsed(doc_type: str, texts) -> Dict:
        lines = []
        for index, text in enumerate(texts):
            box = [50.0, 100.0 + index * 20, 400.0, 112.0 + index * 20]
            lines.append({"text": text, "confidence": 0.99, "line_index": index,
                          "bbox": box,
                          "parts": [{"text": text, "confidence": 0.99, "bbox": box}]})
        return {"doc_type": doc_type,
                "pages": [{"page": 1, "width": 595, "height": 842,
                           "lines": lines, "tables": []}]}

    def test_labelled_regex_does_not_take_another_number(self):
        """官方征信报告只有「中征码」，没有「报告编号」：不能拿中征码顶替。"""
        parsed = self._parsed("CREDIT_REPORT_ENTERPRISE", [
            "企业名称： 虚构演示企业",
            "中征码： 0000000012345678",
            "统一社会信用代码： 91000000DEMA000013",
            "报告时间： 2024-02-03T10:05:15",
        ])
        fields = extract_document(parsed, load_config())["fields"]
        self.assertIsNone(fields["cr.report_number"]["value"])
        self.assertEqual(fields["cr.report_number"]["status"], "missing")
        self.assertEqual(fields["cr.unified_social_credit_code"]["value"],
                         "91000000DEMA000013")
        # 报告时间就是这份报告的查询时点，别名解析到日期，不丢时分秒以外的信息。
        self.assertEqual(fields["cr.query_date"]["value"], "2024-02-03")

    def test_self_anchored_regex_may_skip_the_label(self):
        """「（小写）」本身是锚点，开 allow_unlabeled 后金额单独成行也能取。"""
        parsed = self._parsed("VAT_INVOICE", [
            "价税合计（大写） 捌仟贰佰圆整",
            "（小写）￥8200.00",
        ])
        fields = extract_document(parsed, load_config())["fields"]
        self.assertEqual(fields["vi.total_with_tax"]["value"], "8200.00")


class HeaderMatrixTests(unittest.TestCase):
    """征信报告的重表头矩阵：列名重复，只能按"表头序列 + 数值格数量"对齐。"""

    SECTION = "未结清信贷及授信信息概要"
    HEADERS = ["正常类", "关注类", "不良类", "合计"]
    COLUMNS = ["账户数", "余额"]

    @staticmethod
    def _line(parts, y: float) -> Dict:
        payload = [{"text": text, "confidence": 0.99, "bbox": [x0, y, x1, y + 10]}
                   for text, x0, x1 in parts]
        return {"text": " ".join(text for text, _, _ in parts), "confidence": 0.99,
                "bbox": [payload[0]["bbox"][0], y, payload[-1]["bbox"][2], y + 10],
                "parts": payload}

    def _lines(self):
        return [
            self._line([(self.SECTION, 74.7, 219.2)], 558.3),
            self._line([("正常类", 171.0, 198.1), ("关注类", 272.2, 299.3),
                        ("不良类", 373.4, 400.5), ("合计", 479.0, 497.1)], 652.6),
            self._line([("账户数", 145.7, 172.8), ("余额", 200.9, 219.1),
                        ("账户数", 246.9, 274.0), ("余额", 302.0, 320.1),
                        ("账户数", 348.1, 375.2), ("余额", 403.1, 421.3),
                        ("账户数", 449.1, 476.2), ("余额", 504.3, 522.5)], 668.7),
            self._line([("中长期借款", 72.7, 117.8), ("6", 157.0, 161.5),
                        ("401.20", 197.6, 222.3), ("1", 258.2, 262.7),
                        ("9.35", 300.9, 321.2), ("1", 359.4, 363.9),
                        ("14.45", 402.1, 422.4), ("8", 458.3, 467.3),
                        ("425.00", 501.1, 525.9)], 684.8),
            self._line([("合计", 86.3, 104.4), ("12", 154.8, 163.9),
                        ("920.15", 195.3, 224.6), ("3", 258.2, 262.7),
                        ("25.10", 300.9, 321.2), ("2", 359.4, 363.9),
                        ("30.25", 399.8, 424.5), ("17", 458.3, 467.3),
                        ("975.50", 498.8, 528.1)], 748.9),
        ]

    def _extract(self, lines, index):
        return extract_header_matrix(
            lines, section=self.SECTION, header_sequence=self.HEADERS,
            column_labels=self.COLUMNS, row_label="合计", value_index=index)

    def test_repeated_columns_map_by_position(self):
        lines = self._lines()
        count = self._extract(lines, 6)
        balance = self._extract(lines, 7)
        self.assertEqual(count["value"], "17")
        self.assertEqual(balance["value"], "975.50")
        self.assertEqual(count["bbox"], [458.3, 748.9, 467.3, 758.9])
        # 不良类账户数（index 4）与关注类余额（index 3）也要对。
        self.assertEqual(self._extract(lines, 4)["value"], "2")
        self.assertEqual(self._extract(lines, 3)["value"], "25.10")

    def test_merged_cell_breaks_the_alignment(self):
        """子表头少一格就不再是完整矩阵，宁可返回空也不按顺序硬对。"""
        lines = self._lines()
        lines[2] = self._line([("账户数", 145.7, 172.8), ("余额", 200.9, 219.1),
                               ("账户数", 246.9, 274.0), ("余额", 302.0, 320.1),
                               ("账户数", 348.1, 375.2), ("余额", 403.1, 421.3),
                               ("账户数", 449.1, 476.2)], 668.7)
        self.assertIsNone(self._extract(lines, 6))

    def test_missing_number_rejects_the_whole_row(self):
        lines = self._lines()
        lines[4] = self._line([("合计", 86.3, 104.4), ("12", 154.8, 163.9),
                               ("920.15", 195.3, 224.6), ("3", 258.2, 262.7),
                               ("25.10", 300.9, 321.2), ("2", 359.4, 363.9),
                               ("30.25", 399.8, 424.5), ("17", 458.3, 467.3)], 748.9)
        self.assertIsNone(self._extract(lines, 6))

    def test_section_and_row_label_are_required(self):
        lines = self._lines()
        self.assertIsNone(extract_header_matrix(
            lines, section="未结清贷款明细", header_sequence=self.HEADERS,
            column_labels=self.COLUMNS, row_label="合计", value_index=6))
        self.assertIsNone(extract_header_matrix(
            lines, section=self.SECTION, header_sequence=self.HEADERS,
            column_labels=self.COLUMNS, row_label="小计", value_index=6))

    def test_report_note_states_the_unit(self):
        line = self._line([("12. 如无特别说明，本报告中的金额类数据项单位均为万元。",
                            50.0, 500.0)], 300.0)
        self.assertEqual(extract_unit([line])["value"], "万元")

    def _parsed(self):
        note = self._line([("12. 如无特别说明，本报告中的金额类数据项单位均为万元。",
                            50.0, 520.0)], 200.0)
        cover = [
            self._line([("企业名称： 虚构演示企业", 50.0, 400.0)], 100.0),
            self._line([("报告时间： 2024-02-03T10:05:15", 50.0, 400.0)], 130.0),
        ]
        pages = [
            {"page": 1, "width": 595, "height": 842, "lines": cover, "tables": []},
            {"page": 2, "width": 595, "height": 842, "lines": [note], "tables": []},
            {"page": 3, "width": 595, "height": 842, "lines": self._lines(), "tables": []},
        ]
        return {"doc_type": "CREDIT_REPORT_ENTERPRISE", "pages": pages}

    def test_fallback_reads_amount_and_converts_wan_yuan(self):
        parsed = self._parsed()
        fields = extract_document(parsed, load_config())["fields"]
        count = fields["cr.loan_account_count"]
        balance = fields["cr.outstanding_loan_balance"]
        self.assertEqual(count["value"], 17)
        self.assertEqual(balance["value"], "9755000.00")
        self.assertEqual(balance["meta"]["unit_factor"], 10000)
        self.assertTrue(balance["meta"]["matrix_fallback"])
        self.assertEqual(balance["grounding"]["page"], 3)
        self.assertEqual(balance["extractor_id"], "header_matrix:合计:7")

        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        result = decide_field(balance, {"document_id": "T1",
                                        "doc_type": "CREDIT_REPORT_ENTERPRISE"},
                              parsed, config, [], Calibrator(config))
        self.assertEqual(result["decision"], "review")
        self.assertIn("MATRIX_FALLBACK", result["review_reasons"])

    def test_labelled_layout_is_untouched_by_the_fallback(self):
        """合成版的「未结清贷款余额：6,513,016.00」仍走 label_value，不碰单位折算。"""
        parsed = self._parsed()
        parsed["pages"][2]["lines"] = [
            self._line([("未结清贷款余额： 6,513,016.00", 50.0, 400.0)], 100.0)]
        fields = extract_document(parsed, load_config())["fields"]
        balance = fields["cr.outstanding_loan_balance"]
        self.assertEqual(balance["value"], "6513016.00")
        self.assertEqual(balance["extractor_id"], "label_value:未结清贷款余额")
        self.assertNotIn("unit_factor", balance.get("meta") or {})


class OcrRecropTests(unittest.TestCase):
    """OCR drops isolated glyphs on a full page; the label gap may be re-read, never trusted."""

    OPTIONS = {"enabled": True, "dpi": 400, "min_confidence": 0.55, "max_chars": 24,
               "max_parts": 1, "max_gap": 160, "gap_pad": 2, "vertical_pad": 0.25,
               "max_per_page": 8, "separator_chars": ["：", ":"]}

    @staticmethod
    def _line(text: str = "未结清贷款笔数：", x: float = 63.43, y: float = 307.23,
              width: float = 70.63, height: float = 12.31) -> Dict:
        bbox = [x, y, x + width, y + height]
        return {"text": text, "bbox": bbox, "confidence": 0.999,
                "parts": [{"text": text, "bbox": bbox, "confidence": 0.999,
                           "engine": "rapidocr"}]}

    @staticmethod
    def _page() -> mock.Mock:
        page = mock.Mock()
        page.rect.width = 595.28
        page.rect.height = 841.89
        return page

    @staticmethod
    def _config(enabled: bool = True) -> Dict:
        config = copy.deepcopy(load_config())
        config["parsing"]["ocr_recrop"]["enabled"] = enabled
        return config

    def test_gap_rect_covers_only_the_blank_right_of_a_label(self):
        rect = gap_recrop_rect(self._line(), 595.28, 841.89, self.OPTIONS)
        self.assertAlmostEqual(rect[0], 136.06, places=2)
        self.assertAlmostEqual(rect[2], 296.06, places=2)
        self.assertLess(rect[1], 307.23)
        self.assertGreater(rect[3], 319.54)
        # A line that already carries a value is never re-read.
        self.assertIsNone(gap_recrop_rect(
            self._line("未结清贷款余额： 8,608,479.00"), 595.28, 841.89, self.OPTIONS))
        # No room before the page edge.
        self.assertIsNone(gap_recrop_rect(self._line(), 138.0, 841.89, self.OPTIONS))

    def test_accepts_one_short_value_only(self):
        self.assertTrue(accepts_recrop_text("4", 0.9, self.OPTIONS))
        self.assertTrue(accepts_recrop_text(" 正常 ", 0.8, self.OPTIONS))
        self.assertFalse(accepts_recrop_text("4", 0.2, self.OPTIONS))
        self.assertFalse(accepts_recrop_text("4:5", 0.9, self.OPTIONS))
        self.assertFalse(accepts_recrop_text("1" * 25, 0.9, self.OPTIONS))
        self.assertFalse(accepts_recrop_text("……", 0.9, self.OPTIONS))
        self.assertFalse(accepts_recrop_text("", 0.9, self.OPTIONS))

    def test_recovered_part_maps_back_to_page_coordinates(self):
        page = self._page()
        rect = gap_recrop_rect(self._line(), page.rect.width, page.rect.height,
                               self.OPTIONS)
        crop = mock.Mock(return_value=[([100.0, 20.0, 140.0, 70.0], "4", 0.93)])
        parts = recover_gap_parts(page, [self._line()], self._config(), crop_ocr=crop)
        self.assertEqual(len(parts), 1)
        part = parts[0]
        self.assertEqual(part["text"], "4")
        self.assertTrue(part["recovered"])
        self.assertEqual(part["engine"], "rapidocr_recrop")
        scale = 72.0 / 400
        self.assertAlmostEqual(part["bbox"][0], rect[0] + 100.0 * scale, places=2)
        self.assertAlmostEqual(part["bbox"][1], rect[1] + 20.0 * scale, places=2)
        self.assertAlmostEqual(part["bbox"][2], rect[0] + 140.0 * scale, places=2)
        self.assertAlmostEqual(part["bbox"][3], rect[1] + 70.0 * scale, places=2)

        line = group_parts([self._line()["parts"][0], part])[0]
        self.assertTrue(line["text"].endswith("4"))
        self.assertTrue(line["recovered"])

    def test_switch_and_guards_drop_the_whole_recovery(self):
        page = self._page()
        crop = mock.Mock(return_value=[([100.0, 20.0, 140.0, 70.0], "4", 0.93)])
        self.assertEqual(recover_gap_parts(page, [self._line()], self._config(False),
                                           crop_ocr=crop), [])
        crop.assert_not_called()

        two = mock.Mock(return_value=[([0.0, 0.0, 10.0, 10.0], "4", 0.93),
                                      ([20.0, 0.0, 30.0, 10.0], "5", 0.93)])
        self.assertEqual(recover_gap_parts(page, [self._line()], self._config(),
                                           crop_ocr=two), [])

        low = mock.Mock(return_value=[([0.0, 0.0, 10.0, 10.0], "4", 0.2)])
        self.assertEqual(recover_gap_parts(page, [self._line()], self._config(),
                                           crop_ocr=low), [])

        fragment = mock.Mock(return_value=[([0.0, 0.0, 10.0, 10.0], "4：5", 0.93)])
        self.assertEqual(recover_gap_parts(page, [self._line()], self._config(),
                                           crop_ocr=fragment), [])

    def test_recovered_value_is_review_only(self):
        label = self._line()["parts"][0]
        recovered = {"text": "4", "bbox": [214.0, 308.47, 219.4, 316.93],
                     "confidence": 0.999, "chars": [],
                     "engine": "rapidocr_recrop", "recovered": True}
        line = group_parts([label, recovered])[0]
        parsed = {"doc_type": "CREDIT_REPORT_ENTERPRISE",
                  "pages": [{"page": 1, "width": 595.28, "height": 841.89,
                             "lines": [line], "tables": []}]}
        config = self._config()
        config["calibration"]["artifact"] = None
        field = extract_document(parsed, config)["fields"]["cr.loan_account_count"]
        self.assertEqual(field["value"], 4)
        self.assertTrue((field.get("meta") or {}).get("local_recrop"))
        result = decide_field(copy.deepcopy(field),
                              {"document_id": "R1",
                               "doc_type": "CREDIT_REPORT_ENTERPRISE"},
                              parsed, config, [], Calibrator(config))
        self.assertEqual(result["decision"], "review")
        self.assertIn("LOCAL_RECROP", result["review_reasons"])

        off = copy.deepcopy(config)
        off["validation"]["review_on_local_recrop"] = False
        result_off = decide_field(copy.deepcopy(field),
                                  {"document_id": "R1",
                                   "doc_type": "CREDIT_REPORT_ENTERPRISE"},
                                  parsed, off, [], Calibrator(off))
        self.assertNotIn("LOCAL_RECROP", result_off["review_reasons"])


class SegmentTableConfigTests(unittest.TestCase):
    """表头行容差与无竖线的合并数字列拆分。"""

    HEADERS = [("名称", 30, 120), ("数量", 200, 250), ("完税价格(¥)", 300, 400),
               ("税率(%)", 400, 470), ("税款金额(¥)", 470, 560)]

    def _config(self) -> Dict:
        config = copy.deepcopy(load_config())
        config["doc_types"]["SYNTH_RECEIPT"] = {
            "tables": {
                "tax_items": {
                    "headers": [header for header, _, _ in self.HEADERS],
                    "min_columns": 4,
                    "header_row_slack": 6,
                    "row_required": ["名称", "税款金额(¥)"],
                    "column_patterns": {
                        "完税价格(¥)": r"^\d[\d,]*\.\d{2}$",
                        "税率(%)": r"^\d{1,2}(?:\.\d{1,4})?$",
                    },
                }
            }
        }
        return config

    @staticmethod
    def _seg(text: str, x0: float, y0: float, x1: float, y1: float) -> Dict:
        return {"text": text, "bbox": [x0, y0, x1, y1], "confidence": 0.99,
                "chars": [], "engine": "pymupdf_text"}

    def _segments(self) -> list:
        # 表头框较高，首行数据框顶边压进表头框 6pt，这是发票表头常见的印刷重叠。
        segments = [self._seg(header, x0, 100, x1, 130)
                    for header, x0, x1 in self.HEADERS]
        segments += [
            self._seg("注塑机模具", 30, 124, 120, 136),
            self._seg("1.00", 200, 124, 250, 136),
            self._seg("6,679,515.0813.0000", 300, 124, 470, 136),
            self._seg("868336.96", 470, 124, 560, 136),
        ]
        return segments

    def test_header_row_slack_admits_the_first_data_row(self):
        config = self._config()
        tables = _segment_tables(self._segments(), "SYNTH_RECEIPT", 1, config, None)
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["rows"]), 1)
        tighter = copy.deepcopy(config)
        tighter["doc_types"]["SYNTH_RECEIPT"]["tables"]["tax_items"]["header_row_slack"] = 2
        self.assertEqual(_segment_tables(self._segments(), "SYNTH_RECEIPT", 1,
                                         tighter, None), [])

    def test_merged_numeric_columns_are_split_only_on_full_match(self):
        config = self._config()
        tables = _segment_tables(self._segments(), "SYNTH_RECEIPT", 1, config, None)
        cells = tables[0]["rows"][0]["cells"]
        self.assertEqual(cells["完税价格(¥)"]["text"], "6,679,515.08")
        self.assertEqual(cells["税率(%)"]["text"], "13.0000")
        self.assertTrue(cells["完税价格(¥)"]["merged_split"])
        self.assertTrue(cells["税率(%)"]["merged_split"])

        spec = config["doc_types"]["SYNTH_RECEIPT"]["tables"]["tax_items"]
        bounds = {header: [x0, 100, x1, 130] for header, x0, x1 in self.HEADERS}
        headers = [header for header, _, _ in self.HEADERS]
        seg = self._seg("6,679,515.0813.0000%", 300, 124, 470, 136)
        self.assertIsNone(_split_merged_numeric(seg, headers, bounds, spec))


class OcrEngineTests(unittest.TestCase):
    """OCR 后端是配置项，不是 parse.py 里写死的一段 import；换后端必须能真的换掉。"""

    @staticmethod
    def _config(engine: str) -> Dict:
        config = copy.deepcopy(load_config())
        config["parsing"]["ocr_engine"] = engine
        return config

    @staticmethod
    def _scan_pdf(path: Path, pixels: tuple = (2480, 3508)) -> Path:
        buffer = io.BytesIO()
        Image.new("RGB", pixels, (250, 250, 250)).save(buffer, "JPEG", quality=85)
        document = fitz.open()
        page = document.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buffer.getvalue())
        document.save(path, garbage=4, deflate=True)
        document.close()
        return path

    def test_engine_id_is_read_from_config(self):
        null = ocr.get_engine(self._config("null"))
        self.assertEqual(null.name, "null")
        self.assertFalse(null.is_ocr)
        self.assertEqual(null.recognize(None), [])
        self.assertTrue(ocr.RapidOcrEngine.is_ocr)
        # 出厂跑的是配置里写的那个引擎，不是代码恰好 import 进来的那个。
        self.assertEqual(ocr.get_engine(load_config()).name,
                         load_config()["parsing"]["ocr_engine"])

    def test_unregistered_engine_is_refused_not_silently_replaced(self):
        for engine_id in ("ppocr", "rapidOCR", ""):
            with self.assertRaises(ValueError) as caught:
                ocr.get_engine(self._config(engine_id))
            message = str(caught.exception)
            self.assertIn(repr(engine_id), message)
            self.assertIn("rapidocr", message)
            self.assertIn("null", message)

    def test_engines_are_built_once_and_not_left_in_the_cache(self):
        built = []

        class Counting(ocr.OcrEngine):
            name = "counting"

            def __init__(self) -> None:
                built.append(1)

            def recognize(self, image):
                return []

        with mock.patch.dict(ocr.BUILDERS, {"counting": Counting}), \
                mock.patch.dict(ocr._INSTANCES, {}, clear=True):
            engine = ocr.get_engine(self._config("counting"))
            self.assertIs(ocr.get_engine(self._config("counting")), engine)
            self.assertEqual(len(built), 1)
        self.assertNotIn("counting", ocr._INSTANCES)

    def test_part_labels_follow_the_configured_engine(self):
        class Toy(ocr.OcrEngine):
            name = "toy"

            def recognize(self, image):
                return [([[0.0, 0.0], [80.0, 0.0], [80.0, 20.0], [0.0, 20.0]],
                         "1,234.56", 0.91)]

        page = mock.Mock()
        page.rect.width = 595.28
        page.rect.height = 841.89
        array = np.full((40, 60, 3), 250, dtype=np.uint8)
        page.get_pixmap.return_value = mock.Mock(samples=array.tobytes(),
                                                 height=40, width=60, n=3)
        config = self._config("toy")
        label = {"text": "未结清贷款笔数：", "bbox": [63.43, 307.23, 134.06, 319.54],
                 "confidence": 0.99, "parts": []}
        with mock.patch.dict(ocr.BUILDERS, {"toy": Toy}), \
                mock.patch.dict(ocr._INSTANCES, {}, clear=True), \
                mock.patch("docpipe.parse.preprocess.ocr_input",
                           side_effect=lambda image, _config: (image, {})):
            parts, mean_score, _ = ocr_parts(page, 300, config)
            recovered = recover_gap_parts(page, [label], config,
                                          crop_ocr=lambda *_a: [([100.0, 20.0, 140.0, 70.0],
                                                                 "4", 0.93)])
        self.assertEqual(parts[0]["engine"], "toy")
        self.assertEqual(parts[0]["bbox"], [0.0, 0.0, 19.2, 4.8])
        self.assertEqual(mean_score, 0.91)
        self.assertEqual(recovered[0]["engine"], "toy_recrop")

    def test_null_engine_switches_the_ocr_leg_off_for_real(self):
        """引擎退回默认就等于「配置是装饰」——真跑一份扫描页，看这页到底还有没有 OCR 证据。"""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._scan_pdf(root / "scan.pdf")
            record = parse_document(source, "NULL1", config=self._config("null"),
                                    out_dir=root / "parsed")
            self.assertEqual(record["pages"][0]["engine"], "null")
            self.assertFalse(record["quality"]["ocr_used"])
            self.assertEqual(record["quality"]["line_count"], 0)
            self.assertEqual(record["quality"]["recrop_parts"], 0)
            self.assertEqual(record["engine"], "null")


class PageQualityTests(unittest.TestCase):
    """页面实际带了多少像素，决定这页的字段能不能自动接受。

    低分辨率难例的失效模式不是 OCR 报错，而是把 120dpi 的糊字放大到 300dpi
    后「高置信度地」读错；只有衡量原始栅格密度才能拦住它。
    """

    @staticmethod
    def _scan_pdf(path: Path, pixels: tuple, page_size: Optional[tuple] = None) -> Path:
        buffer = io.BytesIO()
        Image.new("RGB", pixels, (250, 250, 250)).save(buffer, "JPEG", quality=70)
        width, height = page_size or pixels
        document = fitz.open()
        target = document.new_page(width=width, height=height)
        target.insert_image(target.rect, stream=buffer.getvalue())
        document.save(path, garbage=4, deflate=True)
        document.close()
        return path

    @staticmethod
    def _raster(path: Path, config: Dict) -> Optional[Dict]:
        document = fitz.open(path)
        try:
            return _page_raster(document[0], config)
        finally:
            document.close()

    def test_effective_dpi_comes_from_the_embedded_scan(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temp:
            low = self._scan_pdf(Path(temp) / "low.pdf", (992, 1403), (595, 842))
            high = self._scan_pdf(Path(temp) / "high.pdf", (2480, 3508), (595, 842))
            low_raster = self._raster(low, config)
            high_raster = self._raster(high, config)
        self.assertAlmostEqual(low_raster["effective_dpi"], 120.0, delta=1.0)
        self.assertTrue(low_raster["low_resolution"])
        self.assertAlmostEqual(high_raster["effective_dpi"], 300.0, delta=2.0)
        self.assertFalse(high_raster["low_resolution"])

    def test_image_import_pages_are_judged_against_a4_not_their_point_size(self):
        # 图片导入生成的 PDF 一像素一点：页面 2480×3508 点看起来很大，
        # 但物理上仍是 A4@300dpi；只有按 A4 折算才能得到真实密度。
        config = load_config()
        with tempfile.TemporaryDirectory() as temp:
            imported = self._scan_pdf(Path(temp) / "imported.pdf", (2480, 3508))
            small = self._scan_pdf(Path(temp) / "small.pdf", (992, 1403))
            imported_raster = self._raster(imported, config)
            small_raster = self._raster(small, config)
        self.assertAlmostEqual(imported_raster["effective_dpi"], 300.0, delta=2.0)
        self.assertFalse(imported_raster["low_resolution"])
        self.assertAlmostEqual(small_raster["effective_dpi"], 120.0, delta=1.0)
        self.assertTrue(small_raster["low_resolution"])

    def test_vector_pages_carry_no_raster_profile(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "text.pdf"
            document = fitz.open()
            page = document.new_page(width=595, height=842)
            page.insert_text((60, 80), "示例文本层页面", fontsize=11)
            document.save(path)
            document.close()
            self.assertIsNone(self._raster(path, config))

    def test_raster_gate_thresholds_are_config_driven(self):
        with tempfile.TemporaryDirectory() as temp:
            low = self._scan_pdf(Path(temp) / "low.pdf", (992, 1403), (595, 842))
            relaxed = copy.deepcopy(load_config())
            relaxed["parsing"]["page_quality"]["min_effective_dpi"] = 100
            self.assertFalse(self._raster(low, relaxed)["low_resolution"])
            off = copy.deepcopy(load_config())
            off["parsing"]["page_quality"]["enabled"] = False
            self.assertIsNone(self._raster(low, off))

    def test_parse_marks_the_page_and_the_document(self):
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self._scan_pdf(root / "LOW1.pdf", (992, 1403), (595, 842))
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(path, "LOW1", force=True, config=config,
                                    out_dir=root / "parsed")
        self.assertEqual(parsed["quality"]["low_resolution_pages"], [1])
        self.assertIn("LOW_RESOLUTION", parsed["pages"][0]["quality_flags"])
        self.assertTrue(parsed["pages"][0]["raster"]["low_resolution"])

    def test_imported_photo_is_judged_on_its_normalized_pixels(self):
        # 图片导入的页面没有可换算的物理尺寸（MuPDF 直接按图开页），
        # 有效 dpi 只能来自归一化时记录的像素数。
        config = copy.deepcopy(load_config())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config["paths"]["work"] = str(root / "work")
            small = root / "small.png"
            Image.new("RGB", (992, 1403), (250, 250, 250)).save(small)
            parsed = parse_document(small, "IMPLow", force=True, config=config,
                                    out_dir=root / "parsed")
            big = root / "big.png"
            Image.new("RGB", (2480, 3508), (250, 250, 250)).save(big)
            parsed_high = parse_document(big, "IMPMore", force=True, config=config,
                                         out_dir=root / "parsed")
        self.assertEqual(parsed["pages"][0]["raster"]["pixels"], [992, 1403])
        self.assertAlmostEqual(parsed["pages"][0]["raster"]["effective_dpi"], 120.0, delta=1.0)
        self.assertIn("LOW_RESOLUTION", parsed["pages"][0]["quality_flags"])
        self.assertEqual(parsed_high["quality"]["low_resolution_pages"], [])

    def test_low_resolution_forces_review_and_can_be_switched_off(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        field = {
            "path": "bs.account_name", "label": "户名", "value": "示例公司",
            "value_type": "string", "required": True, "risk": "party_name",
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "grounding_present": 1},
            "meta": {}, "decision": "review", "validators": [], "review_reasons": [],
        }
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 595, "height": 842,
                             "quality_flags": [], "raster": None}]}
        doc = {"document_id": "LOW1", "doc_type": "BANK_STATEMENT"}
        clean = decide_field(copy.deepcopy(field), doc, parsed, config, [], Calibrator(config))
        self.assertEqual(clean["decision"], "accept")
        parsed["pages"][0]["quality_flags"] = ["LOW_RESOLUTION"]
        blocked = decide_field(copy.deepcopy(field), doc, parsed, config, [], Calibrator(config))
        self.assertEqual(blocked["decision"], "review")
        self.assertIn("LOW_RESOLUTION", blocked["review_reasons"])
        config["validation"]["review_on_low_resolution"] = False
        allowed = decide_field(copy.deepcopy(field), doc, parsed, config, [], Calibrator(config))
        self.assertEqual(allowed["decision"], "accept")

    @staticmethod
    def _signal_clean_field(path: str, risk: str, value: str = "示例值") -> Dict:
        """A field that every *field-level* signal likes, so only the page gate can stop it."""
        return {
            "path": path, "label": path.split(".")[-1], "value": value,
            "value_type": "string", "required": True, "risk": risk,
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "grounding_present": 1},
            "meta": {}, "decision": "review", "validators": [], "review_reasons": [],
        }

    @staticmethod
    def _flagged_pages(raster: Optional[Dict]) -> Dict:
        return {"page_count": 1, "quality": {"parse_score": 0.99},
                "pages": [{"page": 1, "width": 595, "height": 842,
                           "quality_flags": ["LOW_RESOLUTION"], "raster": raster}]}

    def test_risk_tiers_grade_the_page_gate(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        config["parsing"]["page_quality"]["risk_min_dpi"] = {
            "enabled": True, "free_text": 110, "monetary": 220}
        parsed = self._flagged_pages({"effective_dpi": 120.0, "low_resolution": True})
        doc = {"document_id": "TIER1", "doc_type": "BANK_STATEMENT"}
        text = decide_field(self._signal_clean_field("bs.bank_name", "free_text"),
                            doc, parsed, config, [], Calibrator(config))
        self.assertEqual(text["decision"], "accept")
        money = decide_field(self._signal_clean_field("bs.closing_balance", "monetary"),
                             doc, parsed, config, [], Calibrator(config))
        self.assertIn("LOW_RESOLUTION", money["review_reasons"])
        # 关掉分级回到整页一刀切：同一页、同一个字段仍然得复核。
        config["parsing"]["page_quality"]["risk_min_dpi"]["enabled"] = False
        flat = decide_field(self._signal_clean_field("bs.bank_name", "free_text"),
                            doc, parsed, config, [], Calibrator(config))
        self.assertIn("LOW_RESOLUTION", flat["review_reasons"])

    def test_unmeasured_page_is_never_exempted(self):
        # 拍照预处理那条路也会打 LOW_RESOLUTION，但页面不一定量得出有效 dpi；
        # 没有数字可依据时一律拦，不能猜「这一档总该够用」。
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        config["parsing"]["page_quality"]["risk_min_dpi"] = {"enabled": True, "free_text": 50}
        blocked = decide_field(
            self._signal_clean_field("bs.bank_name", "free_text"),
            {"document_id": "TIER2", "doc_type": "BANK_STATEMENT"},
            self._flagged_pages(None), config, [], Calibrator(config))
        self.assertIn("LOW_RESOLUTION", blocked["review_reasons"])


class QrCrosscheckTests(unittest.TestCase):
    """二维码只补空与核对，不覆盖 OCR，也不替人做决定。"""

    PAYLOAD = "01,31,,1234567890123456,1130.00,20250102,,ABCD"

    @staticmethod
    def _parsed(payload: str) -> Dict:
        return {
            "document_id": "QR1", "doc_type": "VAT_INVOICE", "page_count": 1,
            "engine": "rapidocr", "source": "synthetic.png",
            "quality": {"parse_score": 0.99},
            "pages": [{"page": 1, "width": 595, "height": 842, "lines": [],
                       "tables": [], "codes": [
                           {"text": payload, "bbox": [20, 20, 120, 120],
                            "confidence": 0.99}]}],
        }

    def test_qr_fills_fields_the_ocr_missed(self):
        parsed = self._parsed(self.PAYLOAD)
        result = extract_document(parsed)
        number = result["fields"]["vi.invoice_number"]
        issue_date = result["fields"]["vi.invoice_date"]
        total = result["fields"]["vi.total_with_tax"]
        self.assertEqual(number["value"], "1234567890123456")
        self.assertEqual(issue_date["value"], "2025-01-02")
        self.assertEqual(total["value"], "1130.00")
        for field in (number, issue_date, total):
            self.assertEqual(field["meta"]["source"], "qr")
            self.assertEqual(field["grounding"]["page"], 1)
            self.assertGreaterEqual(field["confidence"], 0.99)

    def test_qr_conflict_keeps_ocr_value_and_forces_review(self):
        config = copy.deepcopy(load_config())
        parsed = self._parsed(self.PAYLOAD)
        fields = {
            "vi.invoice_number": {
                "path": "vi.invoice_number", "value": "9999999999999999",
                "value_type": "string", "required": True, "risk": "identity",
                "extractor_id": "regex:发票号码", "confidence": 0.94,
                "grounding": {"page": 1, "bbox": [465, 31, 572, 41]},
                "signals": {}, "meta": {},
            }
        }
        specs = [{"path": "vi.invoice_number", "type": "string", "risk": "identity",
                  "qr_field": "invoice_number"}]
        _qr_crosscheck(fields, specs, parsed, "VAT_INVOICE", config,
                       {"unit": None, "unit_factor": None})
        field = fields["vi.invoice_number"]
        self.assertEqual(field["value"], "9999999999999999")
        self.assertEqual(field["meta"]["qr_conflict"]["qr"], "1234567890123456")
        self.assertEqual(field["signals"]["extractor_agreement"], 0.0)

        doc = {"document_id": "QR1", "doc_type": "VAT_INVOICE", "fields": fields}
        validators: Dict[str, list] = {}
        validate_document_fields(doc, parsed, config, validators)
        statuses = [item["status"] for item in validators["vi.invoice_number"]]
        self.assertIn("fail", statuses)
        decided = decide_field(field, doc, parsed, config,
                               validators["vi.invoice_number"], Calibrator(config))
        self.assertEqual(decided["decision"], "review")
        self.assertIn("QR_CONFLICT", decided["review_reasons"])


class ReceiptValidatorTests(unittest.TestCase):
    """两类票据的票面勾稽关系。"""

    @staticmethod
    def _field(value) -> Dict:
        return {"value": value, "status": "ok", "meta": {}}

    def _vat_fields(self) -> Dict:
        return {
            "vi.total_amount": self._field("1000.00"),
            "vi.total_tax": self._field("130.00"),
            "vi.total_with_tax": self._field("1130.00"),
            "vi.items[0].unit_price": self._field("100.00"),
            "vi.items[0].quantity": self._field("10.00"),
            "vi.items[0].amount": self._field("1000.00"),
            "vi.items[0].tax_rate": self._field("13%"),
            "vi.items[0].tax_amount": self._field("130.00"),
        }

    @staticmethod
    def _status(validators: Dict, rule_id: str) -> str:
        statuses = [item["status"] for item in validators.get("vi.total_amount", [])
                    if item["id"] == rule_id]
        return statuses[0] if statuses else ""

    def test_vat_invoice_arithmetic(self):
        config = load_config()
        fields = self._vat_fields()
        validators: Dict[str, list] = {}
        validate_vat_invoice({"fields": fields}, config, validators)
        self.assertEqual(self._status(validators, "V-VI-01"), "pass")
        self.assertEqual(self._status(validators, "V-VI-04"), "pass")
        fields["vi.total_with_tax"]["value"] = "1131.00"
        broken: Dict[str, list] = {}
        validate_vat_invoice({"fields": fields}, config, broken)
        self.assertEqual(self._status(broken, "V-VI-01"), "fail")

    def test_rate_text_follows_the_percent_sign_not_the_magnitude(self):
        """1% 征收率曾被读成 100%：按量级猜百分数在不大于 1 的税率上必错。"""
        cases = [("13%", Decimal("0.13")), ("13.0000", Decimal("0.13")),
                 ("0.13", Decimal("0.13")), ("1%", Decimal("0.01")),
                 ("0.5%", Decimal("0.005")), ("1％", Decimal("0.01")),
                 ("9 %", Decimal("0.09")), ("0%", Decimal("0")),
                 ("1", Decimal("1")), ("", None), (None, None)]
        for text, expected in cases:
            self.assertEqual(_percent(text), expected, text)

    def test_one_percent_rate_passes_the_tax_crosscheck(self):
        config = load_config()
        fields = self._vat_fields()
        fields["vi.total_amount"]["value"] = "1000.00"
        fields["vi.total_tax"]["value"] = "10.00"
        fields["vi.total_with_tax"]["value"] = "1010.00"
        fields["vi.items[0].tax_rate"]["value"] = "1%"
        fields["vi.items[0].tax_amount"]["value"] = "10.00"
        validators: Dict[str, list] = {}
        validate_vat_invoice({"fields": fields}, config, validators)
        item = [entry for entry in validators["vi.items[0].tax_amount"]
                if entry["id"] == "V-VI-02"][0]
        self.assertEqual(item["status"], "pass")
        self.assertIn("1000.00 × 0.01 = 10.00", item["detail"])
        fields["vi.items[0].tax_amount"]["value"] = "100.00"
        broken: Dict[str, list] = {}
        validate_vat_invoice({"fields": fields}, config, broken)
        failed = [entry for entry in broken["vi.items[0].tax_amount"]
                  if entry["id"] == "V-VI-02"][0]
        self.assertEqual(failed["status"], "fail")

    def test_customs_arithmetic_and_declaration_linkage(self):
        config = load_config()
        fields = {
            "cv.total_amount": self._field("13000.00"),
            "cv.payment_number": self._field("123456789012345678"),
            "cv.declaration_number": self._field("123456789012345678"),
            "cv.items[0].dutiable_value": self._field("100000.00"),
            "cv.items[0].tax_rate": self._field("13.0000"),
            "cv.items[0].tax_amount": self._field("13000.00"),
        }
        validators: Dict[str, list] = {}
        validate_customs_payment({"fields": fields}, config, validators)
        self.assertEqual(validators["cv.total_amount"][0]["id"], "V-CV-02")
        self.assertEqual(validators["cv.total_amount"][0]["status"], "pass")
        self.assertEqual(validators["cv.items[0].tax_amount"][0]["status"], "pass")
        self.assertEqual(validators["cv.payment_number"][0]["status"], "pass")

        fields["cv.declaration_number"]["value"] = "999999999999999999"
        broken: Dict[str, list] = {}
        validate_customs_payment({"fields": fields}, config, broken)
        statuses = [item["status"] for item in broken["cv.payment_number"]
                    if item["id"] == "V-CV-03"]
        self.assertEqual(statuses, ["fail"])


class PartialBorderTests(unittest.TestCase):
    """贴边扫描只碰一条边时不算拍漏。"""

    def test_one_touched_side_is_not_partial_but_two_are(self):
        config = load_config()
        options = preprocess.geometry_options(config)
        edge_scan = _perspective_shot([[100, 60], [995, 60], [995, 1240], [100, 1240]])
        _corners, _area, partial = preprocess.document_quad(
            preprocess._gray(edge_scan), options)
        self.assertFalse(partial)
        cut_off = _perspective_shot([[6, 6], [760, 20], [900, 1280], [100, 1150]])
        _corners, _area, partial = preprocess.document_quad(
            preprocess._gray(cut_off), options)
        self.assertTrue(partial)
        stricter = dict(options, partial_min_border_sides=1)
        _corners, _area, partial = preprocess.document_quad(
            preprocess._gray(edge_scan), stricter)
        self.assertTrue(partial)


class ReceiptEndToEndTests(unittest.TestCase):
    """两类票据用合成文本层 PDF 走完整解析 → 抽取 → 校验，不依赖真实样张。"""

    @staticmethod
    def _pdf(lines: list, doc_id: str) -> tuple:
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / f"{doc_id}.pdf"
        document = fitz.open()
        page = synth._new_page(document)
        for text, x, y in lines:
            synth.draw_text(page, x, y, text, size=9)
        document.save(path, garbage=4, deflate=True)
        document.close()
        return temp, path

    @staticmethod
    def _run(path: Path, doc_id: str) -> tuple:
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        work = path.parent / "work"
        config["paths"]["work"] = str(work)
        parsed = parse_document(path, doc_id, force=True, config=config,
                                out_dir=work / "parsed")
        extracted = extract_document(parsed, config)
        docs = assess([extracted], {doc_id: parsed}, config, cross_document=False)
        return config, parsed, docs[0]

    def test_vat_invoice_text_layer_pipeline(self):
        temp, path = self._pdf([
            ("增值税专用发票", 210, 60),
            ("发票号码：2025000000000001", 50, 90),
            ("开票日期：2025年03月04日", 400, 90),
            ("购买方", 50, 130),
            ("名称：", 50, 150), ("示例甲公司", 110, 150),
            ("纳税人识别号：", 50, 170), ("91110000MA00000001", 150, 170),
            ("销售方", 320, 130),
            ("名称：", 320, 150), ("示例乙公司", 380, 150),
            ("纳税人识别号：", 320, 170), ("91110000MA00000002", 420, 170),
            ("合计：", 50, 520), ("1000.00", 300, 520), ("130.00", 420, 520),
            ("价税合计（大写）壹仟壹佰叁拾元整（小写）￥1130.00", 50, 540),
            ("项目名称", 50, 600), ("规格型号", 140, 600), ("单位", 230, 600),
            ("数量", 270, 600), ("单价", 310, 600), ("金额", 370, 600),
            ("税率/征收率", 440, 600), ("税额", 520, 600),
            ("示例货物", 50, 625), ("规格A", 140, 625), ("台", 230, 625),
            ("10", 270, 625), ("100.00", 310, 625), ("1000.00", 370, 625),
            ("13%", 440, 625), ("130.00", 520, 625),
            ("合同编号：HT2025001", 50, 700),
            ("开票人：示例经办", 50, 720),
        ], "SYNVAT")
        with temp:
            _config, _parsed, doc = self._run(path, "SYNVAT")
        self.assertEqual(doc["doc_type"], "VAT_INVOICE")
        fields = doc["fields"]
        expected = {
            "vi.invoice_number": "2025000000000001",
            "vi.invoice_date": "2025-03-04",
            "vi.buyer_name": "示例甲公司",
            "vi.buyer_tax_id": "91110000MA00000001",
            "vi.seller_name": "示例乙公司",
            "vi.seller_tax_id": "91110000MA00000002",
            "vi.total_amount": "1000.00",
            "vi.total_tax": "130.00",
            "vi.total_with_tax": "1130.00",
            "vi.contract_number": "HT2025001",
            "vi.issuer": "示例经办",
            "vi.items[0].name": "示例货物",
            "vi.items[0].quantity": "10.00",
            "vi.items[0].unit_price": "100.00",
            "vi.items[0].amount": "1000.00",
            "vi.items[0].tax_rate": "13%",
            "vi.items[0].tax_amount": "130.00",
        }
        for path_name, value in expected.items():
            self.assertIn(path_name, fields)
            self.assertEqual(fields[path_name]["value"], value, path_name)
            self.assertIsNotNone(fields[path_name]["grounding"], path_name)
        statuses: Dict[str, str] = {}
        for path_name in ("vi.total_with_tax", "vi.total_amount", "vi.total_tax",
                          "vi.items[0].amount", "vi.items[0].tax_rate",
                          "vi.items[0].tax_amount", "vi.items[0].unit_price",
                          "vi.items[0].quantity"):
            for item in fields[path_name]["validators"]:
                statuses.setdefault(item["id"], item["status"])
        for rule in ("V-VI-01", "V-VI-02", "V-VI-03", "V-VI-04"):
            self.assertEqual(statuses.get(rule), "pass", rule)
        self.assertEqual(fields["vi.invoice_number"]["decision"], "review",
                         "冷启动新类型核心字段应如实进复核")
        self.assertEqual(fields["vi.invoice_number"]["confidence_basis"], "heuristic")

    def test_customs_payment_text_layer_pipeline(self):
        temp, path = self._pdf([
            ("海关进口增值税专用缴款书", 160, 60),
            ("填发日期：2026年02月03日", 350, 90),
            ("缴款单位", 50, 130),
            ("名称：", 50, 150), ("示例丙公司", 110, 150),
            ("帐号：", 50, 170), ("91110000MA00000003", 110, 170),
            ("开户银行：", 50, 190), ("示例银行", 130, 190),
            ("收入机关：", 320, 150), ("示例税务局", 390, 150),
            ("科目：", 320, 170), ("进口增值税", 370, 170),
            ("预算级次：", 320, 190), ("中央", 400, 190),
            ("收款国库：", 50, 220), ("示例国库", 130, 220),
            ("税号", 50, 300), ("货物名称", 150, 300), ("数量", 240, 300),
            ("单位", 280, 300), ("完税价格(¥)", 330, 300), ("税率(%)", 430, 300),
            ("税款金额(¥)", 500, 300),
            ("8480719090", 50, 325), ("示例模具", 150, 325), ("1.00", 240, 325),
            ("套", 280, 325), ("100000.00", 330, 325), ("13.0000", 430, 325),
            ("13000.00", 500, 325),
            ("合计 ￥13000.00", 50, 380),
            ("金额人民币（大写）壹万叁仟元整", 50, 400),
            ("报关单编号：123456789012345678", 50, 440),
            ("申请单位编号：3112000001", 320, 440),
            ("缴款期限：2026年02月20日前", 50, 460),
            ("No. 123456789012345678L01", 100, 100),
            ("号码：", 50, 100),
        ], "SYNCV")
        with temp:
            _config, _parsed, doc = self._run(path, "SYNCV")
        self.assertEqual(doc["doc_type"], "CUSTOMS_VAT_PAYMENT")
        fields = doc["fields"]
        expected = {
            "cv.payment_number": "123456789012345678L01",
            "cv.issue_date": "2026-02-03",
            "cv.payer_name": "示例丙公司",
            "cv.payer_tax_id": "91110000MA00000003",
            "cv.bank_name": "示例银行",
            "cv.revenue_authority": "示例税务局",
            "cv.subject": "进口增值税",
            "cv.total_amount": "13000.00",
            "cv.declaration_number": "123456789012345678",
            "cv.application_number": "3112000001",
            "cv.payment_deadline": "2026-02-20",
            "cv.items[0].hs_code": "8480719090",
            "cv.items[0].name": "示例模具",
            "cv.items[0].dutiable_value": "100000.00",
            "cv.items[0].tax_rate": "13.0000",
            "cv.items[0].tax_amount": "13000.00",
        }
        for path_name, value in expected.items():
            self.assertIn(path_name, fields)
            self.assertEqual(fields[path_name]["value"], value, path_name)
            self.assertIsNotNone(fields[path_name]["grounding"], path_name)
        statuses: Dict[str, str] = {}
        for path_name in ("cv.total_amount", "cv.payment_number", "cv.declaration_number",
                          "cv.items[0].dutiable_value", "cv.items[0].tax_rate",
                          "cv.items[0].tax_amount"):
            for item in fields[path_name]["validators"]:
                statuses.setdefault(item["id"], item["status"])
        self.assertEqual(statuses.get("V-CV-01"), "pass")
        self.assertEqual(statuses.get("V-CV-02"), "pass")
        self.assertEqual(statuses.get("V-CV-03"), "pass")
        self.assertEqual(fields["cv.payment_number"]["decision"], "review")


class HardSetTests(unittest.TestCase):
    """P0-1 难例集：生成、双栏字段覆盖、按退化归因。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.manifest = hardset.generate_hard_set(cls.root / "samples", cls.root / "labels",
                                                 per_kind=1)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_generation_covers_every_degradation(self):
        documents = self.manifest["documents"]
        self.assertEqual(len(documents), len(hardset.DEGRADATIONS))
        self.assertEqual({item["degradation"] for item in documents},
                         set(hardset.DEGRADATIONS))
        for item in documents:
            pdf = self.root / "samples" / f"{item['document_id']}.pdf"
            label = self.root / "labels" / f"{item['document_id']}.json"
            self.assertTrue(pdf.exists(), pdf)
            self.assertTrue(label.exists(), label)
            payload = json.loads(label.read_text(encoding="utf-8"))
            self.assertEqual(payload["doc_type"], item["doc_type"])
            self.assertGreater(len(payload["fields"]), 0)
            with fitz.open(pdf) as doc:
                self.assertEqual(doc.page_count, item["page_count"])

    def test_raster_ground_truth_stays_on_page(self):
        for item in self.manifest["documents"]:
            if item["format"] != "scan":
                continue
            label = json.loads((self.root / "labels" / f"{item['document_id']}.json")
                               .read_text(encoding="utf-8"))
            for path, entry in label["fields"].items():
                x0, y0, x1, y1 = entry["bbox"]
                self.assertGreaterEqual(x0, -2, f"{item['document_id']} {path}")
                self.assertGreaterEqual(y0, -2, f"{item['document_id']} {path}")
                self.assertLessEqual(x1, 595.28 + 2, f"{item['document_id']} {path}")
                self.assertLessEqual(y1, 841.89 + 2, f"{item['document_id']} {path}")

    def test_two_column_builders_cover_configured_fields(self):
        company = synth.make_company(random.Random(3), 5)
        doc = fitz.open()
        credit = hardset.build_two_column_credit_report(
            doc, company, synth.make_credit_data(random.Random(4), company))
        doc.close()
        expected = {spec["path"] for spec in field_specs("CREDIT_REPORT_ENTERPRISE", self.config)}
        self.assertEqual(set(credit), expected)

        doc = fitz.open()
        license_gt = hardset.build_two_column_business_license(doc, company)
        doc.close()
        expected = {spec["path"] for spec in field_specs("BUSINESS_LICENSE", self.config)}
        self.assertEqual(set(license_gt), expected)

    def test_degradation_aggregation_names_the_failure_mode(self):
        def record(doc_id, decision, correct, grounded):
            return {"document_id": doc_id, "decision": decision, "correct": correct,
                    "risk": "date", "grounded": grounded, "review_status": None,
                    "required": True,
                    "reasons": [], "signals": {}, "extractor_id": "test", "pred": "x"}

        records = [record("A1", "accept", True, True),
                   record("A1", "accept", False, True),
                   record("B1", "review", True, False)]
        rows = hardset._aggregate_by_degradation(
            records, {"A1": "jpeg_heavy", "B1": "jpeg_heavy"})
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["name"], "jpeg_heavy")
        self.assertEqual(row["documents"], 2)
        self.assertEqual(row["field_count"], 3)
        self.assertEqual(row["accept_count"], 2)
        self.assertEqual(row["accepted_error_count"], 1)
        self.assertAlmostEqual(row["auto_accept_rate"], 0.6667)
        self.assertAlmostEqual(row["grounding_hit_rate"], 0.6667)

    def test_page_signals_order_compression_and_blur(self):
        """第二根轴的指标要先量对方向，再谈能不能当门。"""
        page = np.full((800, 600), 255, dtype=np.uint8)
        for y in range(40, 760, 28):
            cv2.putText(page, "Sample text 1234567890", (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 0, 1, cv2.LINE_AA)

        def jpeg(quality: int) -> np.ndarray:
            ok, buffer = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, quality])
            self.assertTrue(ok)
            return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE).astype(np.float64)

        low = hardset.page_signals(jpeg(55))
        high = hardset.page_signals(jpeg(95))
        self.assertGreater(low["blockiness"], high["blockiness"])
        self.assertLess(low["hf_ratio"], high["hf_ratio"])

        blurred = cv2.GaussianBlur(page, (0, 0), 2.0).astype(np.float64)
        self.assertLess(hardset.page_signals(blurred)["laplacian_variance"],
                        hardset.page_signals(page.astype(np.float64))["laplacian_variance"])
        self.assertEqual(hardset.page_signals(np.zeros((20, 20))), {})

    def test_line_metrics_react_to_blur_and_flat_crop(self):
        """P0-5c-2 的裁片指标要先证明能分辨清晰度，再谈它能不能区分错值。"""
        page = np.full((160, 520), 255, dtype=np.uint8)
        cv2.putText(page, "Sample 12345", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 0, 2, cv2.LINE_AA)
        box = [8, 60, 300, 110]
        sharp = hardset._line_metrics(page.astype(np.float32), box)
        blurred = hardset._line_metrics(
            cv2.GaussianBlur(page, (0, 0), 2.0).astype(np.float32), box)
        self.assertTrue(sharp and blurred)
        self.assertGreater(sharp["edge_sharp"], blurred["edge_sharp"])
        self.assertGreater(sharp["laplacian_rel"], blurred["laplacian_rel"])
        self.assertEqual(hardset._line_metrics(
            np.full((60, 60), 255, dtype=np.float32), [5, 5, 50, 50]), {})

    def test_auc_averages_ties(self):
        self.assertEqual(hardset._auc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]), 1.0)
        self.assertEqual(hardset._auc([0.1, 0.2, 0.3, 0.4], [True, True, False, False]), 0.0)
        self.assertEqual(hardset._auc([1.0, 1.0, 1.0, 1.0],
                                      [True, False, True, False]), 0.5)
        self.assertIsNone(hardset._auc([1.0, 2.0], [True, True]))

    def test_sweep_cells_rasterise_the_same_base_documents(self):
        """扫描网格必须配对：同一份基文档栅格化进每一格，格与格之间只差退化参数。

        不配准时每一格都在生成自己的文档，同一 (dpi, JPEG) 两次跑能得出 37 与 1 个静默错误，
        这张表面就不能用来定档位。
        """
        doc = fitz.open()
        gt, _ = hardset._build_base("BANK_STATEMENT",
                                    hardset._company_for(hardset.HARD_SET_SEED, 0),
                                    random.Random(7), doc)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.pdf"
            doc.save(base, garbage=4, deflate=True)
            doc.close()
            per_cell = []
            for dpi in (120, 260):
                scanned = hardset._rasterise(
                    base, Path(tmp) / f"scan-{dpi}.pdf", copy.deepcopy(gt),
                    random.Random(dpi), "low_resolution",
                    {"dpi": dpi, "quality": 55, "angle": 0.2})
                per_cell.append({path: item["value"] for path, item in scanned.items()})
            self.assertTrue(per_cell[0])
            self.assertEqual(per_cell[0], per_cell[1])

            # 退化参数也必须只由文档序号决定：同一个种子在同一个 dpi 上要给出同一张图，
            # 否则格与格之间比的是旋转角抽签，不是分辨率。
            spec = {"dpi": 150, "quality": 55, "angle": 0.2}
            shots = []
            for name in ("a", "b"):
                path = Path(tmp) / f"same-{name}.pdf"
                hardset._rasterise(base, path, copy.deepcopy(gt),
                                   random.Random(hardset.HARD_SET_SEED * 11),
                                   "low_resolution", spec)
                with fitz.open(path) as scanned:
                    shots.append(scanned[0].get_pixmap(dpi=72).samples)
            self.assertEqual(shots[0], shots[1])


class SchemaCoverageTests(unittest.TestCase):
    def test_business_license_individual_operator_labels(self):
        """个体户执照的「经营者 / 组成形式 / 经营场所」要落到同名字段上。

        八种执照版式里负责人一栏随主体类型改名，个体户没有「法定代表人」四个字；
        只认法定代表人的话这类执照的负责人字段会整份判缺失。
        """
        doc = fitz.open()
        page = synth._new_page(doc, width=595, height=842)
        lines = [
            ("营业执照", 16),
            ("统一社会信用代码：91110000MA01ABCD12", 10),
            ("名称：示例商行", 10),
            ("组成形式：个人经营", 10),
            ("经营者：张三", 10),
            ("经营场所：示例市示例路1号", 10),
            ("注册资本：100000元", 10),
            ("成立日期：2020年01月02日", 10),
            ("登记机关：示例市市场监督管理局", 10),
        ]
        for index, (text, size) in enumerate(lines):
            synth.draw_text(page, 50, 100 + index * 30, text, size=size)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "individual.pdf"
            doc.save(path, garbage=4, deflate=True)
            doc.close()
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(path, "IND", force=True, config=config,
                                    out_dir=root / "parsed")
            self.assertEqual(parsed["doc_type"], "BUSINESS_LICENSE")
            fields = extract_document(parsed, config)["fields"]
            self.assertEqual(fields["bl.company_name"]["value"], "示例商行")
            self.assertEqual(fields["bl.legal_representative"]["value"], "张三")
            self.assertEqual(fields["bl.entity_type"]["value"], "个人经营")
            self.assertEqual(fields["bl.address"]["value"], "示例市示例路1号")

    def test_income_statement_new_standard_rows_are_extracted(self):
        """新准则利润表的减值/处置/公允价值/综合收益行要能落到字段上。

        这几行决定「利润里有多少不是主营赚来的」，信贷审查必须拿到；老模板没有
        这些行时字段留空，不影响原有行。
        """
        data = synth.make_income_data(random.Random(7))
        data["period_start"] = date(2025, 1, 1)
        data["period_end"] = date(2025, 12, 31)
        data["credit_impairment_loss"] = Decimal("-123456.78")
        data["asset_impairment_loss"] = Decimal("-7890.00")
        data["asset_disposal_gain"] = Decimal("50000.00")
        data["fair_value_change_gain"] = Decimal("12345.00")
        data["total_comprehensive_income"] = data["net_profit"] + Decimal("350.00")
        company = synth.make_company(random.Random(7), 0)
        doc = fitz.open()
        synth.build_income_statement(doc, company, data, use_wan=False, extra_rows=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "income.pdf"
            doc.save(path, garbage=4, deflate=True)
            doc.close()
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(path, "INC", force=True, config=config,
                                    out_dir=root / "parsed")
            self.assertEqual(parsed["doc_type"], "INCOME_STATEMENT")
            fields = extract_document(parsed, config)["fields"]
            for name in ("credit_impairment_loss", "asset_impairment_loss",
                         "asset_disposal_gain", "fair_value_change_gain",
                         "total_comprehensive_income"):
                field = fields[f"is.{name}"]
                self.assertEqual(Decimal(field["value"]), data[name], name)
                self.assertIsNotNone(field["grounding"], name)
            self.assertEqual(Decimal(fields["is.net_profit"]["value"]), data["net_profit"])

    def test_million_unit_is_recognised(self):
        """「人民币百万元」要先于「人民币元」命中，否则金额会被当成元原样保留。"""
        def line(text):
            return {"text": text, "bbox": [0, 0, 200, 10], "confidence": 0.99,
                    "parts": [{"text": text, "bbox": [0, 0, 200, 10], "confidence": 0.99}]}

        hit = extract_unit([line("（金额单位：人民币百万元）")])
        self.assertEqual(hit["value"], "人民币百万元")
        self.assertEqual(extract_unit([line("金额单位：万元")])["value"], "万元")

    def test_year_header_anchors_merge_split_suffix(self):
        """文本层会把「2024」「年度」拆成两段，锚点要取整格的右边界。"""
        line = {"parts": [
            {"text": "附注", "bbox": [345.86, 133.46, 368.74, 143.16]},
            {"text": "2024", "bbox": [408.0, 132.72, 431.08, 144.6]},
            {"text": "年度", "bbox": [431.85, 133.4, 452.13, 143.1]},
            {"text": "2023", "bbox": [487.37, 132.59, 510.45, 144.66]},
            {"text": "年度", "bbox": [511.22, 133.46, 531.5, 143.16]},
        ]}
        options = copy.deepcopy(load_config()["doc_types"]["INCOME_STATEMENT"]
                                ["tables"]["income"]["line_fallback"])
        self.assertEqual(_statement_header_anchors(line, options), [452.13, 531.5])

    def test_income_validator_counts_new_standard_rows(self):
        """新准则的四行参与营业利润；漏掉时这几行非零的报表会被误判为跨字段失败。"""
        config = copy.deepcopy(load_config())
        values = {
            "revenue": "500000", "cost_of_sales": "300000", "taxes_surcharges": "5000",
            "selling_expenses": "20000", "admin_expenses": "30000", "rd_expenses": "10000",
            "finance_expenses": "1000", "other_income": "2000", "investment_income": "3000",
            "fair_value_change_gain": "500", "credit_impairment_loss": "-800",
            "asset_impairment_loss": "-1200", "asset_disposal_gain": "400",
            "operating_profit": "137900", "non_operating_income": "1500",
            "non_operating_expense": "2400", "total_profit": "137000",
            "income_tax": "34250", "net_profit": "102750",
        }
        fields = {f"is.{key}": {"value": value, "value_type": "decimal"}
                  for key, value in values.items()}
        validators = {}
        validate_income_statement({"fields": fields, "currency_unit": "CNY_MILLION"},
                                  config, validators)
        status = {item["id"]: item["status"] for item in validators["is.operating_profit"]}
        self.assertEqual(status["V-IS-01"], "pass")
        self.assertEqual(status["V-IS-02"], "pass")
        net_status = {item["id"]: item["status"] for item in validators["is.net_profit"]}
        self.assertEqual(net_status["V-IS-03"], "pass")

        fields["is.operating_profit"]["value"] = "140000"
        invalid = {}
        validate_income_statement({"fields": fields, "currency_unit": "CNY_MILLION"},
                                  config, invalid)
        self.assertEqual(invalid["is.operating_profit"][0]["status"], "fail")

    def test_annual_report_income_statement_line_fallback(self):
        """年报式利润表没有「项目/本期金额/上期金额」表头，也没有框线。

        真实版式是「项目（无表头） 附注 2024 年度 2023 年度」，金额右对齐。
        行回落要按行名和年份列右边界取数并按百万元折算，同时因为没有表头覆盖
        与框线证据，所有值都必须进复核，不能借高置信度静默接受。
        """
        doc = fitz.open()
        page = synth._new_page(doc)
        synth.draw_text(page, 58, 60, "合并利润表", size=18, bold=True)
        synth.draw_text(page, 58, 85, "2024 年度", size=10)
        synth.draw_text(page, 55, 105, "（金额单位：人民币百万元）", size=9)
        x_note, x_current, x_prior = 340, 490, 560
        synth.draw_text(page, x_note, 132, "附注", size=9)
        synth.draw_text(page, x_current, 132, "2024 年度", size=9, align="right")
        synth.draw_text(page, x_prior, 132, "2023 年度", size=9, align="right")
        rows = [
            ("一、营业收入", "五、 35", "500,000", "450,000"),
            ("减：营业成本", "五、 35", "300,000", "280,000"),
            ("税金及附加", "五、 36", "5,000", "4,500"),
            ("销售费用", "五、 37", "20,000", "18,000"),
            ("管理费用", "五、 38", "30,000", "28,000"),
            ("研发费用", "六、 1", "10,000", "9,000"),
            ("财务费用", "五、 39", "1,000", "900"),
            ("加：其他收益", "五、 40", "2,000", "1,800"),
            ("投资收益（损失以 “-” 号填列）", "五、 41", "3,000", "2,500"),
            ("公允价值变动损益（损失以 “-” 号填列）", "", "500", "400"),
            ("信用减值损失（损失以 “-” 号填列）", "五、 42", "(800)", "(700)"),
            ("资产减值损失（损失以 “-” 号填列）", "五、 43", "(1,200)", "(1,000)"),
            ("资产处置收益（损失以 “-” 号填列）", "", "400", "300"),
            ("二、营业利润", "", "137,900", "120,000"),
            ("加：营业外收入", "五、 44", "1,500", "1,300"),
            ("减：营业外支出", "五、 45", "2,400", "2,000"),
            ("三、利润总额", "", "137,000", "119,300"),
            ("减：所得税费用", "五、 46", "34,250", "29,825"),
            ("四、净利润", "", "102,750", "89,475"),
            ("六、综合收益总额", "", "103,000", "89,600"),
        ]
        y = 155
        for label, note, current, prior in rows:
            synth.draw_text(page, 58, y, label, size=9)
            synth.draw_text(page, x_note, y, note, size=9)
            synth.draw_text(page, x_current, y, current, size=9, align="right")
            synth.draw_text(page, x_prior, y, prior, size=9, align="right")
            y += 16
        synth.draw_text(page, 58, y + 20, "后附的财务报表附注为本财务报表的组成部分。", size=8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "annual-income.pdf"
            doc.save(path, garbage=4, deflate=True)
            doc.close()
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(path, "ANNUAL", force=True, config=config,
                                    out_dir=root / "parsed")
            self.assertEqual(parsed["doc_type"], "INCOME_STATEMENT")
            self.assertFalse([table for page_data in parsed["pages"]
                              for table in page_data["tables"]])
            extracted = extract_document(parsed, config)
            fields = extracted["fields"]
            self.assertEqual(fields["is.currency_unit"]["value"], "CNY_MILLION")
            expected = {
                "revenue": 500000, "cost_of_sales": 300000,
                "operating_profit": 137900, "total_profit": 137000,
                "net_profit": 102750, "prior_revenue": 450000,
                "prior_net_profit": 89475, "credit_impairment_loss": -800,
                "fair_value_change_gain": 500,
                "total_comprehensive_income": 103000,
            }
            for name, million in expected.items():
                field = fields[f"is.{name}"]
                self.assertEqual(Decimal(field["value"]), Decimal(million) * 1000000, name)
                self.assertTrue(field["meta"]["line_fallback"], name)
                self.assertIsNotNone(field["grounding"], name)
            self.assertEqual(fields["is.revenue"]["raw_text"], "500,000")

            assessed = assess([extracted], {extracted["document_id"]: parsed}, config,
                              cross_document=False)[0]
            for name in expected:
                field = assessed["fields"][f"is.{name}"]
                self.assertEqual(field["decision"], "review", name)
                self.assertIn("LINE_FALLBACK", field["review_reasons"], name)
            operating = assessed["fields"]["is.operating_profit"]
            status = {item["id"]: item["status"] for item in operating["validators"]}
            self.assertEqual(status["V-IS-01"], "pass")
            self.assertEqual(status["V-IS-02"], "pass")
            net_status = {item["id"]: item["status"]
                          for item in assessed["fields"]["is.net_profit"]["validators"]}
            self.assertEqual(net_status["V-IS-03"], "pass")

    def test_debit_credit_statement_reads_direction_from_the_filled_column(self):
        """借贷双列流水：方向由「哪一列有金额」读出来，支出金额补齐负号。

        借记（支出）列印的是正数，取出来后要按方向取负；方向不是印出来的字，
        所以 meta 里记住是从哪一列推出来的，金额本身走 SIGN_INFERRED 复核。
        """
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            company = synth.make_company(random.Random(7), 0)
            data = TableRobustnessTests._fixture_data()
            doc = fitz.open()
            gt = synth.build_debit_credit_fixture(doc, company, data)
            path = root / "debit-credit.pdf"
            doc.save(path, garbage=4, deflate=True)
            doc.close()
            config = copy.deepcopy(load_config())
            config["paths"]["work"] = str(root / "work")
            parsed = parse_document(path, "DC", force=True, config=config,
                                    out_dir=root / "parsed")
            tables = [t for page in parsed["pages"] for t in page["tables"]]
            self.assertTrue(tables)
            self.assertIn("借方发生额", tables[0]["headers"])
            self.assertIn("贷方发生额", tables[0]["headers"])
            extracted = extract_document(parsed, config)
            fields = extracted["fields"]
            self.assertEqual(TableRobustnessTests._txn_count(extracted),
                             len(data["transactions"]))
            for index, txn in enumerate(data["transactions"]):
                prefix = f"bs.transactions[{index}]"
                expected_column = ("借方发生额" if txn["direction"] == "expense"
                                   else "贷方发生额")
                direction = fields[prefix + ".direction"]
                self.assertEqual(direction["value"], txn["direction"])
                self.assertEqual(direction["meta"]["source_column"], expected_column)
                amount = fields[prefix + ".amount"]
                self.assertEqual(Decimal(amount["value"]), txn["amount"])
                # 只有支出列需要补负号；收入列原样是正数，符号来自文字本身。
                expected_sign = "direction" if txn["direction"] == "expense" else "text"
                self.assertEqual(amount["meta"]["sign_source"], expected_sign)
                self.assertEqual(amount["meta"]["source_column"], expected_column)
                self.assertEqual(gt[prefix + ".amount"]["value"], amount["value"])
            assessed = assess([extracted], {extracted["document_id"]: parsed}, config,
                              cross_document=False)[0]
            for index, txn in enumerate(data["transactions"]):
                field = assessed["fields"][f"bs.transactions[{index}].amount"]
                if txn["direction"] == "expense":
                    self.assertIn("SIGN_INFERRED", field["review_reasons"])


class ExtractorCrosscheckTests(unittest.TestCase):
    """两条互不相干的定位通路读同一个数：对不上就交给人，程序既不猜也不改值。

    表格引擎用二维框格决定片段属于哪一列；line_cell 只按「行名之后第 r 个数字片段」分列。
    SCENARIO.md 把 EXTRACTOR_CONFLICT 列为硬性原因，这条通路是它目前的触发来源。
    """

    HEADERS = ["项目", "本期金额", "上期金额"]

    @staticmethod
    def _part(text, x0, x1, y=180):
        return {"text": text, "bbox": [x0, y, x1, y + 12], "confidence": 0.99,
                "chars": [], "engine": "pymupdf_text"}

    @classmethod
    def _parsed(cls, cells, line_parts, extra_lines=()):
        lines = [cls._line(line_parts)]
        lines += [cls._line(parts) for parts in extra_lines]
        page = {"page": 1, "width": 595, "height": 842, "engine": "pymupdf_text",
                "preprocess": {}, "quality_flags": [], "codes": [], "recrop_parts": [],
                "lines": lines,
                "tables": [{"table_id": "p1-income", "name": "income", "page": 1,
                            "bbox": [35, 150, 560, 400], "headers": cls.HEADERS,
                            "columns": [{"header": header, "center": center, "radius": 80}
                                        for header, center in zip(cls.HEADERS,
                                                                  (60, 300, 470))],
                            "rows": [{"row_index": 0, "page": 1,
                                      "cells": {"项目": cls._part("营业收入", 40, 80),
                                                **cells}}],
                            "engine": "pdfplumber"}]}
        return {"document_id": "XCHK", "doc_type": "INCOME_STATEMENT", "page_count": 1,
                "engine": "pymupdf_text", "source": "xchk.pdf", "sha256": "0" * 64,
                "quality": {"parse_score": 0.99}, "pages": [page]}

    @classmethod
    def _line(cls, parts):
        return {"text": " ".join(part["text"] for part in parts),
                "bbox": [40, parts[0]["bbox"][1], 560, parts[0]["bbox"][3]],
                "confidence": 0.99, "parts": parts, "engine": "pymupdf_text",
                "recovered": False}

    def test_agreeing_witness_is_recorded_and_needs_no_review(self):
        parsed = self._parsed(
            {"本期金额": self._part("1000.00", 260, 340),
             "上期金额": self._part("900.00", 430, 510)},
            [self._part("营业收入", 40, 80), self._part("1000.00", 260, 340),
             self._part("900.00", 430, 510)])
        field = extract_document(parsed)["fields"]["is.revenue"]
        self.assertEqual(field["meta"]["extractor_alternates"], {"line_cell": "agreed"})
        self.assertEqual(field["signals"]["extractor_agreement"], 1.0)
        self.assertNotIn("extractor_conflict", field["meta"])

    def test_misassigned_column_conflicts_and_forces_review(self):
        """表格框把「823.69 228,」当成一个值取了前者；按列序这一格读到的是「228,」。

        180dpi 退化件上真实出现过这个形状，TABLE_ANOMALY 没报，两条通路对上了号。
        """
        parsed = self._parsed({"本期金额": self._part("823.69 228,", 260, 340)},
                              [self._part("营业收入", 40, 80), self._part("228,", 250, 270),
                               self._part("823.69", 300, 340)])
        config = copy.deepcopy(load_config())
        extracted = extract_document(parsed, config)
        field = extracted["fields"]["is.revenue"]
        conflict = field["meta"]["extractor_conflict"]
        self.assertEqual(conflict["path"], "table_row vs line_cell")
        self.assertEqual(conflict["alt_value"], "228.00")
        # 证据要带原文：单独的 228.00 像另一个数，配上「228,」才看得出数字被截断了一组。
        self.assertEqual(conflict["alt_raw"], "228,")
        self.assertEqual(field["signals"]["extractor_agreement"], 0.0)
        self.assertEqual(field["value"], "823.69")     # 值原样保留，不自己「修好」
        decided = decide_field(copy.deepcopy(field), extracted, parsed, config,
                               [], Calibrator(config))
        self.assertIn("EXTRACTOR_CONFLICT", decided["review_reasons"])
        self.assertEqual(decided["decision"], "review")

    def test_too_many_numbers_withhold_the_witness(self):
        """混进附注号之类的多余数字时列序不可信，宁可不作证也不制造假冲突。"""
        parsed = self._parsed({"本期金额": self._part("1000.00", 260, 340)},
                              [self._part("营业收入", 40, 80), self._part("35", 200, 215),
                               self._part("1000.00", 260, 340),
                               self._part("900.00", 430, 510)])
        field = extract_document(parsed)["fields"]["is.revenue"]
        self.assertNotIn("extractor_alternates", field["meta"])

    def test_repeated_label_with_different_numbers_withholds_the_witness(self):
        parsed = self._parsed({"本期金额": self._part("1000.00", 260, 340)},
                              [self._part("营业收入", 40, 80),
                               self._part("1000.00", 260, 340)],
                              extra_lines=[[self._part("营业收入", 40, 80),
                                            self._part("700.00", 260, 340)]])
        field = extract_document(parsed)["fields"]["is.revenue"]
        self.assertNotIn("extractor_alternates", field["meta"])

    def test_switch_off_leaves_no_evidence_and_reason_is_configurable(self):
        parsed = self._parsed({"本期金额": self._part("823.69 228,", 260, 340)},
                              [self._part("营业收入", 40, 80), self._part("228,", 250, 270),
                               self._part("823.69", 300, 340)])
        off = copy.deepcopy(load_config())
        off["validation"]["extractor_crosscheck"]["enabled"] = False
        field = extract_document(parsed, off)["fields"]["is.revenue"]
        self.assertNotIn("extractor_alternates", field["meta"])

        relaxed = copy.deepcopy(load_config())
        relaxed["validation"]["review_on_extractor_conflict"] = False
        extracted = extract_document(parsed, relaxed)
        field = extracted["fields"]["is.revenue"]
        self.assertIn("extractor_conflict", field["meta"])
        decided = decide_field(copy.deepcopy(field), extracted, parsed, relaxed,
                               [], Calibrator(relaxed))
        self.assertNotIn("EXTRACTOR_CONFLICT", decided["review_reasons"])

    def test_line_cell_picks_the_prior_column_by_ordinal(self):
        config = load_config()
        parsed = self._parsed(
            {"本期金额": self._part("1000.00", 260, 340),
             "上期金额": self._part("900.00", 430, 510)},
            [self._part("营业收入", 40, 80), self._part("1000.00", 260, 340),
             self._part("900.00", 430, 510)])
        spec = field_spec("INCOME_STATEMENT", "is.prior_revenue", config)
        witness = _extract_line_cell(parsed, spec, config, {"unit": None, "unit_factor": None})
        self.assertEqual(witness["value"], "900.00")
        self.assertEqual(witness["meta"]["line_cell"], True)
        self.assertEqual(witness["grounding"]["page"], 1)


class RawTextAuditTests(unittest.TestCase):
    """编造率回查：字段声称读到的那串字，必须真在页面上逐字出现过。

    原文都是抽取器从解析层抄来的，所以这条在标注集上零命中；它守的是以后有人把
    原文拼出来。判定只记证据、不改值，硬性复核要显式开开关。
    """

    HEADERS = ["项目", "本期金额", "上期金额"]

    @staticmethod
    def _part(text, x0, x1, y=180):
        return {"text": text, "bbox": [x0, y, x1, y + 12], "confidence": 0.99,
                "chars": [], "engine": "pymupdf_text"}

    @classmethod
    def _page(cls, number, revenue):
        parts = [cls._part("营业收入", 40, 80), cls._part(revenue, 260, 340)]
        line = {"text": " ".join(part["text"] for part in parts),
                "bbox": [40, 180, 340, 192], "confidence": 0.99, "parts": parts,
                "engine": "pymupdf_text", "recovered": False}
        return {"page": number, "width": 595, "height": 842, "engine": "pymupdf_text",
                "preprocess": {}, "quality_flags": [], "codes": [], "recrop_parts": [],
                "lines": [line],
                "tables": [{"table_id": f"p{number}-income", "name": "income",
                            "page": number, "bbox": [35, 150, 560, 400],
                            "headers": cls.HEADERS,
                            "columns": [{"header": header, "center": center, "radius": 80}
                                        for header, center in zip(cls.HEADERS,
                                                                  (60, 300, 470))],
                            "rows": [{"row_index": 0, "page": number,
                                      "cells": {"项目": cls._part("营业收入", 40, 80),
                                                "本期金额": cls._part(revenue, 260, 340)}}],
                            "engine": "pdfplumber"}]}

    @classmethod
    def _parsed(cls, pages):
        return {"document_id": "RAW1", "doc_type": "INCOME_STATEMENT",
                "page_count": len(pages), "engine": "pymupdf_text",
                "source": "raw.pdf", "sha256": "0" * 64,
                "quality": {"parse_score": 0.99}, "pages": pages}

    def test_copied_raw_is_certified_without_touching_the_value(self):
        parsed = self._parsed([self._page(1, "1,000.00")])
        field = extract_document(parsed)["fields"]["is.revenue"]
        self.assertEqual(field["meta"]["raw_grounded"], "page")
        self.assertEqual(field["signals"]["raw_grounded"], 1.0)
        self.assertNotIn("raw_unverified", field["meta"])

    def test_spacing_and_full_width_differences_are_not_invention(self):
        """片段切法与全角半角造成的差异不算编造：两边都做同样的规范化再比。"""
        parsed = self._parsed([self._page(1, "1，000．00")])
        field = extract_document(parsed)["fields"]["is.revenue"]
        self.assertEqual(field["meta"]["raw_grounded"], "page")

    def test_invented_raw_is_flagged_and_value_is_left_alone(self):
        config = load_config()
        parsed = self._parsed([self._page(1, "1,000.00")])
        extracted = extract_document(parsed, config)
        field = extracted["fields"]["is.revenue"]
        field["raw_text"] = "9,999,999.00"
        field["meta"].pop("raw_grounded")
        _raw_text_audit(extracted["fields"], parsed, config)
        self.assertEqual(field["meta"]["raw_unverified"]["raw_text"], "9,999,999.00")
        self.assertEqual(field["meta"]["raw_unverified"]["page"], 1)
        self.assertEqual(field["signals"]["raw_grounded"], 0.0)
        self.assertEqual(field["value"], "1000.00")

    def test_raw_found_on_another_page_counts_as_grounded(self):
        """跨页表格续列会读到邻页的字符：算有据，但归因要和命中本页分开。"""
        config = load_config()
        parsed = self._parsed([self._page(1, "1,000.00"), self._page(2, "2,000.00")])
        field = extract_document(parsed, config)["fields"]["is.revenue"]
        field["raw_text"] = "2,000.00"
        field["meta"].pop("raw_grounded")
        _raw_text_audit({"is.revenue": field}, parsed, config)
        self.assertEqual(field["meta"]["raw_grounded"], "other_page")
        self.assertNotIn("raw_unverified", field["meta"])

    def test_empty_raw_is_left_to_the_missing_field_checks(self):
        config = load_config()
        parsed = self._parsed([self._page(1, "1,000.00")])
        extracted = extract_document(parsed, config)
        field = extracted["fields"]["is.revenue"]
        field["raw_text"] = ""
        field["meta"].pop("raw_grounded")
        _raw_text_audit(extracted["fields"], parsed, config)
        self.assertNotIn("raw_unverified", field["meta"])
        self.assertNotIn("raw_grounded", field["meta"])

    def test_audit_off_leaves_no_evidence_and_reason_is_configurable(self):
        config = load_config()
        parsed = self._parsed([self._page(1, "1,000.00")])
        extracted = extract_document(parsed, config)
        field = extracted["fields"]["is.revenue"]
        field["raw_text"] = "9,999,999.00"
        field["meta"].pop("raw_grounded")

        off = copy.deepcopy(config)
        off["validation"]["raw_text_audit"]["enabled"] = False
        _raw_text_audit({"is.revenue": field}, parsed, off)
        self.assertNotIn("raw_unverified", field["meta"])

        _raw_text_audit({"is.revenue": field}, parsed, config)
        # 出厂配置就带着这道门：编造信号不需要任何手工加码就该落到复核。以前这条靠
        # 验收手改 config 才成立，等于替代码补了一条它自己没有的保证。
        decided = decide_field(copy.deepcopy(field), extracted, parsed, config,
                               [], Calibrator(config))
        self.assertEqual(decided["decision"], "review")
        self.assertIn("RAW_UNVERIFIED", decided["review_reasons"])

        off = copy.deepcopy(config)
        off["validation"]["review_on_raw_unverified"] = False
        decided = decide_field(copy.deepcopy(field), extracted, parsed, off,
                               [], Calibrator(off))
        self.assertNotIn("RAW_UNVERIFIED", decided["review_reasons"])


class ReviewActionContractTests(unittest.TestCase):
    """拒绝契约：每个硬性原因码都要带一句人话，并且这句话跟着结果一起出去。

    码本身只够机器分流，复核员看到 EXTRACTOR_CONFLICT 并不知道下一步做什么。文案放
    配置里，三个出口（复核台 / out/fields.csv / final/final_fields.csv）共用同一份。
    """

    def test_every_hard_reason_has_a_nonempty_action(self):
        config = load_config()
        table = review_action_table(config)
        missing = [reason for reason in config["hard_abstain_reasons"]
                   if not str(table.get(reason) or "").strip()]
        self.assertEqual(missing, [])
        self.assertGreaterEqual(len(table), len(config["hard_abstain_reasons"]))

    def test_actions_follow_reason_order_and_skip_unmapped_codes(self):
        table = {"LOW_OCR": "逐字比对", "GROUNDING_MISSING": "先找坐标"}
        actions = field_actions(
            {"review_reasons": ["GROUNDING_MISSING", "LOW_OCR", "NOPE", "LOW_OCR"]}, table)
        self.assertEqual(actions, ["先找坐标", "逐字比对"])

    def test_summary_counts_observed_reasons_and_reports_unmapped(self):
        config = load_config()
        table = review_action_table(config)
        document = {"fields": {
            "a": {"review_reasons": ["LOW_OCR"]},
            "b": {"review_reasons": ["LOW_OCR", "GROUNDING_MISSING"]},
            "c": {"review_reasons": []},
        }}
        payload = review_actions([document], config)
        self.assertEqual(payload["by_reason"]["LOW_OCR"]["field_count"], 2)
        self.assertEqual(payload["by_reason"]["LOW_OCR"]["action"], table["LOW_OCR"])
        self.assertEqual(payload["by_reason"]["GROUNDING_MISSING"]["field_count"], 1)
        self.assertEqual(payload["unmapped"], [])

        config["validation"]["review_actions"].pop("LOW_OCR")
        payload = review_actions([document], config)
        self.assertEqual(payload["unmapped"], ["LOW_OCR"])
        self.assertEqual(payload["by_reason"]["LOW_OCR"]["action"], "")

    def test_run_fields_csv_carries_the_action_text(self):
        from docpipe.run import _write_csv

        config = load_config()
        table = review_action_table(config)
        payload = [{
            "document_id": "R1", "doc_type": "BANK_STATEMENT",
            "fields": {"bs.name": {
                "value": "张三", "value_type": "string", "confidence": 0.5,
                "decision": "review", "grounding": {"page": 1, "bbox": [1, 2, 3, 4]},
                "extractor_id": "label_value:户名",
                "review_reasons": ["LOW_OCR", "GROUNDING_MISSING"],
            }},
        }]
        with tempfile.TemporaryDirectory() as temp:
            _write_csv(Path(temp), payload, config)
            with (Path(temp) / "fields.csv").open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0])[-3:], ["review_actions", "value_rewrite", "chain_evidence"])
        self.assertEqual(rows[0]["review_actions"],
                         table["LOW_OCR"] + "|" + table["GROUNDING_MISSING"])

    def test_export_csv_and_summary_carry_the_action_text(self):
        config = load_config()
        table = review_action_table(config)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            (out / "T1").mkdir(parents=True)
            (out / "review").mkdir()
            (out / "summary.json").write_text(json.dumps({
                "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
                "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                               "path": "T1/T1.json"}],
            }), encoding="utf-8")
            (out / "T1" / "T1.json").write_text(json.dumps({
                "document_id": "T1", "doc_type": "BANK_STATEMENT",
                "fields": {"bs.account_name": {
                    "path": "bs.account_name", "value": "旧值", "value_type": "string",
                    "confidence": 0.4, "decision": "review",
                    "grounding": {"page": 1, "bbox": [1, 2, 3, 4]},
                    "extractor_id": "label_value:户名",
                    "review_reasons": ["LOW_OCR", "GROUNDING_MISSING"],
                }},
            }, ensure_ascii=False), encoding="utf-8")

            result = export(out, out / "final", config)
            with (out / "final" / "final_fields.csv").open(
                    "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["review_actions"],
                             table["LOW_OCR"] + "|" + table["GROUNDING_MISSING"])
            self.assertEqual(result["review_actions"]["by_reason"]["LOW_OCR"]["field_count"], 1)
            self.assertEqual(result["review_actions"]["unmapped"], [])


class ReviewLoopTests(unittest.TestCase):
    """复核台账要能变回校准样本：同一条特征向量上的人裁判决法，正是校准要学的对照。

    单测不重训生产模型（要保持快），验判决方向、去重规则、"标注优先"与"批次太小整批
    不接"，以及开关关着时一个目录都不扫——出厂默认必须是关的。
    """

    PATHS = {"bl.registered_capital": "monetary", "bl.legal_representative": "party_name"}

    @staticmethod
    def _field(path):
        return {
            "path": path, "risk": ReviewLoopTests.PATHS[path], "value_type": "decimal",
            "value": "100.00", "decision": "review", "review_reasons": ["LOW_OCR"],
            "grounding": {"page": 1, "bbox": [50.0, 100.0, 200.0, 118.0]},
            "signals": {"ocr_conf": 0.61, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 0.8, "cross_field_score": 0.5,
                        "grounding_present": 1.0},
        }

    @classmethod
    def _fixture(cls, root, records, doc_id="D900"):
        """结果目录的样子：out/<doc_id>/<doc_id>.json 与 out/review/<doc_id>.review.jsonl。"""
        out = root / "out"
        (out / doc_id).mkdir(parents=True, exist_ok=True)
        (out / doc_id / f"{doc_id}.json").write_text(json.dumps({
            "document_id": doc_id, "doc_type": "business_license",
            "fields": {path: cls._field(path) for path in cls.PATHS},
        }, ensure_ascii=False), encoding="utf-8")
        review = out / "review"
        review.mkdir(parents=True, exist_ok=True)
        lines = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                 for item in records]
        (review / f"{doc_id}.review.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8")
        return out

    @staticmethod
    def _config(**overrides):
        config = copy.deepcopy(load_config())
        config["calibration"].update(overrides)
        return config

    @staticmethod
    def _record(path, action, timestamp, value=None):
        record = {"document_id": "D900", "field_path": path, "action": action,
                  "auto_decision": "review", "timestamp": timestamp}
        if value is not None:
            record["corrected_value"] = value
        return record

    def test_gate_closed_scans_nothing_and_explicit_dir_obeys_the_operator(self):
        config = self._config(learn_from_review=False, review_dirs=["out/user"])
        self.assertEqual(calibrate.review_sample_dirs(config), [])
        self.assertEqual(calibrate.review_sample_dirs(
            self._config(learn_from_review=True))[-1].name, "user")
        with tempfile.TemporaryDirectory() as tmp:
            forced = calibrate.review_sample_dirs(config, [tmp])
        self.assertEqual([str(item) for item in forced], [tmp])

    def test_accept_is_positive_and_junk_records_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                self._record("bl.registered_capital", "accept", "2026-10-26T09:00:00"),
                "{这不是 JSON",
                self._record("bl.nope", "accept", "2026-10-26T09:00:01"),
                self._record("bl.legal_representative", "pending", "2026-10-26T09:00:02"),
            ])
            samples = calibrate.review_samples([root], self._config())
        self.assertEqual([item["path"] for item in samples], ["bl.registered_capital"])
        sample = samples[0]
        self.assertEqual(sample["y"], 1)
        self.assertEqual(sample["source"], "review")
        self.assertEqual(sample["auto_decision"], "review")
        self.assertEqual(sample["features"],
                         calibrate.field_features(self._field("bl.registered_capital"),
                                                  "business_license", self._config()))

    def test_corrected_value_is_a_negative_on_the_same_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                self._record("bl.legal_representative", "correct",
                             "2026-10-26T09:00:00", value="李四"),
            ])
            samples = calibrate.review_samples([root], self._config())
            config = self._config()
        self.assertEqual([(item["y"], item["correct"], item["review_action"])
                          for item in samples], [(0, False, "correct")])
        self.assertEqual(samples[0]["features"],
                         calibrate.field_features(self._field("bl.legal_representative"),
                                                  "business_license", config))

    def test_last_word_wins_inside_a_file_and_across_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                self._record("bl.registered_capital", "correct",
                             "2026-10-26T09:00:05", value="120.00"),
                self._record("bl.registered_capital", "accept", "2026-10-26T09:00:09"),
            ])
            other = Path(tmp) / "second"
            shutil.copytree(root / "D900", other / "out" / "D900")
            (other / "out" / "review").mkdir(parents=True)
            for stamp, expect in (("2026-10-26T08:00:00", "accept"),
                                  ("2026-10-26T10:00:00", "correct")):
                (other / "out" / "review" / "D900.review.jsonl").write_text(
                    json.dumps(self._record("bl.registered_capital", "correct", stamp),
                               ensure_ascii=False) + "\n", encoding="utf-8")
                samples = calibrate.review_samples([root, other / "out"], self._config())
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0]["review_action"], expect)

    def test_labels_shadow_records_and_a_small_batch_is_refused(self):
        records = [self._record("bl.registered_capital", "accept", "2026-10-26T09:00:00"),
                   self._record("bl.legal_representative", "correct",
                                "2026-10-26T09:00:01", value="李四")]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), records)
            samples = calibrate.review_samples([root], self._config())
            config = self._config(review_min_samples=20)
            labels = [{"document_id": "D900", "path": "bl.registered_capital"}]
            merged, stats = calibrate.merge_review_samples(labels, samples, config)
            self.assertEqual((stats["records"], stats["shadowed_by_labels"], stats["usable"]),
                             (2, 1, 1))
            self.assertFalse(stats["enabled"])
            self.assertEqual(merged, labels)
            enough = self._config(review_min_samples=1)
            merged, stats = calibrate.merge_review_samples([], samples, enough)
        self.assertTrue(stats["enabled"])
        self.assertEqual(len(merged), 2)
        self.assertEqual((stats["accepted"], stats["corrected"]), (1, 1))

    def test_train_folds_the_ledger_only_when_asked_and_records_the_source(self):
        config = self._config(learn_from_review=False, review_min_samples=1,
                              min_total_samples=2, min_samples_per_field=1)
        labels = []
        for index, path in enumerate(sorted(self.PATHS)):
            field = self._field(path)
            labels.append({
                "document_id": f"L{index}", "doc_type": "business_license", "path": path,
                "group_path": path, "risk": field["risk"], "y": index,
                "correct": bool(index), "grounded": True, "source": "label",
                "features": calibrate.field_features(field, "business_license", config),
                "heuristic": calibrate.heuristic_confidence(field, config),
            })
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                self._record("bl.registered_capital", "accept", "2026-10-26T09:00:00"),
                self._record("bl.legal_representative", "correct",
                             "2026-10-26T09:00:01", value="李四"),
            ])
            with mock.patch.object(calibrate, "build_samples", return_value=labels):
                shipped = calibrate.train("dev", config)
                folded = calibrate.train("dev", config, review_override=[str(root)])
        self.assertEqual(shipped["sample_sources"], {"label": 2, "review": 0, "synthetic": 0})
        self.assertFalse(shipped["review_loop"]["enabled"])
        self.assertEqual(shipped["n_samples"], 2)
        self.assertEqual(folded["sample_sources"], {"label": 2, "review": 2, "synthetic": 0})
        self.assertEqual(folded["negative_sources"]["label"], 1)
        self.assertEqual(folded["negative_sources"]["review"], 1)
        self.assertTrue(folded["review_loop"]["enabled"])
        self.assertEqual([str(item) for item in folded["review_loop"]["dirs"]], [str(root)])
        self.assertEqual(folded["reliability"]["review_only"]["samples"], 2)
        self.assertEqual(folded["reliability"]["label_only"]["samples"], 2)

    def test_a_json_line_that_is_not_an_object_does_not_break_the_ledger(self):
        """台账可被人工/工具改过：一行合法 JSON 但是裸串/数组/null，也必须当坏行跳过。

        这类行以前会让复核台整页 500（record.get 打在 str 上）。验收夹具就是这么发现的。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                json.dumps("{这不是 JSON", ensure_ascii=False),
                json.dumps(["bl.registered_capital"], ensure_ascii=False),
                "null",
                self._record("bl.registered_capital", "accept", "2026-10-26T09:00:00"),
            ])
            records = read_review_records(root, "D900")
            samples = calibrate.review_samples([root], self._config())
        self.assertEqual(sorted(records), ["bl.registered_capital"])
        self.assertEqual(records["bl.registered_capital"]["action"], "accept")
        self.assertEqual([item["y"] for item in samples], [1])

    def test_all_positive_pool_refuses_to_fit_and_points_at_the_ledger(self):
        """标注集被抽到全对时，重训必须报错说「缺负例、去折复核台账」，不许静默出个 1.0。

        这里刻意不走「把阈值调低让指标好看」那条路：全是正例的池子拟合出来的概率
        没有对照，读不得。复核台账里的 correct 就是同特征向量上的真实负例。
        """
        config = self._config(review_min_samples=1, min_total_samples=2,
                              min_samples_per_field=1)
        labels = self._all_positive_labels(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(Path(tmp), [
                self._record("bl.registered_capital", "correct",
                             "2026-10-26T09:00:00", value="120.00"),
            ])
            with mock.patch.object(calibrate, "build_samples", return_value=labels):
                with self.assertRaises(SystemExit) as caught:
                    calibrate.train("dev", config)
                artifact = calibrate.train("dev", config, review_override=[str(root)])
        message = str(caught.exception)
        self.assertIn("0 条负例", message)
        self.assertIn("--review", message)
        self.assertNotIn("ValueError", message)
        self.assertEqual(artifact["sample_sources"], {"label": 2, "review": 1, "synthetic": 0})
        self.assertEqual(artifact["negative_sources"]["review"], 1)

    def _all_positive_labels(self, config):
        labels = []
        for path in sorted(self.PATHS):
            field = self._field(path)
            labels.append({
                "document_id": "L1", "doc_type": "business_license", "path": path,
                "group_path": path, "risk": field["risk"], "y": 1,
                "correct": True, "grounded": True, "source": "label",
                "features": calibrate.field_features(field, "business_license", config),
                "heuristic": calibrate.heuristic_confidence(field, config),
            })
        return labels

    @staticmethod
    def _probe_result():
        decision = {"threshold": 0.9, "auto_accept_rate": 0.998,
                    "accepted_error_rate": 0.0, "mean_confidence": 0.97}
        return {
            "real_samples": 2, "synthetic_samples": 1, "synthetic_by_kind": {"wrong_value": 1},
            "synthetic_with_identical_features": 1, "production_source": "shipped_artifact",
            "verdict": "still_ordinal", "reason": "特征看不见值对不对",
            "production": {"decision": decision, "reliability": {}},
            "production_on_synthetic": {"decision": decision, "reliability": {}},
            "augmented": {"decision_on_real": decision, "decision_on_synthetic": decision,
                          "reliability_on_real": {}, "reliability_on_synthetic": {}},
        }

    def test_cli_probe_survives_a_refused_fit_but_does_not_claim_a_new_artifact(self):
        """`--probe` 是诊断，不能被重训那一步连坐：曲线要照出，产物不写，退出码 1。

        以前样本池全正例时 `calibrate --split dev --probe` 直接被拒训的 SystemExit 带走，
        文档里这条常用命令跑不动；改成非零退出是为了别让人以为 work/calibration.json 已重训。
        """
        config = self._config(review_min_samples=1, min_total_samples=2,
                              min_samples_per_field=1)
        labels = self._all_positive_labels(config)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            config["paths"]["work"] = str(work)
            out = Path(tmp) / "calibration.json"
            stdout = io.StringIO()
            with mock.patch.object(calibrate, "build_samples", return_value=labels), \
                    mock.patch.object(calibrate, "load_config", return_value=config), \
                    mock.patch.object(calibrate, "probe", return_value=self._probe_result()) \
                    as probe_run, \
                    mock.patch.object(calibrate, "_write_reliability_svg") as svg, \
                    contextlib.redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as caught:
                    calibrate.main(["--split", "dev", "--probe", "--out", str(out)])
            self.assertEqual(caught.exception.code, 1)
            self.assertFalse(out.exists())
            probe_run.assert_called_once()
            self.assertEqual(svg.call_count, 3)
            self.assertTrue((work / "calibration_probe.json").exists())
            self.assertIn("校准未写入", stdout.getvalue())
            self.assertIn("0 条负例", stdout.getvalue())
            self.assertIn("shipped_artifact", stdout.getvalue())
            stdout2 = io.StringIO()
            with mock.patch.object(calibrate, "build_samples", return_value=labels), \
                    mock.patch.object(calibrate, "load_config", return_value=config), \
                    mock.patch.object(calibrate, "probe") as probe_run, \
                    contextlib.redirect_stdout(stdout2):
                with self.assertRaises(SystemExit) as plain:
                    calibrate.main(["--split", "dev", "--out", str(out)])
        self.assertIn("0 条负例", str(plain.exception))
        self.assertEqual(stdout2.getvalue(), "")
        probe_run.assert_not_called()
        self.assertFalse(out.exists())

class TableStructureScoreTests(unittest.TestCase):
    """G4/P1-6：检出网格 vs 标注网格，逻辑坐标对不上就是取错格。

    这套夹具用真实的 schema 表头（docpipe.config.table_spec），列下标跟着配置走；
    故意只标注三列，用来盯住「没标注的列不许算成多检」这条口径。
    """

    BANDS = {0: (35.0, 100.0), 1: (100.0, 160.0), 5: (355.0, 420.0)}
    COLUMNS = (("txn_date", 0), ("summary", 1), ("amount", 5))

    def setUp(self):
        self.config = load_config()
        self.headers = list(table_spec("BANK_STATEMENT", "transactions", self.config)["headers"])

    def _band(self, col, row, page=1, tight=False, shift=0.0):
        x0, x1 = self.BANDS[col]
        top = 232.0 + 22.0 * row
        bottom = top + 22.0
        if tight:
            return [x0 + 4.0 + shift, top + 8.0, x1 - 22.0 + shift, bottom - 5.0]
        return [x0 + shift, top, x1 + shift, bottom]

    def _labels(self, rows, page=1):
        fields = {}
        for index, values in enumerate(rows):
            for column, col in self.COLUMNS:
                fields[f"bs.transactions[{index}].{column}"] = {
                    "value": values[column], "page": page,
                    "bbox": self._band(col, index, page)}
        return {"document_id": "TS1", "doc_type": "BANK_STATEMENT", "fields": fields}

    def _table(self, cells_by_row, page=1, headers=None, engine="pdfplumber"):
        table_id = f"p{page}-transactions"
        rows = []
        for index, cells in enumerate(cells_by_row):
            rows.append({"row_index": index, "page": page,
                         "cells": {header: {"text": "x", "bbox": bbox, "confidence": 0.99}
                                   for header, bbox in cells.items()}})
        return {"table_id": table_id, "name": "transactions", "page": page,
                "bbox": [35.0, 200.0, 560.0, 400.0],
                "headers": headers or self.headers,
                "columns": [{"header": header, "center": 0.0, "radius": 80.0}
                            for header in (headers or self.headers)],
                "rows": rows, "engine": engine}

    def _parsed(self, pages):
        return {"document_id": "TS1", "doc_type": "BANK_STATEMENT", "page_count": len(pages),
                "engine": "pymupdf_text", "source": "ts.pdf", "sha256": "0" * 64,
                "quality": {"parse_score": 0.99},
                "pages": [{"page": page, "width": 595.0, "height": 842.0,
                           "engine": "pymupdf_text", "quality_flags": [], "lines": [],
                           "tables": tables} for page, tables in pages]}

    def _band_cells(self, row, page=1, tight=False, shift=0.0):
        return {self.headers[col]: self._band(col, row, page, tight, shift)
                for _, col in self.COLUMNS}

    def _score(self, labels, parsed, **overrides):
        config = copy.deepcopy(self.config)
        config.setdefault("evaluation", {}).setdefault("table_structure", {}).update(overrides)
        rec = tablescore.score_document("TS1", labels["doc_type"], labels, parsed, config,
                                        tablescore._params(config))["tables"].get("transactions")
        if rec:
            self.assertEqual(rec["aligned"] + rec["misassigned"] + rec["missed_cell"],
                             rec["gt_cells"], "每个标注格必须落进且只落进一个桶")
        return rec or {}

    def test_clean_grid_scores_full_structure(self):
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "服务费", "amount": "1.00"},
                              {"txn_date": "2025-07-02", "summary": "货款", "amount": "2.00"}])
        parsed = self._parsed([(1, [self._table([self._band_cells(0), self._band_cells(1)])])])
        rec = self._score(labels, parsed)
        self.assertEqual((rec["gt_cells"], rec["det_cells"]), (6, 6))
        self.assertEqual(rec["structure_exact_rate"], 1.0)
        self.assertEqual(rec["cell_found_rate"], 1.0)
        self.assertEqual(rec["misassigned"], 0)
        self.assertEqual(rec["extra_det"], 0)
        self.assertEqual(rec["mean_iou"], 1.0)

    def test_tighter_box_from_segment_engine_still_counts_as_aligned(self):
        """segment_table 交的是贴着字的框，IoU 只有 0.3 上下。

        按 docTR 原样用 IoU 配对这些格子会算漏检；本项目按嵌套度配对，结构分不受框语义影响。
        这条测试钉住那个改写——有人改回 IoU 就会红。
        """
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "服务费", "amount": "1.00"}])
        parsed = self._parsed([(1, [self._table([self._band_cells(0, tight=True)],
                                                engine="segment_table")])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["missed_cell"], 0)
        self.assertEqual(rec["structure_exact_rate"], 1.0)
        self.assertLess(rec["mean_iou"], 0.5)
        self.assertEqual(rec["by_engine"]["segment_table"]["cells"], 3)

    def test_header_order_swapped_shows_as_column_misassignment(self):
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "服务费", "amount": "1.00"}])
        swapped = list(self.headers)
        swapped[1], swapped[5] = swapped[5], swapped[1]
        cells = {swapped[1]: self._band(1, 0), swapped[5]: self._band(5, 0),
                 self.headers[0]: self._band(0, 0)}
        parsed = self._parsed([(1, [self._table([cells], headers=swapped)])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["align_col_off"], 2)
        self.assertEqual(rec["align_row_off"], 0)
        self.assertEqual(rec["structure_exact_rate"], 0.3333)
        example = rec["examples"][0]
        self.assertNotEqual(example["gt_col"], example["det_col"])
        self.assertEqual(example["path"], "bs.transactions[0].summary")

    def test_dropped_row_shifts_the_rows_below(self):
        labels = self._labels([{"txn_date": d, "summary": s, "amount": a} for d, s, a in
                               (("2025-07-01", "a", "1.00"), ("2025-07-02", "b", "2.00"),
                                ("2025-07-03", "c", "3.00"))])
        parsed = self._parsed([(1, [self._table([self._band_cells(0), self._band_cells(2)])])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["missed_cell"], 3)
        self.assertEqual(rec["align_row_off"], 3)
        self.assertEqual(rec["row_count_delta"], -1)
        # 第 2 行的三格还在（只是行号从 2 变成 1），第 1 行的三格整排找不到，所以结构分 3/6。
        self.assertEqual(rec["structure_exact_rate"], 0.5)

    def test_phantom_row_becomes_extra_detected_cells(self):
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "a", "amount": "1.00"}])
        parsed = self._parsed([(1, [self._table([self._band_cells(0), self._band_cells(1)])])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["extra_det"], 3)
        self.assertEqual(rec["aligned"], 3)
        self.assertEqual(rec["row_count_delta"], 1)

    def test_continuation_page_stacks_row_numbers(self):
        labels = self._labels([{"txn_date": d, "summary": s, "amount": a} for d, s, a in
                               (("2025-07-01", "a", "1.00"), ("2025-07-02", "b", "2.00"),
                                ("2025-07-03", "c", "3.00"))])
        labels["fields"]["bs.transactions[2].amount"]["page"] = 2
        labels["fields"]["bs.transactions[2].summary"]["page"] = 2
        labels["fields"]["bs.transactions[2].txn_date"]["page"] = 2
        labels["fields"]["bs.transactions[2].txn_date"]["bbox"] = self._band(0, 0, page=2)
        labels["fields"]["bs.transactions[2].summary"]["bbox"] = self._band(1, 0, page=2)
        labels["fields"]["bs.transactions[2].amount"]["bbox"] = self._band(5, 0, page=2)
        parsed = self._parsed([(1, [self._table([self._band_cells(0), self._band_cells(1)],
                                                page=1)]),
                               (2, [self._table([self._band_cells(0, page=2)], page=2)])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["missed_cell"], 0)
        self.assertEqual(rec["align_row_off"], 0)
        self.assertEqual(rec["structure_exact_rate"], 1.0)
        self.assertEqual([row["row_offset"] for row in rec["grid_instances"]], [0, 2])

    def test_reordered_columns_on_continuation_page_are_blamed_on_columns(self):
        """限制 3 的机理：续页的表头顺序与列几何整套从上一页搬过来（_continued_table）。

        续页一旦换了列的物理顺序，每个字都会落进上页那套列中心里最近的那一列——错的列。
        度量不许把这件事洗成 1.0：行号照样堆对（不甩锅成漏行），本页三格里两格错位全记在列上；
        整份文档的结构分会被好页稀释成 0.7778，所以按引擎归因才看得见坏在哪一页。
        而 header_order_matches_schema 在这一页仍是 True（表头根本没重读，只是列换了地方），
        所以那个标记抓不到跨页换列序，不能当运行时门（决策 31）。
        """
        labels = self._labels([{"txn_date": d, "summary": s, "amount": a} for d, s, a in
                               (("2025-07-01", "a", "1.00"), ("2025-07-02", "b", "2.00"),
                                ("2025-07-03", "c", "3.00"))])
        for column, col in self.COLUMNS:
            field = labels["fields"][f"bs.transactions[2].{column}"]
            field["page"] = 2
            field["bbox"] = self._band(col, 0, page=2)
        swapped = {self.headers[0]: self._band(0, 0, page=2),
                   self.headers[1]: self._band(5, 0, page=2),
                   self.headers[5]: self._band(1, 0, page=2)}
        continuation = self._table([swapped], page=2, engine="continuation")
        continuation["continued_from"] = "p1-transactions"
        parsed = self._parsed([(1, [self._table([self._band_cells(0), self._band_cells(1)],
                                                page=1)]),
                               (2, [continuation])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["missed_cell"], 0)
        self.assertEqual(rec["cell_found_rate"], 1.0)
        self.assertEqual((rec["aligned"], rec["misassigned"]), (7, 2))
        self.assertEqual(rec["align_col_off"], 2)
        self.assertEqual(rec["align_row_off"], 0)
        self.assertEqual(rec["structure_exact_rate"], 0.7778)
        self.assertEqual([row["row_offset"] for row in rec["grid_instances"]], [0, 2])
        self.assertTrue([row["header_order_matches_schema"]
                         for row in rec["grid_instances"]])
        self.assertEqual([row["engine"] for row in rec["grid_instances"]],
                         ["pdfplumber", "continuation"])
        self.assertEqual([row["continued_from"] for row in rec["grid_instances"]],
                         [None, "p1-transactions"])
        self.assertEqual(rec["by_engine"]["continuation"]["cells"], 3)
        self.assertEqual(rec["by_engine"]["continuation"]["structure_exact_rate"], 0.3333)
        self.assertEqual([row["path"] for row in rec["examples"]],
                         ["bs.transactions[2].summary", "bs.transactions[2].amount"])

    def test_pairing_stays_within_one_page(self):
        """页 1 的格子和页 2 的标注坐标完全重合，也不能算命中——那是两张不同的纸。"""
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "a", "amount": "1.00"}])
        for path in ("bs.transactions[0].txn_date", "bs.transactions[0].summary",
                     "bs.transactions[0].amount"):
            labels["fields"][path]["page"] = 2
        parsed = self._parsed([(1, [self._table([self._band_cells(0)])])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["missed_cell"], 3)
        self.assertEqual(rec["aligned"], 0)

    def test_wide_merged_box_leaves_the_second_column_misassigned(self):
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "a", "amount": "1.00"}])
        merged = {self.headers[0]: self._band(0, 0),
                  self.headers[1]: [100.0, 232.0, 420.0, 254.0]}
        parsed = self._parsed([(1, [self._table([merged], engine="segment_table")])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["align_col_off"], 1)
        self.assertEqual(rec["shared_det"], 1)
        self.assertEqual(rec["examples"][0]["path"], "bs.transactions[0].amount")

    def test_unannotated_columns_are_not_counted_as_extra(self):
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "a", "amount": "1.00"}])
        cells = self._band_cells(0)
        cells[self.headers[2]] = [160.0, 232.0, 240.0, 254.0]
        cells[self.headers[6]] = [420.0, 232.0, 490.0, 254.0]
        parsed = self._parsed([(1, [self._table([cells])])])
        rec = self._score(labels, parsed)
        self.assertEqual(rec["extra_det"], 0)
        self.assertEqual(rec["det_cells"], 3)

    def test_row_label_anchored_fields_are_counted_not_scored(self):
        """利润表按行标签取数，真值里没有第几行的下标，所以不进结构分。"""
        income = table_spec("INCOME_STATEMENT", "income", self.config)["headers"]
        labels = {"document_id": "TS2", "doc_type": "INCOME_STATEMENT",
                  "fields": {"is.revenue": {"value": "1000.00", "page": 1,
                                            "bbox": [260.0, 180.0, 340.0, 192.0]}}}
        parsed = {"document_id": "TS2", "doc_type": "INCOME_STATEMENT", "page_count": 1,
                  "engine": "pymupdf_text", "source": "ts2.pdf", "sha256": "0" * 64,
                  "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 595.0, "height": 842.0, "engine": "pymupdf_text",
                             "quality_flags": [], "lines": [],
                             "tables": [{"table_id": "p1-income", "name": "income", "page": 1,
                                         "bbox": [35, 150, 560, 400], "headers": income,
                                         "columns": [], "engine": "pdfplumber",
                                         "rows": [{"row_index": 0, "page": 1,
                                                   "cells": {income[1]: {"text": "1000.00",
                                                                         "bbox": [260.0, 180.0,
                                                                                    340.0, 192.0],
                                                                         "confidence": 0.99}}}]}]}]}
        config = copy.deepcopy(self.config)
        payload = tablescore.score_document("TS2", "INCOME_STATEMENT", labels, parsed, config,
                                            tablescore._params(config))
        self.assertEqual(payload["tables"], {})
        self.assertEqual(payload["skipped"]["行标签锚定（真值没有行号）"], 1)
        self.assertEqual(payload["detected_skipped"]["该列无带行号的标注真值（不计入多检）"], 1)

    def test_cover_threshold_comes_from_config(self):
        """嵌套度门槛必须真的跟着配置走：调到 0.99 就没人配对，结构分如实变差，不许有硬编码兜底。"""
        labels = self._labels([{"txn_date": "2025-07-01", "summary": "服务费", "amount": "1.00"}])
        parsed = self._parsed([(1, [self._table([self._band_cells(0, shift=20.0)],
                                                engine="segment_table")])])
        loose = self._score(labels, parsed)
        strict = self._score(labels, parsed, cover_threshold=0.99)
        self.assertEqual(loose["missed_cell"], 0)
        self.assertEqual(loose["structure_exact_rate"], 1.0)
        self.assertEqual(strict["missed_cell"], 3)
        self.assertIsNone(strict["structure_exact_rate"])


class SmallSampleClaimTests(unittest.TestCase):
    """SCENARIO §6.5 的小样本说明做成判决：门槛全在配置里，判决按每一级自己的分母走。

    这一类存在的理由是过去两种都不诚实的写法：n≥100 时整条检查消失（看不出守没守），
    以及把 100 硬写在代码里（违反「阈值全外置」）。现在 `not_met / no_sample / provisional /
    met / not_applicable` 五个状态各有明确的触发条件，测试逐个钉住。
    """

    CONFIG = {"acceptance": {
        "min_auto_accept_rate": 0.60,
        "max_accepted_error_rate": 0.02,
        "max_monetary_accepted_error_rate": 0.01,
        "max_accepted_grounding_missing": 0,
        "min_grounding_hit_rate": 0.98,
        "min_accept_for_claim": 100,
        "claim_confidence": 0.95,
    }}

    @staticmethod
    def _record(risk="date", decision="accept", correct=True, grounded=True):
        return {"document_id": "d1", "doc_type": "X", "path": "f", "risk": risk,
                "value_type": "text", "decision": decision, "correct": correct,
                "grounded": grounded, "review_status": None, "required": True}

    def _claim(self, records, **overrides):
        config = copy.deepcopy(self.CONFIG)
        config["acceptance"].update(overrides)
        targets = claim_targets(config)
        stats = _decision_stats(records, targets)
        return stats["claim"], {row["name"]: row for row in stats["claim"]["metrics"]}

    def test_thresholds_actually_come_from_config(self):
        """改一个配置数字，判决就得跟着变；否则「阈值可配置」只是句空话。"""
        targets = claim_targets(self.CONFIG)
        self.assertEqual(targets["min_accept_for_claim"], 100)
        self.assertEqual(targets["max_accepted_error_rate"], 0.02)
        self.assertEqual(targets["claim_confidence"], 0.95)
        self.assertEqual(targets["z"], 1.96)
        self.assertEqual(claim_targets({"acceptance": {"claim_confidence": 0.99}})["z"], 2.5758)
        records = ([self._record() for _ in range(120)]
                   + [self._record(correct=False) for _ in range(4)])
        self.assertEqual(self._claim(records)[0]["status"], "not_met")
        self.assertEqual(self._claim(records, max_accepted_error_rate=0.05)[0]["status"], "met")
        clean = [self._record() for _ in range(120)]
        self.assertEqual(self._claim(clean)[0]["status"], "met")
        self.assertEqual(self._claim(clean, min_accept_for_claim=300)[0]["status"], "provisional")

    def test_sample_floor_is_a_boundary_not_a_suggestion(self):
        """恰好到达下限才算 met，差一格就是 provisional——差的那一格不许被四舍五入过去。"""
        claim, rows = self._claim([self._record() for _ in range(100)])
        self.assertEqual(claim["status"], "met")
        self.assertEqual(rows["accepted_error_rate"]["denominator"], 100)
        claim, rows = self._claim([self._record() for _ in range(99)])
        self.assertEqual(claim["status"], "provisional")
        self.assertEqual(rows["accepted_error_rate"]["denominator"], 99)
        self.assertEqual(rows["auto_accept_rate"]["denominator_name"], "required_field_count")
        self.assertEqual([row["status"] for row in claim["metrics"] if
                          row["status"] != "not_applicable"],
                         ["provisional"] * 4)
        self.assertEqual(claim["min_accept_for_claim"], 100)

    def test_money_metric_does_not_decide_a_group_without_money(self):
        """纯文本分组里没有金额字段：这一维是「不该由它表态」，不是「样本不够」。"""
        claim, rows = self._claim([self._record(risk="free_text") for _ in range(200)])
        self.assertEqual(claim["status"], "met")
        self.assertEqual(claim["not_applicable"], ["monetary_accepted_error_rate"])
        self.assertEqual((rows["monetary_accepted_error_rate"]["population"],
                          rows["monetary_accepted_error_rate"]["denominator"]), (0, 0))
        self.assertEqual(rows["monetary_accepted_error_rate"]["value"], 0.0)
        self.assertIsNone(rows["monetary_accepted_error_rate"]["ci"])
        # 有金额字段却一格都没自动接受：那一维确实不知道，只能算 no_sample。
        claim, rows = self._claim([self._record(risk="monetary", decision="review") for _ in range(10)]
                                  + [self._record() for _ in range(120)])
        self.assertEqual(claim["status"], "no_sample")
        self.assertEqual(rows["monetary_accepted_error_rate"]["status"], "no_sample")
        self.assertEqual(claim["not_applicable"], [])
        # 整级一个字段都没有：如实说没什么可说，不许落成 met。
        self.assertEqual(self._claim([])[0]["status"], "not_applicable")

    def test_a_failing_group_stays_failing_at_any_size(self):
        """小样本豁免的是「宣称达标」，不是「承认失败」：没过线就记 not_met。"""
        claim, rows = self._claim([self._record() for _ in range(40)]
                                  + [self._record(correct=False) for _ in range(4)])
        self.assertEqual(rows["accepted_error_rate"]["value"], 0.0909)
        self.assertEqual(rows["accepted_error_rate"]["status"], "not_met")
        self.assertEqual(rows["auto_accept_rate"]["status"], "provisional")
        self.assertEqual(claim["status"], "not_met")

    def test_grounding_missing_is_judged_as_a_count(self):
        """缺溯源是计数门槛（max=0）：缺一格就 not_met，一格不缺但样本小也只给 provisional。"""
        claim, rows = self._claim([self._record() for _ in range(120)] + [self._record(grounded=False)])
        self.assertEqual(rows["accepted_grounding_missing"]["value"], 1)
        self.assertEqual(rows["accepted_grounding_missing"]["status"], "not_met")
        self.assertEqual(claim["status"], "not_met")
        self.assertIsNone(rows["accepted_grounding_missing"]["ci"])
        _, rows = self._claim([self._record() for _ in range(50)])
        self.assertEqual(rows["accepted_grounding_missing"]["value"], 0)
        self.assertEqual(rows["accepted_grounding_missing"]["status"], "provisional")

    def test_interval_width_follows_configured_confidence(self):
        """置信度是真的：0.99 的区间必须比 0.95 宽，且判决里带的就是这一条区间。"""
        base, base_rows = self._claim([self._record() for _ in range(200)])
        high, high_rows = self._claim([self._record() for _ in range(200)], claim_confidence=0.99)
        self.assertEqual(base["confidence"], 0.95)
        self.assertEqual(high["confidence"], 0.99)
        self.assertEqual(base_rows["accepted_error_rate"]["ci"], [0.0, 0.0188])
        self.assertEqual(high_rows["accepted_error_rate"]["ci"][0], 0.0)
        self.assertGreater(high_rows["accepted_error_rate"]["ci"][1],
                           base_rows["accepted_error_rate"]["ci"][1])

    def test_claim_only_travels_with_targets(self):
        """不传 targets 的调用方（旧的统计口径）不能被凭空塞进判决；母体计数则恒定发出。"""
        stats = _decision_stats([self._record()])
        self.assertNotIn("claim", stats)
        self.assertEqual(stats["monetary_field_count"], 0)
        self.assertIn("claim", _decision_stats([self._record()], claim_targets(self.CONFIG)))

    def test_acceptance_tripwire_catches_a_short_group_claiming_met(self):
        """验收那两条反向用例：分母不足却自称 met 的分组必须被点名（含按风险那一层），
        没有判决的分组也必须被点名——否则下一句就该问「这条检查到底守没守」。"""
        from docpipe.acceptance import Report, _check_gates

        targets = claim_targets(self.CONFIG)
        big = [self._record(risk="monetary") for _ in range(400)]
        claim = _decision_stats(big, targets)["claim"]

        def gates(by_doc_type, by_risk):
            overall = _decision_stats(big, targets)
            overall["by_doc_type"], overall["by_risk"] = by_doc_type, by_risk
            report = Report()
            _check_gates(report, {"summary": overall}, self.CONFIG)
            return {item["name"]: item for item in report.checks}

        sized = [{"name": "BIG", "accept_count": 400, "claim": claim}]
        clean = gates(sized, [])
        for name in ("gate.sample_size", "gate.small_group_not_claimed",
                     "gate.claim_travels_with_stats"):
            self.assertTrue(clean[name]["ok"], clean[name]["detail"])
        # 分组没有判决 → 必须点名到「哪一级的哪一组」。
        caught = gates(sized + [{"name": "NOCLAIM", "accept_count": 5}], [])
        self.assertFalse(caught["gate.claim_travels_with_stats"]["ok"])
        self.assertIn("by_doc_type:NOCLAIM", caught["gate.claim_travels_with_stats"]["detail"])
        # 伪造一个 accept_count=30 却自称 met 的风险分组：总览体量再大也不许替它背书。
        fake = dict(claim, status="met", accept_count=30)
        dirty = gates(sized, [{"name": "free_text", "accept_count": 30, "claim": fake}])
        self.assertFalse(dirty["gate.small_group_not_claimed"]["ok"])
        self.assertIn("by_risk:free_text", dirty["gate.small_group_not_claimed"]["detail"])


class DocumentPayloadTests(unittest.TestCase):
    """章的位置必须一路走到复核界面，否则「章压住的字」只能靠人在整页里找。"""

    def _payload(self, enhancement):
        from docpipe.run import _document_payload
        parsed = {
            "sha256": "0" * 64,
            "page_count": 1,
            "quality": {},
            "pages": [{"page": 1, "preprocess": {"rectified": True,
                                                 "enhancement": enhancement}}],
        }
        extracted = {"document_id": "D001", "doc_type": "BANK_STATEMENT",
                     "doc_type_confidence": 1.0, "fields": {}}
        return _document_payload(extracted, parsed, Path("a.pdf"))

    def test_seal_regions_ride_along_with_the_payload(self):
        regions = [[0.3, 0.2, 0.7, 0.6]]
        pages = self._payload({"seal_suppressed": True, "seal_regions": regions,
                               "seal_edge_fragment": True})["document_quality"]["preprocess"]
        self.assertEqual(pages[0]["seal_regions"], regions)
        self.assertIsInstance(pages[0]["seal_regions"][0][0], float)
        self.assertTrue(pages[0]["seal_edge_fragment"])
        empty = self._payload({})["document_quality"]["preprocess"]
        self.assertEqual(empty[0]["seal_regions"], [])
        self.assertFalse(empty[0]["seal_edge_fragment"])

    def test_the_console_draws_seals_under_the_field_boxes(self):
        root = Path(__file__).resolve().parents[1] / "docpipe" / "web"
        script = (root / "app.js").read_text(encoding="utf-8")
        style = (root / "style.css").read_text(encoding="utf-8")
        self.assertIn("seal_regions", script)
        self.assertIn("seal-box", script)
        self.assertIn(".seal-box", style)

    def test_the_console_labels_an_edge_fragment_as_a_fragment(self):
        root = Path(__file__).resolve().parents[1] / "docpipe" / "web"
        script = (root / "app.js").read_text(encoding="utf-8")
        style = (root / "style.css").read_text(encoding="utf-8")
        # 整枚章与残片必须在界面上分得开：红框本身看不出区别，靠的是旗标驱动的角标与页眉提示。
        self.assertIn("seal_edge_fragment", script)
        self.assertIn('dataset.label = "骑缝残片"', script)
        self.assertIn("骑缝章残片", script)
        self.assertIn("[data-label]", style)
        self.assertIn("content: attr(data-label)", style)
        # 跨页是否同一枚章还没做，提示语必须留「尚未核对」，不许把没验证的事写成结论。
        self.assertIn("尚未核对", script)
        # 先画章、后画字段框：DOM 后出现的压在上面，字段框的点击与高亮不能被章挡住。
        self.assertLess(script.index("renderSealRegions(overlay)"),
                        script.index("const compact = window.innerWidth < 720"))

    def test_tone_rewrites_ride_along_with_the_payload(self):
        pages = self._payload({"contrast_lifted": True, "brightness_lifted": True})[
            "document_quality"]["preprocess"]
        self.assertTrue(pages[0]["contrast_lifted"])
        self.assertTrue(pages[0]["brightness_lifted"])
        quiet = self._payload({})["document_quality"]["preprocess"]
        self.assertFalse(quiet[0]["contrast_lifted"])
        self.assertFalse(quiet[0]["brightness_lifted"])

    def test_the_console_and_the_report_both_name_the_tone_rewrites(self):
        root = Path(__file__).resolve().parents[1] / "docpipe"
        script = (root / "web" / "app.js").read_text(encoding="utf-8")
        viewer = (root / "report.py").read_text(encoding="utf-8")
        # 复核台与单文件报表是两个独立出口，任一处漏写都会变成「只有另一边知道页面被拉过」。
        for source in (script, viewer):
            self.assertIn("contrast_lifted", source)
            self.assertIn("brightness_lifted", source)
            self.assertIn("对比度拉伸", source)
            self.assertIn("暗页提亮", source)
        # 改写只落在交给 OCR 的那一份像素上，页面显示的仍是拍摄原样——这句必须让人看见。
        self.assertIn("原文页仍是拍摄原样", script)


    def test_the_value_rewrite_note_rides_along_with_the_field(self):
        from docpipe.run import _document_payload
        parsed = {"sha256": "0" * 64, "page_count": 1, "quality": {}, "pages": [{"page": 1}]}
        extracted = {"document_id": "D001", "doc_type": "BANK_STATEMENT",
                     "doc_type_confidence": 1.0,
                     "fields": {
                         "bs.transactions[0].amount": {
                             "value": "-500.00", "decision": "accept",
                             "meta": {"sign_source": "direction", "direction_column": "借方发生额"}},
                         "bs.issue_date": {"value": "2024-01-01", "decision": "accept", "meta": {}},
                     }}
        fields = _document_payload(extracted, parsed, Path("a.pdf"))["fields"]
        self.assertIn("「借方发生额」", fields["bs.transactions[0].amount"]["rewrite"])
        # 没改写过就不许多一个键：全仓库几千个字段都带空字符串等于噪音。
        self.assertNotIn("rewrite", fields["bs.issue_date"])

    def test_every_outlet_names_the_value_rewrite(self):
        from docpipe.review_actions import value_rewrite_note
        root = Path(__file__).resolve().parents[1] / "docpipe"
        console = (root / "web" / "app.js").read_text(encoding="utf-8")
        viewer = (root / "report.py").read_text(encoding="utf-8")
        runner = (root / "run.py").read_text(encoding="utf-8")
        exporter = (root / "export.py").read_text(encoding="utf-8")
        # 四个出口少一个，就变成「只有另一边知道这个值被改过」。
        self.assertIn("field.rewrite", console)
        self.assertIn("值改写", console)
        self.assertIn('"rewrite": field.get("rewrite")', viewer)
        self.assertIn("值改写", viewer)
        for source in (runner, exporter):
            self.assertIn("value_rewrite", source)
            self.assertIn("value_rewrite_note", source)
        self.assertTrue(callable(value_rewrite_note))


class ConfigThresholdTests(unittest.TestCase):
    """铁律：阈值和规则在配置里，不在代码里。这里钉住「改配置真的会移动判据」，
    以及「配置里不许写没有代码读取的键」——装饰性配置键在 ocr_engine 上骗过一次。
    """

    def _inputs(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact"] = None
        field = {
            "path": "bs.opening_balance", "label": "期初余额", "value": "100.00",
            "value_type": "decimal", "required": True, "risk": "monetary",
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "extractor_id": "table_cell:期初余额",
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 0.4, "cross_field_score": 1.0,
                        "grounding_present": 1},
            "meta": {}, "decision": "review", "validators": [], "review_reasons": [],
        }
        parsed = {"page_count": 1, "quality": {"parse_score": 0.45},
                  "pages": [{"page": 1, "width": 200, "height": 200}]}
        doc = {"document_id": "T1", "doc_type": "BANK_STATEMENT"}
        return config, field, parsed, doc

    def _reasons(self, config, field, parsed, doc):
        fresh = json.loads(json.dumps(field))
        return decide_field(fresh, doc, parsed, config, [],
                            Calibrator(config))["review_reasons"]

    def test_page_quality_threshold_moves_the_gate(self):
        config, field, parsed, doc = self._inputs()
        config["validation"]["min_table_score"] = 0.2
        self.assertIn("PARSE_LOW_QUALITY", self._reasons(config, field, parsed, doc))
        config["parsing"]["min_page_quality"] = 0.2
        self.assertNotIn("PARSE_LOW_QUALITY", self._reasons(config, field, parsed, doc))

    def test_table_score_threshold_moves_the_gate(self):
        config, field, parsed, doc = self._inputs()
        config["parsing"]["min_page_quality"] = 0.0
        self.assertIn("TEMPLATE_MISMATCH", self._reasons(config, field, parsed, doc))
        config["validation"]["min_table_score"] = 0.2
        self.assertNotIn("TEMPLATE_MISMATCH", self._reasons(config, field, parsed, doc))

    def test_defaults_reproduce_the_shipped_numbers(self):
        config, field, parsed, doc = self._inputs()
        self.assertEqual(float(config["parsing"]["min_page_quality"]), 0.5)
        self.assertEqual(float(config["validation"]["min_table_score"]), 0.5)

    def test_error_category_follows_the_same_number(self):
        record = {"pred": "1.00", "extractor_id": "table_cell:期初余额",
                  "reasons": [], "signals": {"ocr_conf": 0.99, "table_score": 0.4}}
        self.assertEqual(error_category(record, load_config()), "TABLE_ALIGNMENT")
        loose = copy.deepcopy(load_config())
        loose["validation"]["min_table_score"] = 0.2
        self.assertNotEqual(error_category(record, loose), "TABLE_ALIGNMENT")

    def test_low_ocr_threshold_moves_gate_and_attribution_together(self):
        config, field, parsed, doc = self._inputs()
        field["signals"]["ocr_conf"] = 0.7
        config["validation"]["min_table_score"] = 0.2
        record = {"pred": "100.00", "extractor_id": "table_cell:期初余额",
                  "reasons": [], "signals": copy.deepcopy(field["signals"])}
        self.assertNotIn("LOW_OCR", self._reasons(config, field, parsed, doc))
        self.assertNotEqual(error_category(record, config), "OCR")
        strict = copy.deepcopy(config)
        strict["validation"]["low_ocr_threshold"] = 0.8
        self.assertIn("LOW_OCR", self._reasons(strict, field, parsed, doc))
        self.assertEqual(error_category(record, strict), "OCR")

    def test_code_fallbacks_match_the_shipped_config(self):
        from docpipe.configaudit import audit
        result = audit(Path(__file__).resolve().parents[1], load_config())
        self.assertGreater(result["checked"], 200)
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["unknown_keys"], [])
        # 解析不到配置段的接收者只剩这两处（下标是变量的 `config['doc_types'].get(...)`），
        # 它们退回按键名比对。新出现一个就说明有新的间接访问形态，得先看再放行。
        self.assertEqual([f"{item['file']}({item['owner']})" for item in result["unresolved"]],
                         ["config.py(config['doc_types'].get(doc_type, {}))",
                          "config.py(config['doc_types'].get(doc_type, {}))"])

    def test_fallback_audit_flags_a_disagreeing_default(self):
        from docpipe.configaudit import audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe").mkdir()
            (root / "docpipe" / "probe.py").write_text(
                "def f(config):\n    return config.get('limit', 0.9)\n", encoding="utf-8")
            broken = audit(root, {"limit": 0.6})
            self.assertEqual([item["key"] for item in broken["mismatches"]], ["limit"])
            self.assertEqual(broken["unresolved"], [])
            self.assertEqual(audit(root, {"limit": 0.9})["mismatches"], [])
            missing = audit(root, {})
            self.assertEqual([item["key"] for item in missing["unknown_keys"]], ["limit"])

    def test_fallback_audit_checks_the_section_not_just_the_name(self):
        from docpipe.configaudit import audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe").mkdir()
            (root / "docpipe" / "probe.py").write_text(
                "def f(config):\n"
                "    return config.get('parsing', {}).get('render_dpi', 150)\n"
                "def g(config):\n"
                "    rc = config.get('report', {})\n"
                "    return rc.get('file_name', 'x.html')\n", encoding="utf-8")
            shipped = {"parsing": {"render_dpi": 300}, "ui": {"render_dpi": 150},
                       "report": {"file_name": "x.html"}}
            broken = audit(root, shipped)
            # 150 是 ui.render_dpi 的值：只按键名比会放行，按配置段比必须报出来。
            self.assertEqual([item["path"] for item in broken["mismatches"]],
                             [["parsing", "render_dpi"]])
            self.assertEqual(broken["unresolved"], [])
            shipped["parsing"]["render_dpi"] = 150
            self.assertEqual(audit(root, shipped)["mismatches"], [])

    def test_fallback_audit_resolves_module_constants(self):
        from docpipe.configaudit import audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe").mkdir()
            (root / "docpipe" / "probe.py").write_text(
                "TIMEOUT = 30.0\n"
                "def f(config):\n    return config.get('timeout', TIMEOUT)\n",
                encoding="utf-8")
            # 常量当兜底时也要比：30.0 与配置里的 30 相等（数值相同），改配置必须报出来。
            self.assertEqual(audit(root, {"timeout": 30.0})["mismatches"], [])
            self.assertEqual([item["key"] for item in audit(root, {"timeout": 15})["mismatches"]],
                             ["timeout"])

    def test_fallback_audit_admits_when_it_cannot_locate_the_section(self):
        from docpipe.configaudit import audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe").mkdir()
            (root / "docpipe" / "probe.py").write_text(
                "def f(config):\n    return _options(config).get('limit', 0.9)\n",
                encoding="utf-8")
            result = audit(root, {"limit": 0.9})
            self.assertEqual([item["owner"] for item in result["unresolved"]],
                             ["_options(config)"])
            self.assertIsNone(audit(root, {"limit": 0.6})["mismatches"][0]["path"])

    def test_fallback_audit_follows_a_private_helper_parameter(self):
        from docpipe.configaudit import audit
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe").mkdir()
            (root / "docpipe" / "probe.py").write_text(
                "def _opts(report_config, override):\n"
                "    return {'dpi': report_config.get('page_dpi', 96)}\n"
                "def build(config):\n"
                "    report_config = config.get('report', {})\n"
                "    return _opts(report_config, None)\n", encoding="utf-8")
            shipped = {"report": {"page_dpi": 96}, "ui": {"page_dpi": 200}}
            self.assertEqual(audit(root, shipped)["unresolved"], [])
            self.assertEqual(audit(root, shipped)["mismatches"], [])
            # 96 是 report 段的值、200 是 ui 段的；按段比必须报出 report.page_dpi。
            shipped["report"]["page_dpi"] = 120
            self.assertEqual([item["path"] for item in audit(root, shipped)["mismatches"]],
                             [["report", "page_dpi"]])
            # 两个调用点指向不同段时不猜：整体退回按键名比并如实进 unresolved。
            (root / "docpipe" / "probe.py").write_text(
                "def _opts(report_config, override):\n"
                "    return {'dpi': report_config.get('page_dpi', 96)}\n"
                "def a(config):\n    return _opts(config.get('report', {}), None)\n"
                "def b(config):\n    return _opts(config.get('ui', {}), None)\n",
                encoding="utf-8")
            conflicting = audit(root, {"report": {"page_dpi": 96},
                                       "ui": {"page_dpi": 96}})
            self.assertEqual([item["owner"] for item in conflicting["unresolved"]],
                             ["report_config"])

    def test_json_indent_setting_reaches_delivered_files(self):
        for indent, expected in ((2, 2), (4, 4)):
            config = copy.deepcopy(load_config())
            config["output"]["json_indent"] = indent
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / "out"
                (out / "T1").mkdir(parents=True)
                (out / "summary.json").write_text(json.dumps({
                    "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                                   "path": "T1/T1.json"}],
                }), encoding="utf-8")
                (out / "T1" / "T1.json").write_text(json.dumps({
                    "document_id": "T1", "doc_type": "BANK_STATEMENT", "fields": {},
                }, ensure_ascii=False), encoding="utf-8")
                export(out, out / "final", config)
                lines = (out / "final" / "final_summary.json").read_text(
                    encoding="utf-8").splitlines()
                body = next(line for line in lines[1:] if line.strip())
                self.assertEqual(len(body) - len(body.lstrip(" ")), expected)

    def test_judgement_sections_have_no_unread_keys(self):
        """parsing / validation / output 下每个声明的键都必须被 docpipe/ 按名字读取。

        code_patterns 的子键是按规则名动态查的（trust.py 取 rule_ids 交集），
        永远不会以字面量出现在代码里，所以只有它豁免。
        """
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((Path(__file__).resolve().parents[1] / "docpipe").glob("*.py")))
        dead = []
        for section in ("parsing", "validation", "output"):
            for key, value in load_config()[section].items():
                names = [key] + (list(value) if isinstance(value, dict) and key != "code_patterns" else [])
                for name in names:
                    if f'"{name}"' not in sources and f"'{name}'" not in sources:
                        dead.append(f"{section}.{name}")
        self.assertEqual(dead, [])


class DocTypeGateTests(unittest.TestCase):
    """文档类型判定的两道门与置信度公式都在配置里（过去硬写在 parse.py）。

    判决本身（认出哪一类）没变，默认值逐字等于旧硬编码；这三条钉住「默认不变」
    和「改配置真的会动判决」，防止再退回装饰性配置。
    """

    PROBE = [{"text": "银行流水 交易明细 账号 户名 期末余额"}]

    def test_defaults_reproduce_the_shipped_numbers(self):
        options = load_config()["parsing"]["doc_type"]
        self.assertEqual(float(options["min_score"]), 3.0)
        self.assertEqual(float(options["min_margin"]), 1.0)
        self.assertEqual(float(options["confidence_base"]), 0.55)
        self.assertEqual(float(options["confidence_per_point"]), 0.04)
        self.assertEqual(float(options["confidence_cap"]), 0.99)
        self.assertEqual(detect_doc_type(self.PROBE, load_config()),
                         ("BANK_STATEMENT", 0.87))

    def test_each_gate_alone_can_reject_and_both_zero_lets_anything_through(self):
        margin_strict = copy.deepcopy(load_config())
        margin_strict["parsing"]["doc_type"]["min_margin"] = 99
        self.assertEqual(detect_doc_type(self.PROBE, margin_strict), ("UNKNOWN", 0.0))
        score_strict = copy.deepcopy(load_config())
        score_strict["parsing"]["doc_type"]["min_score"] = 99
        self.assertEqual(detect_doc_type(self.PROBE, score_strict), ("UNKNOWN", 0.0))
        loose = copy.deepcopy(load_config())
        loose["parsing"]["doc_type"].update({"min_score": 0, "min_margin": 0})
        typed, confidence = detect_doc_type([{"text": "议付行 密押 骑缝"}], loose)
        self.assertNotEqual(typed, "UNKNOWN")
        self.assertGreater(confidence, 0.0)

    def test_confidence_follows_base_per_point_and_cap(self):
        config = copy.deepcopy(load_config())
        config["parsing"]["doc_type"].update(
            {"confidence_base": 0.5, "confidence_per_point": 0.05})
        self.assertEqual(detect_doc_type(self.PROBE, config),
                         ("BANK_STATEMENT", 0.9))
        config["parsing"]["doc_type"]["confidence_cap"] = 0.8
        self.assertEqual(detect_doc_type(self.PROBE, config),
                         ("BANK_STATEMENT", 0.8))


class ValueRewriteNoteTests(unittest.TestCase):
    """屏幕上 -500.00、纸上那格没有负号；屏幕上 12,000、纸上印「1.2 万元」。
    这类改写本身是对的，不说出来就会被当成抄错。"""

    def test_unit_conversion_quotes_the_unit_printed_on_the_page(self):
        from docpipe.extract import _normalize
        from docpipe.review_actions import value_rewrite_note
        value, status, meta = _normalize(
            "1.2", {"type": "decimal", "unit_from_document": True}, load_config(),
            {"unit": "CNY_WAN_YUAN", "unit_factor": 10000, "unit_text": "万元"})
        self.assertEqual((status, value), ("ok", "12000.00"))
        self.assertEqual(meta["unit_factor"], 10000)
        self.assertEqual(meta["unit_text"], "万元")
        note = value_rewrite_note({"meta": meta})
        self.assertIn("万元", note)
        self.assertIn("×10,000", note)
        self.assertIn("纸上印的是折算前的数", note)

    def test_a_page_that_already_states_yuan_is_not_called_a_rewrite(self):
        from docpipe.review_actions import value_rewrite_note
        self.assertEqual(value_rewrite_note(
            {"meta": {"unit_from_document": True, "unit_factor": 1, "unit_text": "元"}}), "")

    def test_a_unit_that_was_never_found_is_admitted(self):
        from docpipe.review_actions import value_rewrite_note
        note = value_rewrite_note({"meta": {"unit_from_document": True, "unit_unknown": True}})
        self.assertIn("没找到单位说明", note)
        self.assertIn("原数", note)

    def test_direction_cell_text_is_recorded_and_quoted(self):
        from docpipe.extract import _normalize
        from docpipe.review_actions import value_rewrite_note
        value, status, meta = _normalize(
            "500.00", {"type": "decimal", "signed_by_direction": True,
                       "direction_columns": {"借方发生额": "支出", "贷方发生额": "收入"}},
            load_config(), {}, {"收支方向": {"text": "支出"}, "交易金额": {"text": "500.00"}})
        self.assertEqual((status, value), ("ok", "-500.00"))
        self.assertEqual(meta["sign_source"], "direction")
        self.assertEqual(meta["direction_text"], "支出")
        self.assertIn("「支出」", value_rewrite_note({"meta": meta}))
        self.assertIn("纸上这一格没有负号", value_rewrite_note({"meta": meta}))

    def test_credit_layout_names_the_column_the_amount_sat_in(self):
        from docpipe.extract import _normalize
        from docpipe.review_actions import value_rewrite_note
        value, status, meta = _normalize(
            "500.00", {"type": "decimal", "signed_by_direction": True,
                       "direction_columns": {"借方发生额": "支出", "贷方发生额": "收入"}},
            load_config(), {}, {"借方发生额": {"text": "500.00"}, "贷方发生额": {"text": ""}})
        self.assertEqual((status, value), ("ok", "-500.00"))
        self.assertEqual(meta["direction_column"], "借方发生额")
        # 方向不是从「收支方向」列印出来的，就不许假装引用了一个不存在的字。
        self.assertNotIn("direction_text", meta)
        note = value_rewrite_note({"meta": meta})
        self.assertIn("「借方发生额」", note)
        self.assertNotIn("列印的「」", note)

    def test_conflict_between_printed_sign_and_direction_is_its_own_sentence(self):
        from docpipe.review_actions import value_rewrite_note
        note = value_rewrite_note({"meta": {"sign_source": "text", "sign_conflict": True}})
        self.assertIn("带负号", note)
        self.assertIn("收入", note)
        self.assertNotIn("负号是按", note)

    def test_both_rewrites_on_one_field_are_said_together(self):
        from docpipe.review_actions import value_rewrite_note
        note = value_rewrite_note({"meta": {"unit_from_document": True, "unit_factor": 10000,
                                            "unit_text": "万元", "sign_source": "direction",
                                            "direction_text": "支出"}})
        self.assertIn("万元", note)
        self.assertIn("「支出」", note)
        self.assertEqual(note.count("；"), 1)

    def test_a_field_nobody_rewrote_gets_no_note(self):
        from docpipe.review_actions import value_rewrite_note
        self.assertEqual(value_rewrite_note({"meta": {"raw_grounded": True}}), "")
        self.assertEqual(value_rewrite_note({}), "")


class ChainDisclosureTests(unittest.TestCase):
    """流水链被判坏时只给一个 CHAIN_UNRELIABLE 码：6 行里错 5 行和错 30 行推给同样
    一批字段，复核员看不出这份流水坏到什么程度，只能整份重看。诊断数本来就算过，
    算完就丢——这里盯的是「算了但没说」这一类。"""

    GOOD = [("100.00", "1100.00"), ("100.00", "1200.00"), ("100.00", "1300.00")]

    @classmethod
    def _statement(cls, rows=None, closing="1300.00", totals=None, lines=()):
        def field(path, value, risk="monetary"):
            return {"path": path, "label": path.rsplit(".", 1)[-1], "value": value,
                    "value_type": "decimal", "required": True, "risk": risk,
                    "grounding": {"page": 1, "bbox": [10.0, 10.0, 80.0, 20.0]},
                    "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                                "table_score": 1.0, "cross_field_score": 1.0,
                                "grounding_present": 1},
                    "meta": {}, "decision": "review", "validators": [], "review_reasons": []}

        rows = cls.GOOD if rows is None else rows
        fields = {"bs.account_number": field("bs.account_number", "6222021234567890", "identity"),
                  "bs.opening_balance": field("bs.opening_balance", "1000.00")}
        if closing is not None:
            fields["bs.closing_balance"] = field("bs.closing_balance", closing)
        if totals is not None:
            fields["bs.total_income"] = field("bs.total_income", totals[0])
            fields["bs.total_expense"] = field("bs.total_expense", totals[1])
        for index, (amount, balance) in enumerate(rows):
            for column, value in (("amount", amount), ("balance", balance),
                                  ("txn_date", f"2026-03-{index + 1:02d}")):
                risk = "date" if column == "txn_date" else "monetary"
                fields[f"bs.transactions[{index}].{column}"] = field(
                    f"bs.transactions[{index}].{column}", value, risk)
        doc = {"document_id": "CD1", "doc_type": "BANK_STATEMENT", "fields": fields}
        parsed = {"document_id": "CD1", "doc_type": "BANK_STATEMENT", "page_count": 1,
                  "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 595.0, "height": 842.0, "quality_flags": [],
                             "lines": [{"text": text} for text in lines], "tables": []}]}
        return doc, parsed

    @classmethod
    def _stats(cls, doc, parsed):
        from docpipe.trust import validate_bank_statement
        validators = {}
        return validate_bank_statement(doc, parsed, load_config(), validators)

    def test_healthy_statement_says_nothing(self):
        from docpipe.review_actions import chain_note
        doc, parsed = self._statement()
        stats = self._stats(doc, parsed)
        self.assertEqual(stats["triggers"], [])
        self.assertFalse(stats["chain_unreliable"])
        self.assertEqual(chain_note(stats), "")
        # 缺键的老 payload 也不许凭空造句（链没跑过的文档不该看到一句「链判坏了」）。
        self.assertEqual(chain_note({}), "")
        self.assertEqual(chain_note({"chain_unreliable": True, "triggers": []}), "")

    def test_each_kind_of_broken_chain_names_its_own_evidence(self):
        from docpipe.review_actions import chain_note
        broken = [
            ([self.GOOD[0]] + [("100.00", "999999.00")] * 5, None, None, (), "ratio"),
            (None, "1500.00", None, (), "closing"),
            (None, "1300.00", ("900.00", "0.00"), (), "totals"),
            (None, "1300.00", None, ("账号：6222020000000000",), "account"),
        ]
        needles = {"ratio": "6 行里 5 行", "closing": "末行余额与期末余额对不上",
                   "totals": "期初＋收入－支出与期末余额对不上", "account": "跨页账号不是同一个"}
        for rows, closing, totals, lines, kind in broken:
            with self.subTest(kind=kind):
                doc, parsed = self._statement(rows, closing, totals, lines)
                stats = self._stats(doc, parsed)
                self.assertEqual(stats["triggers"], [kind])
                note = chain_note(stats)
                self.assertTrue(note.startswith("流水链被判不可靠："), note)
                self.assertIn(needles[kind], note)

    def test_the_ratio_sentence_quotes_the_real_counts(self):
        from docpipe.review_actions import chain_note
        note = chain_note({"chain_unreliable": True, "triggers": ["ratio"], "rows": 40,
                           "chain_failures": 3, "failure_ratio": 0.075})
        self.assertIn("40 行里 3 行", note)
        self.assertIn("7.5%", note)
        # 只说「链不可靠」等于没说：人数必须出现在句子里。
        self.assertNotEqual(note, "流水链被判不可靠：")

    def test_no_trigger_code_is_left_without_a_sentence(self):
        """加了第五种触发证据却忘了造句，那句话就永远只说一半——和 P0-8 漏文案同一条纪律。"""
        from docpipe.review_actions import chain_note
        source = (ROOT / "docpipe" / "trust.py").read_text(encoding="utf-8")
        codes = re.findall(r'triggers\.append\("([a-z]+)"\)', source)
        self.assertTrue(codes, "validate_bank_statement 的触发码写法变了，同步改这条检查")
        self.assertEqual(len(set(codes)), len(codes))
        sentences = {}
        for code in codes:
            note = chain_note({"chain_unreliable": True, "triggers": [code], "rows": 6,
                               "chain_failures": 5, "failure_ratio": 0.8333})
            self.assertTrue(note, code)
            sentences[code] = note.split("：", 1)[1]
        self.assertEqual(len(set(sentences.values())), len(codes), sentences)
        both = chain_note({"chain_unreliable": True, "triggers": ["ratio", "account"],
                           "rows": 6, "chain_failures": 5, "failure_ratio": 0.8333})
        self.assertEqual(both.count("；"), 1, both)

    def test_the_note_rides_only_on_the_fields_the_chain_blocked(self):
        from docpipe.run import _document_payload
        doc, parsed = self._statement()
        doc["chain_stats"] = {"chain_unreliable": True, "triggers": ["closing"]}
        doc["fields"]["bs.closing_balance"]["review_reasons"] = ["CHAIN_UNRELIABLE"]
        doc["fields"]["bs.bank_name"] = dict(doc["fields"]["bs.opening_balance"])
        doc["fields"]["bs.bank_name"]["review_reasons"] = ["LOW_OCR"]
        payload = _document_payload(doc, parsed, Path("cd1.pdf"))
        self.assertIn("流水链", payload["fields"]["bs.closing_balance"]["chain"])
        self.assertNotIn("chain", payload["fields"]["bs.bank_name"])
        self.assertNotIn("chain", payload["fields"]["bs.opening_balance"])
        # 链健康时同样的判决一个 chain 键都不许有：几千格都带提示，这句话当天作废。
        healthy, healthy_parsed = self._statement()
        healthy["chain_stats"] = {"chain_unreliable": False, "triggers": []}
        healthy["fields"]["bs.closing_balance"]["review_reasons"] = ["CHAIN_UNRELIABLE"]
        quiet = _document_payload(healthy, healthy_parsed, Path("cd1.pdf"))["fields"]
        self.assertFalse([path for path, field in quiet.items() if field.get("chain")])

    def test_the_web_report_lifts_the_sentence_once_per_document(self):
        from docpipe.report import _doc_chain_note
        sentence = "流水链被判不可靠：6 行里 5 行余额接不上（83.3%）"
        doc = {"fields": {"bs.opening_balance": {"chain": sentence},
                          "bs.closing_balance": {"chain": sentence}}}
        self.assertEqual(_doc_chain_note(doc), sentence)
        self.assertEqual(_doc_chain_note({"fields": {"bs.opening_balance": {}}}), "")

    def test_both_field_csvs_carry_the_column_and_agree_with_acceptance(self):
        from docpipe.acceptance import CSV_COLUMNS
        from docpipe.export import COLUMNS
        from docpipe.run import _write_csv
        payload = [{"document_id": "CD1", "doc_type": "BANK_STATEMENT", "fields": {
            "bs.closing_balance": {"value": "1300.00", "value_type": "decimal",
                                   "confidence": 0.4, "decision": "review",
                                   "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
                                   "extractor_id": "label_value:期末余额",
                                   "review_reasons": ["CHAIN_UNRELIABLE"],
                                   "chain": "流水链被判不可靠：末行余额与期末余额对不上"},
            "bs.opening_balance": {"value": "1000.00", "value_type": "decimal",
                                   "confidence": 0.9, "decision": "accept",
                                   "grounding": {"page": 1, "bbox": [10, 30, 80, 40]},
                                   "extractor_id": "label_value:期初余额",
                                   "review_reasons": []},
        }}]
        config = load_config()
        with tempfile.TemporaryDirectory() as temp:
            _write_csv(Path(temp), payload, config)
            with (Path(temp) / "fields.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), CSV_COLUMNS)
        # 两份 CSV 的列不同（final 多了最终值/复核状态），但披露列必须一起走到尾巴上：
        # 一边加了另一边没加，就变成「纸上查得到的证据屏幕上查不到」。
        self.assertEqual(COLUMNS[-3:], CSV_COLUMNS[-3:])
        self.assertEqual(COLUMNS[-3:], ["review_actions", "value_rewrite", "chain_evidence"])
        self.assertTrue(rows[0]["chain_evidence"].startswith("流水链被判不可靠"))
        self.assertEqual(rows[1]["chain_evidence"], "")


from docpipe.offlineaudit import (DEFAULTS, audit as offline_audit, network_calls,
                                  options as offline_options, remote_asset_refs,
                                  remote_urls)


class OfflineAuditTests(unittest.TestCase):
    """铁律审计 VII：「全链路本地/离线」做成静态门槛，扫描器自己也必须会咬人。"""

    def setUp(self):
        self.config = load_config()
        self.opts = offline_options(self.config)

    def test_loopback_and_namespace_are_not_exfiltration(self):
        text = ('http://127.0.0.1:8766/api/x http://localhost/ '
                'xmlns="http://www.w3.org/2000/svg" '
                'https://cdn.example.com/font.css ftp://files.example.com/pub')
        self.assertEqual([url for url, _ in remote_urls(text, self.opts)],
                         ["https://cdn.example.com", "ftp://files.example.com"])

    def test_the_judgement_reads_the_config_not_a_hardcoded_list(self):
        text = "ftp://files.example.com/pub https://cdn.example.com/a"
        self.assertEqual(len(remote_urls(text, self.opts)), 2)
        no_ftp = offline_options({"security": {"offline": {"blocked_schemes": ["https"]}}})
        self.assertEqual([url for url, _ in remote_urls(text, no_ftp)], ["https://cdn.example.com"])
        trust_ftp = offline_options({"security": {"offline": {"loopback_hosts": ["files.example.com"]}}})
        self.assertEqual([url for url, _ in remote_urls(text, trust_ftp)], ["https://cdn.example.com"])
        # 豁免本身也是配置项：清掉 inert_attrs，命名空间声明就要报——证明它不是写死的白名单。
        strict = offline_options({"security": {"offline": {"inert_attrs": []}}})
        self.assertEqual([url for url, _ in remote_urls('xmlns="http://www.w3.org/2000/svg"', strict)],
                         ["http://www.w3.org"])

    def test_network_calls_and_imports_are_found(self):
        source = ('import socket\nimport requests\n'
                  'def f():\n'
                  '    return socket.create_connection(("10.0.0.1", 9))\n')
        labels = {item["call"] for item in network_calls(source)}
        self.assertEqual(labels, {"import socket", "import requests", "socket.create_connection"})
        self.assertEqual(network_calls("import json\ndef f():\n    return json\n"), [])

    def test_asset_refs_allow_relative_and_reject_remote(self):
        html = ('<link rel="stylesheet" href="app.css"><a href="#top">x</a>'
                '<img src="data:image/png;base64,AAAA">'
                '<img src="//cdn.example.com/b.png">'
                '<script>fetch("https://api.example.com/v1")</script>'
                '<style>background:url(/static/bg.png)</style>'
                '<svg xmlns="http://www.w3.org/2000/svg">'
                '<image xlink:href="http://cdn.example.com/logo.png"/></svg>')
        self.assertEqual(sorted(item["url"] for item in remote_asset_refs(html, self.opts)),
                         sorted(["//cdn.example.com", "https://api.example.com",
                                 "http://cdn.example.com"]))

    def test_the_shipped_delivery_surface_is_clean(self):
        result = offline_audit(ROOT, self.config)
        self.assertGreaterEqual(result["scanned"], 20)
        for key in ("remote_literals", "network_call_sites", "remote_asset_refs"):
            self.assertEqual(result[key], [], key)

    def test_the_audit_bites_on_a_planted_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docpipe" / "web").mkdir(parents=True)
            (root / "docpipe" / "plant.py").write_text(
                'FONT = "https://cdn.example.com/a.css"\nimport requests\n', encoding="utf-8")
            (root / "docpipe" / "web" / "index.html").write_text(
                '<img src="//evil.example.com/b.png">', encoding="utf-8")
            result = offline_audit(root, self.config)
        self.assertEqual([item["url"] for item in result["remote_literals"]],
                         ["https://cdn.example.com"])
        self.assertEqual([item["call"] for item in result["network_call_sites"]], ["import requests"])
        self.assertEqual([item["url"] for item in result["remote_asset_refs"]],
                         ["//evil.example.com"])

    def test_code_defaults_match_the_shipped_security_section(self):
        shipped = ((self.config.get("security") or {}).get("offline") or {})
        self.assertEqual(sorted(shipped), sorted(DEFAULTS))
        for key, value in DEFAULTS.items():
            self.assertEqual(shipped.get(key), value, key)
        self.assertEqual(offline_options({}), dict(DEFAULTS))


from docpipe.deliveryaudit import (DEFAULTS as DELIVERY_DEFAULTS,
                                   audit as delivery_audit,
                                   document_files as delivery_document_files,
                                   main as deliveryaudit_main,
                                   options as delivery_options)


class DeliveryAuditTests(unittest.TestCase):
    """铁律审计 VIII：出货的每一格必须带着判决层的痕迹，三层计数必须对得上。"""

    @staticmethod
    def _grounding(page=1):
        """交付面的坐标必须**成对**：`bbox` 是页面尺度下的原始框，`bbox_norm` 才是前台摆的那套
        百分比方框。夹具只许写「同一个页面尺度除得出来」的对，否则第六问会在共用同一个 CLI 的
        前面几轮（IX/X）测试里假报警。"""
        width, height = 595.0, 842.0
        box = [30.0, 60.0, 120.0, 78.0]
        return {"page": page, "bbox": box,
                "bbox_norm": [round(box[0] / width, 4), round(box[1] / height, 4),
                              round(box[2] / width, 4), round(box[3] / height, 4)]}

    @staticmethod
    def _delivery(root, docs):
        """写一份「每份文档一个 JSON + summary.json」的最小交付面，三层都自洽。

        每一格还要**能被出厂门槛解释**（第七问）：`risk` 与 `path` 都写出来，放行的格子
        置信度取 `optional_low_risk` 档门槛（0.93）之上的 0.95，而 `review` 的格子故意留
        0.9——它本来就低于门槛，那是它进复核队列的原因，不是矛盾。铁律 IX/X/XI 那三轮共用
        这个 CLI，少了这两笔就会被第七问打成退出码 1（和 XI 的坐标成对是同一个坑）。
        """
        total = {"field_count": 0, "accept_count": 0, "review_count": 0}
        listed = []
        for document_id, items in docs.items():
            fields = {}
            for path, state in items:
                fields[path] = {"value": path, "path": path, "risk": "optional_low_risk",
                                "decision": state, "confidence": 0.95 if state == "accept" else 0.9,
                                "review_reasons": [] if state == "accept" else ["REQUIRED_EMPTY"],
                                "grounding": DeliveryAuditTests._grounding()}
            folder = root / document_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{document_id}.json").write_text(
                json.dumps({"document_id": document_id, "source": {"page_count": 1},
                            "fields": fields}, ensure_ascii=False),
                encoding="utf-8")
            counts = {"field_count": len(fields), "accept_count": 0, "review_count": 0}
            for _, state in items:
                counts[f"{state}_count"] += 1
            listed.append(dict(counts, document_id=document_id))
            for key, value in counts.items():
                total[key] += value
        (root / "summary.json").write_text(
            json.dumps({"overall": total, "documents": listed}, ensure_ascii=False),
            encoding="utf-8")
        return root

    def setUp(self):
        self.config = load_config()

    def test_a_trace_clean_delivery_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out", {
                "D001": [("bl.company_name", "accept"), ("bl.issue_date", "accept")],
                "D002": [("vi.total_amount", "review")],
            })
            result = delivery_audit(root, self.config)
        self.assertEqual(result["undecided"], [])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["by_state"], {"accept": 2, "review": 1})
        self.assertEqual(result["fields"], 3)

    def test_a_field_missing_the_judgement_keys_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out",
                                  {"D001": [("bl.a", "accept"), ("bl.b", "accept")]})
            payload = json.loads((root / "D001" / "D001.json").read_text(encoding="utf-8"))
            for key in ("decision", "confidence"):
                del payload["fields"]["bl.a"][key]
            (root / "D001" / "D001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = delivery_audit(root, self.config)
        self.assertEqual([(item["document_id"], item["offenders"]) for item in result["undecided"]],
                         [("D001", [{"path": "bl.a", "problem": "missing_keys",
                                     "missing": ["decision", "confidence"]}])])
        # 缺判决痕迹的那一格数不进任何状态，汇总里的 accept_count 就再也复现不出来。
        self.assertIn("accept", [item["problem"] for item in result["mismatches"]])

    def test_an_invented_state_is_reported_even_with_every_key_present(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out", {"D001": [("bl.a", "accept")]})
            payload = json.loads((root / "D001" / "D001.json").read_text(encoding="utf-8"))
            payload["fields"]["bl.a"]["decision"] = "accepted"
            (root / "D001" / "D001.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result = delivery_audit(root, self.config)
        self.assertEqual([item["offenders"] for item in result["undecided"]],
                         [[{"path": "bl.a", "problem": "unknown_state", "decision": "accepted"}]])

    def test_counts_must_agree_between_detail_summary_and_overall(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out", {"D001": [("bl.a", "accept")],
                                                       "D002": [("bl.b", "review")]})
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            summary["overall"]["accept_count"] += 4  # 只动汇总，明细一个字节不动
            (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False),
                                               encoding="utf-8")
            result = delivery_audit(root, self.config)
        self.assertEqual(result["undecided"], [])
        layers = {(item["layer"], item["problem"]) for item in result["mismatches"]}
        self.assertIn(("summary.overall", "accept"), layers)
        self.assertIn(("documents_vs_overall", "accept"), layers)

    def test_a_document_in_the_summary_without_a_payload_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out", {"D001": [("bl.a", "accept")],
                                                       "D002": [("bl.b", "accept")]})
            shutil.rmtree(root / "D002")
            result = delivery_audit(root, self.config)
        self.assertEqual([(item["document_id"], item["problem"], item["expected"], item["actual"])
                          for item in result["mismatches"] if item["problem"] == "no_payload"],
                         [("D002", "no_payload", 1, 0)])

    def test_the_required_keys_and_states_come_from_the_config(self):
        lenient_cfg = {"validation": {"delivery": {"required_field_keys": ["decision"]}}}
        with tempfile.TemporaryDirectory() as temp:
            root = self._delivery(Path(temp) / "out", {"D001": [("bl.a", "accept"),
                                                                ("bl.b", "review")]})
            strict = delivery_audit(root, {"validation": {"delivery":
                                                          {"decided_states": ["accept"]}}})
            lenient = delivery_audit(root, lenient_cfg)
            self.assertEqual([(item["offender_count"], item["offenders"][0]["problem"])
                              for item in strict["undecided"]], [(1, "unknown_state")])
            self.assertEqual(strict["mismatches"], [])
            self.assertEqual(lenient["undecided"], [])
            payload = json.loads((root / "D001" / "D001.json").read_text(encoding="utf-8"))
            del payload["fields"]["bl.b"]["grounding"]
            (root / "D001" / "D001.json").write_text(json.dumps(payload, ensure_ascii=False),
                                                     encoding="utf-8")
            offenders = delivery_audit(root, self.config)["undecided"][0]["offenders"]
            self.assertEqual(offenders[0]["missing"], ["grounding"])
            # 同一份改动，在「只要求 decision」的配置下什么都不报。
            self.assertEqual(delivery_audit(root, lenient_cfg)["undecided"], [])

    def test_a_thin_scan_is_a_failure_not_a_clean_bill(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = Path(temp) / "out"
            empty.mkdir()
            self.assertEqual(delivery_document_files(empty), [])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = deliveryaudit_main(["--out", str(empty)])
        self.assertEqual(code, 1)
        self.assertIn("覆盖不足", stdout.getvalue())

    def test_code_defaults_match_the_shipped_delivery_section(self):
        shipped = ((self.config.get("validation") or {}).get("delivery") or {})
        self.assertEqual(sorted(shipped), sorted(DELIVERY_DEFAULTS))
        for key, value in DELIVERY_DEFAULTS.items():
            self.assertEqual(shipped.get(key), value, key)
        self.assertEqual(delivery_options({}), dict(DELIVERY_DEFAULTS))


from docpipe.deliveryaudit import MANIFEST_NAME as PACKAGE_MANIFEST_NAME
from docpipe.deliveryaudit import package_report as delivery_package_report
from docpipe.export import expected_members as expected_package_members


class DeliveryPackageTests(unittest.TestCase):
    """铁律审计 IX：交付包自己必须说得出装了哪些文件，清单与包内成员逐字对得上。

    `final/` 原本是照着磁盘扫的：上一轮遗留的调试表、别人塞进来的原始 PDF 都会跟着出去，
    而清单里只有一句粗粒度的 `contents`，事后查不出来。这里把「说得出」拆成能各自失败的
    几条：缺字段、包里有而清单没记、清单记了而包里没有、内容哈希验不上、计数对不上、下限。
    """

    @staticmethod
    def _files(count=20):
        """合成包内成员：`final/m00.json` … 全是无害的小文件。"""
        return {f"final/m{i:02d}.json": json.dumps({"i": i}).encode("utf-8")
                for i in range(count)}

    @staticmethod
    def _entries(files, drop=(), lying_name=None):
        """照着一份文件表写清单条目；`drop` 模拟清单漏记，`lying_name` 模拟哈希写错。"""
        entries = []
        for name, payload in sorted(files.items()):
            if name in drop:
                continue
            digest = hashlib.sha256(payload).hexdigest()
            entries.append({"name": name, "bytes": len(payload),
                            "sha256": "0" * 64 if name == lying_name else digest})
        return entries

    @staticmethod
    def _manifest(entries, member_count=None, skipped=()):
        """一份说得出自己装了什么（或少装了什么）的清单：十个必需字段全在。"""
        return {"source_out": "out/dev", "workspace": "dev",
                "exported_at": "2026-10-28T00:00:00+08:00", "document_count": 1,
                "overall": {"field_count": len(entries)},
                "report": {"file": "查看结果.html", "size": 1, "embedded_pages": 0},
                "contents": ["final/", "查看结果.html", PACKAGE_MANIFEST_NAME],
                "member_count": len(entries) + 1 if member_count is None else member_count,
                "members": entries,
                "skipped_members": [dict(item) for item in skipped]}

    @staticmethod
    def _package(root, files, manifest, name="交付包.zip"):
        path = Path(root) / name
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in sorted(files.items()):
                archive.writestr(name, payload)
            archive.writestr(PACKAGE_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False))
        return path

    def setUp(self):
        self.config = load_config()
        self.opts = delivery_options(self.config)

    def _report(self, temp, files, entries, **manifest_kwargs):
        path = self._package(Path(temp), files, self._manifest(entries, **manifest_kwargs))
        return delivery_package_report(path, self.opts)

    def test_a_package_that_names_every_member_is_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            result = self._report(temp, files, self._entries(files))
        self.assertEqual(result["missing_keys"], [])
        self.assertEqual(result["unlisted"], [])
        self.assertEqual(result["phantom"], [])
        self.assertEqual(result["bad_digest"], [])
        self.assertEqual(result["skipped"], [])
        self.assertFalse(result["thin"])
        # 清单声明的条数与逐条数出来的条数必须相等——差一条就是有人在别处改过包。
        self.assertEqual(result["member_count"], result["declared_count"])
        self.assertEqual(result["member_count"], len(files) + 1)

    def test_a_member_the_manifest_forgets_is_named_even_when_the_count_is_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            forgotten = sorted(files)[3]
            entries = self._entries(files, drop=[forgotten])
            # 连声明条数一起改小：粗计数完全自洽，只有逐条比对才看得见少了谁。
            result = self._report(temp, files, entries)
        self.assertEqual(result["unlisted"], [forgotten])
        self.assertEqual(result["phantom"], [])
        self.assertEqual(result["member_count"], result["declared_count"])

    def test_a_manifest_entry_with_no_file_in_the_package_is_phantom(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            entries = self._entries(files)
            entries.append({"name": "final/本轮没写.json", "bytes": 2,
                            "sha256": hashlib.sha256(b"{}").hexdigest()})
            result = self._report(temp, files, entries,
                                  member_count=len(entries) + 1)
        self.assertEqual(result["phantom"], ["final/本轮没写.json"])
        self.assertEqual(result["unlisted"], [])

    def test_a_rewritten_member_is_caught_by_its_hash_not_by_its_name(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            changed = sorted(files)[5]
            entries = self._entries(files, lying_name=changed)
            result = self._report(temp, files, entries)
        self.assertEqual(result["bad_digest"], [changed])
        self.assertEqual(result["unlisted"], [])
        self.assertEqual(result["phantom"], [])

    def test_a_count_that_disagrees_with_the_member_list_is_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            result = self._report(temp, files, self._entries(files), member_count=99)
        self.assertEqual(result["member_count"], len(files) + 1)
        self.assertEqual(result["declared_count"], 99)
        self.assertNotEqual(result["member_count"], result["declared_count"])

    def test_a_manifest_missing_a_required_key_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            path = self._package(Path(temp), files, {key: value for key, value in
                                                     self._manifest(self._entries(files)).items()
                                                     if key != "skipped_members"})
            result = delivery_package_report(path, self.opts)
        self.assertEqual(result["missing_keys"], ["skipped_members"])

    def test_refused_leftovers_are_carried_in_the_manifest_and_shown(self):
        with tempfile.TemporaryDirectory() as temp:
            files = self._files()
            leftovers = [{"name": "final/上一轮的调试表.csv", "bytes": 6,
                          "reason": "not_exported_this_run"}]
            path = self._package(Path(temp), files,
                                 self._manifest(self._entries(files), skipped=leftovers))
            clean = delivery_package_report(path, self.opts)
            strict = delivery_package_report(
                path, dict(self.opts, min_package_members=len(files) + 2))
        self.assertEqual([item["name"] for item in clean["skipped"]],
                         ["final/上一轮的调试表.csv"])
        # 下限读的是配置：同一个包，把门限抬一条就变成「这一轮基本没被审到」。
        self.assertFalse(clean["thin"])
        self.assertTrue(strict["thin"])

    def test_the_cli_only_bites_once_the_package_leg_is_asked_for(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = DeliveryAuditTests._delivery(root / "out", {
                f"D00{i}": [(f"bl.f{j}", "accept") for j in range(200)] for i in range(1, 6)})
            files = self._files()
            good = self._package(root, files, self._manifest(self._entries(files)))
            forgotten = sorted(files)[0]
            bad = self._package(root, files,
                                self._manifest(self._entries(files, drop=[forgotten])),
                                name="撒谎.zip")
            plain = io.StringIO()
            with contextlib.redirect_stdout(plain):
                without = deliveryaudit_main(["--out", str(out)])
            checked = io.StringIO()
            with contextlib.redirect_stdout(checked):
                clean_code = deliveryaudit_main(["--out", str(out), "--zip", str(good)])
            bitten = io.StringIO()
            with contextlib.redirect_stdout(bitten):
                tampered = deliveryaudit_main(["--out", str(out), "--zip", str(bad)])
        self.assertEqual(without, 0)
        self.assertEqual(clean_code, 0)
        self.assertEqual(tampered, 1)
        self.assertIn("包里有而清单没记", bitten.getvalue())
        self.assertIn(forgotten, bitten.getvalue())
        self.assertNotIn("清单没记", checked.getvalue())


class ExportPackageTests(unittest.TestCase):
    """打包侧：只有本轮 export() 亲手写出的文件才是交付物，遗留文件既不带、又要点名。"""

    @staticmethod
    def _workspace(root):
        out = root / "out" / "user"
        (out / "T1").mkdir(parents=True)
        (out / "review").mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({
            "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
            "documents": [{"document_id": "T1", "doc_type": "BANK_STATEMENT",
                           "path": "T1/T1.json", "review_count": 0}],
        }), encoding="utf-8")
        (out / "T1" / "T1.json").write_text(json.dumps({
            "document_id": "T1", "doc_type": "BANK_STATEMENT",
            "fields": {"bs.account_name": {
                "path": "bs.account_name", "value": "示例公司", "value_type": "string",
                "decision": "accept", "confidence": 0.99,
                "grounding": {"page": 1, "bbox": [1, 2, 3, 4]}}},
        }, ensure_ascii=False), encoding="utf-8")
        (out / "review" / "T1.review.jsonl").write_text("", encoding="utf-8")
        config = copy.deepcopy(load_config())
        config["paths"]["work"] = str(root / "work")
        return out, config

    def test_expected_members_lists_exactly_what_this_run_exported(self):
        members = expected_package_members({"documents": [{"document_id": "T1"},
                                                         {"document_id": "T2"}]})
        self.assertEqual(members, ["final/T1/T1.json", "final/T2/T2.json",
                                   "final/final_summary.json", "final/final_fields.csv"])
        self.assertEqual(expected_package_members({}), ["final/final_summary.json",
                                                        "final/final_fields.csv"])

    def test_a_clean_run_lists_every_member_with_a_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, config = self._workspace(root)
            result = export_zip(out, config=config)
            opts = delivery_options(config)
            report = delivery_package_report(Path(result["path"]), opts)
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
        # 成员少不等于坏包：单文档夹具本来就装不到出厂下限，这里只审一致性。
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unlisted"], [])
        self.assertEqual(report["phantom"], [])
        self.assertEqual(report["bad_digest"], [])
        self.assertEqual(report["skipped"], [])
        self.assertTrue(report["thin"])
        self.assertEqual(report["member_count"], report["declared_count"])
        self.assertEqual({item["name"] for item in result["members"]} | {PACKAGE_MANIFEST_NAME},
                         names)

    def test_leftovers_and_foreign_source_files_are_refused_by_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out, config = self._workspace(root)
            export_zip(out, config=config)  # 先跑一轮，让 final/ 落到盘上
            final = out / "final"
            (final / "上一轮的调试表.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (final / "原始材料.pdf").write_bytes(b"%PDF-1.4 planted\n%%EOF\n")
            (out / "review" / "ZZ99.review.jsonl").write_text('{"x":1}\n', encoding="utf-8")
            result = export_zip(out, config=config)
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
            reasons = {item["name"]: item["reason"] for item in result["skipped_members"]}
            report = delivery_package_report(Path(result["path"]),
                                             dict(delivery_options(config),
                                                  min_package_members=0))
        self.assertNotIn("final/原始材料.pdf", names)
        self.assertNotIn("final/上一轮的调试表.csv", names)
        self.assertNotIn("review/ZZ99.review.jsonl", names)
        self.assertEqual(reasons, {"final/上一轮的调试表.csv": "not_exported_this_run",
                                   "final/原始材料.pdf": "not_exported_this_run",
                                   "review/ZZ99.review.jsonl": "document_not_in_export"})
        self.assertEqual(sorted(item["name"] for item in report["skipped"]), sorted(reasons))
        self.assertEqual(report["unlisted"], [])
        self.assertEqual(report["phantom"], [])


from docpipe.deliveryaudit import reconcile as delivery_reconcile


class DeliveryReconcileTests(unittest.TestCase):
    """铁律审计 X：`review/*.review.jsonl` 与 `final/` 必须逐格对得上。

    「人签的字有没有真的落到成品里」「成品宣称的已复核查不查得到是谁签的」以前只由
    `export()` 的取值规则保证，成品里看不见。这里把「对得上」拆成能各自失败的几条，
    每条反向用例只种一个破口，于是一条 FAIL 只指向一个责任判决。
    """

    AMOUNT = "bs.transactions[9].amount"
    PERSON = "bl.legal_representative"

    @classmethod
    def _cells(cls):
        return {
            "bl.company_name": {"value": "示例信息技术有限公司",
                                "effective_value": "示例信息技术有限公司",
                                "decision": "accept", "confidence": 0.9,
                                "review_reasons": [], "grounding": {"page": 1}},
            cls.AMOUNT: {"value": "-47796.99", "effective_value": "-47797.00",
                         "decision": "review", "confidence": 0.4,
                         "review_reasons": ["LOW_OCR"],
                         "review_status": "reviewed_corrected",
                         "grounding": {"page": 1}},
            cls.PERSON: {"value": "张三", "effective_value": "张三",
                         "decision": "review", "confidence": 0.5,
                         "review_reasons": ["TEMPLATE_MISMATCH"],
                         "review_status": "reviewed_accept", "grounding": {"page": 1}},
        }

    @classmethod
    def _record(cls, path, action, corrected=None, **overrides):
        record = {"document_id": "D001", "field_path": path,
                  "original_value": cls._cells().get(path, {}).get("value"),
                  "corrected_value": corrected, "action": action, "reviewer": "李四",
                  "timestamp": "2026-10-28T09:00:00+08:00"}
        record.update(overrides)
        return record

    @classmethod
    def _records(cls):
        return [cls._record(cls.AMOUNT, "correct", "-47797.00"),
                cls._record(cls.PERSON, "accept")]

    @classmethod
    def _trail(cls, root, cells=None, records=None, declared=None, name="D001"):
        """往一棵树里补上「交付成品 + 复核台账」：`final/` 与 `review/` 逐格配一遍的两侧。"""
        root = Path(root)
        cells = cls._cells() if cells is None else cells
        records = cls._records() if records is None else records
        statuses = [field["review_status"] for field in cells.values()
                    if field.get("review_status")]
        counts = declared if declared is not None else {
            "reviewed_count": len(statuses),
            "reviewed_corrected_count": sum(1 for s in statuses if s == "reviewed_corrected")}
        doc_dir = root / "final" / name
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / f"{name}.json").write_text(json.dumps({"document_id": name, "fields": cells},
                                                         ensure_ascii=False), encoding="utf-8")
        (root / "final" / "final_summary.json").write_text(
            json.dumps({"documents": [{"document_id": name, **counts}], "overall": dict(counts)},
                       ensure_ascii=False), encoding="utf-8")
        review_dir = root / "review"
        review_dir.mkdir(exist_ok=True)
        (review_dir / f"{name}.review.jsonl").write_text(
            "".join(line if isinstance(line, str) and line.endswith("\n")
                    else (line if isinstance(line, str)
                          else json.dumps(line, ensure_ascii=False)) + "\n"
                    for line in records), encoding="utf-8")
        return root

    @classmethod
    def _workspace(cls, temp, **kwargs):
        root = Path(temp) / "ws"
        root.mkdir(parents=True, exist_ok=True)
        return cls._trail(root, **kwargs)

    @staticmethod
    def _problems(result):
        return [item["problem"] for item in result["problems"]]

    def test_a_clean_trail_reconciles_and_is_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            result = delivery_reconcile(self._workspace(temp), dict(DELIVERY_DEFAULTS))
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["cells"], {"reviewed_count": 2, "reviewed_corrected_count": 1})
        self.assertFalse(result["no_trail"])

    def test_a_status_with_no_ledger_record_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, records=[self._record(self.PERSON, "accept")])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["unbacked_status"])
        self.assertIn(self.AMOUNT, str(result["problems"]))

    def test_a_record_that_matches_no_cell_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, records=self._records()
                                   + [self._record("bl.made_up_field", "accept")])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["record_without_cell"])
        self.assertIn("字段不在成品里", str(result["problems"]))

    def test_a_correction_that_never_landed_is_named(self):
        cells = self._cells()
        cells[self.AMOUNT]["effective_value"] = cells[self.AMOUNT]["value"]
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, cells=cells)
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["unapplied_correction"])

    def test_the_last_ledger_record_is_the_one_that_must_land(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, records=self._records()
                                   + [self._record(self.AMOUNT, "correct", "-48000.00")])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["stale_winner"])
        self.assertIn("更早一条记录", str(result["problems"]))

    def test_an_accept_that_moves_the_value_is_named(self):
        cells = self._cells()
        cells[self.PERSON]["effective_value"] = "李四"
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, cells=cells)
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["value_changed_without_correction"])

    def test_a_status_outside_the_vocabulary_is_named(self):
        cells = self._cells()
        cells[self.AMOUNT]["review_status"] = "reviewed_ok"
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, cells=cells)
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["unknown_state"])
        self.assertEqual(result["mismatches"], [])

    def test_status_and_action_must_pair_up(self):
        cells = self._cells()
        cells[self.AMOUNT]["review_status"] = "reviewed_accept"
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, cells=cells)
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(self._problems(result), ["status_action_mismatch"])

    def test_a_broken_line_and_an_incomplete_record_are_both_named(self):
        with tempfile.TemporaryDirectory() as temp:
            # 两条合法记录仍然在场：坏行与缺键记录只该被点名，不该把配好对的格子拖下水。
            root = self._workspace(temp, records=self._records()
                                   + ['这不是 JSON\n',
                                      {"field_path": self.AMOUNT, "action": "correct"}])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(sorted(self._problems(result)),
                         sorted(["missing_record_keys", "malformed_line"]))
        self.assertIn("这不是 JSON", str(result["problems"]))

    def test_an_incomplete_record_backs_no_status(self):
        with tempfile.TemporaryDirectory() as temp:
            # 台账里写了这一格，但少了 `reviewer`/`timestamp`/`original_value`：签字查不到是谁。
            root = self._workspace(temp, records=[{"field_path": self.AMOUNT, "action": "correct",
                                                   "corrected_value": "-47797.00"},
                                                  self._record(self.PERSON, "accept")])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(sorted(self._problems(result)),
                         sorted(["missing_record_keys", "unbacked_status"]))
        self.assertIn(self.AMOUNT, str(result["problems"]))

    def test_the_review_vocabulary_follows_the_config(self):
        cells = self._cells()
        pairs = {"accept": "signed", "correct": "reviewed_corrected"}
        base = Path(tempfile.mkdtemp())
        try:
            untuned = delivery_reconcile(self._workspace(base / "a"), dict(DELIVERY_DEFAULTS))
            renamed = self._workspace(base / "b", cells={**cells, self.PERSON: {
                **cells[self.PERSON], "review_status": "signed"}})
            retuned = delivery_reconcile(renamed, dict(DELIVERY_DEFAULTS, review_state_pairs=pairs))
            # 同一份成品，词汇表一换，出厂的 `reviewed_accept` 就成了越表状态。
            tuned_old = delivery_reconcile(self._workspace(base / "c"),
                                           dict(DELIVERY_DEFAULTS, review_state_pairs=pairs))
        finally:
            shutil.rmtree(base, ignore_errors=True)
        self.assertEqual(untuned["problems"], [])
        self.assertEqual(retuned["problems"], [])
        self.assertEqual(self._problems(tuned_old), ["unknown_state"])

    def test_which_actions_may_move_the_value_follows_the_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp)
            cleared = delivery_reconcile(root, dict(DELIVERY_DEFAULTS, value_changing_actions=[]))
        self.assertEqual(self._problems(cleared), ["value_changed_without_correction"])

    def test_a_tree_with_no_delivery_side_is_nothing_to_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "ws").mkdir()
            result = delivery_reconcile(Path(temp) / "ws", dict(DELIVERY_DEFAULTS))
        # 汇总侧查无此键时不能报「对不上」——没有 `final/` 就没有声明可验，这一问没内容。
        self.assertEqual(result["documents"], 0)
        self.assertEqual(result["mismatches"], [])
        self.assertTrue(result["no_trail"])

    def test_no_trail_at_all_is_reported_as_nothing_to_audit(self):
        cells = {path: {key: value for key, value in field.items() if key != "review_status"}
                 for path, field in self._cells().items()}
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, cells=cells, records=[])
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(result["problems"], [])
        self.assertTrue(result["no_trail"])
        self.assertEqual(result["record_count"], 0)

    def test_a_review_side_count_split_is_named_without_naming_a_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._workspace(temp, declared={"reviewed_count": 5,
                                                   "reviewed_corrected_count": 1})
            result = delivery_reconcile(root, dict(DELIVERY_DEFAULTS))
        self.assertEqual(result["problems"], [])
        self.assertTrue(result["mismatches"])
        self.assertEqual({item["problem"] for item in result["mismatches"]},
                         {"reviewed_count"})
        self.assertIn("final_summary", str(result["mismatches"]))

    def test_the_cli_reports_the_reconciliation_and_exits_nonzero_on_a_break(self):
        with tempfile.TemporaryDirectory() as temp:
            # 一棵完整的树：`audit()` 的覆盖下限要满足，第五问才有机会被问到。
            root = DeliveryAuditTests._delivery(Path(temp) / "ws", {
                f"D00{i}": [(f"bl.f{j}", "accept") for j in range(200)] for i in range(1, 6)})
            self._trail(root)
            clean = io.StringIO()
            with contextlib.redirect_stdout(clean):
                clean_code = deliveryaudit_main(["--out", str(root)])
            self._trail(root, records=[])
            bitten = io.StringIO()
            with contextlib.redirect_stdout(bitten):
                bitten_code = deliveryaudit_main(["--out", str(root)])
        self.assertEqual(clean_code, 0)
        self.assertIn("人工修正对账", clean.getvalue())
        self.assertNotIn("没有内容可审", clean.getvalue())
        self.assertEqual(bitten_code, 1)
        self.assertIn("台账与成品对不上", bitten.getvalue())
        self.assertIn(self.AMOUNT, bitten.getvalue())


from docpipe.deliveryaudit import coordinates as delivery_coordinates


class DeliveryCoordinatesTests(unittest.TestCase):
    """铁律审计 XI：交付的每一格坐标，人在前台那两层真的画得出来。

    `audit()` 查 `grounding` 这个键在不在，验收 `output.accepted_grounding` 查原始框有四元、
    面积为正；可复核台和单文件网页版摆方框用的是 `bbox_norm`（`extract._grounding` 在页面宽高
    缺失时把它写成 `None`，界面的 `grounding.bbox_norm` 过滤器就静默跳过那一格）。于是「坐标
    齐全」与「人看得见坐标」之间没有任何判据。每个用例只种一个破口，于是一条 FAIL 只指向一处。
    """

    PAGE = (595.0, 842.0)

    @classmethod
    def _field(cls, value="编号", decision="accept", page=1, box=(30.0, 60.0, 120.0, 78.0),
               norm=None):
        """按页面尺度 595×842 写一套自洽的坐标：`bbox_norm` 默认就是同一个尺度除出来的。"""
        width, height = cls.PAGE
        if norm is None:
            norm = [round(box[0] / width, 4), round(box[1] / height, 4),
                    round(box[2] / width, 4), round(box[3] / height, 4)]
        return {"value": value, "decision": decision, "confidence": 0.9, "review_reasons": [],
                "grounding": {"page": page, "bbox": list(box), "bbox_norm": list(norm)}}

    @staticmethod
    def _tree(temp, cells, page_count=1):
        root = Path(temp) / "out"
        folder = root / "D001"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"document_id": "D001", "fields": cells}
        if page_count is not None:
            payload["source"] = {"page_count": page_count}
        (folder / "D001.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return root

    def _run(self, temp, cells, page_count=1, **opts):
        root = self._tree(temp, cells, page_count)
        return delivery_coordinates(root, dict(DELIVERY_DEFAULTS, **opts))

    def test_a_self_consistent_pair_is_drawable_and_counted(self):
        cells = {"bl.a": self._field(), "bl.b": self._field("金额", decision="review")}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["offenders"], [])
        self.assertEqual(result["verdicts"], {})
        self.assertEqual(result["counts"], {"cells": 2, "audited": 2, "accepted": 1})
        self.assertEqual(result["drawn_keys"], ["bbox_norm"])
        self.assertEqual(result["raw_key"], "bbox")

    def test_a_value_with_no_grounding_at_all_is_named(self):
        cells = {"bl.a": self._field()}
        del cells["bl.a"]["grounding"]
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        # 没有 grounding 就连「哪一页」都不知道，不再叠加报 norm 缺失。
        self.assertEqual(result["verdicts"], {"no_grounding": 1})
        self.assertEqual(result["counts"], {"cells": 1, "audited": 0, "accepted": 1})
        self.assertEqual(result["offenders"][0]["problem"], "no_grounding")

    def test_a_frame_the_report_never_normalized_is_named(self):
        cells = {"bl.a": self._field()}
        cells["bl.a"]["grounding"]["bbox_norm"] = None
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"norm_absent": 1})
        self.assertIn("null", result["offenders"][0]["actual"])

    def test_a_norm_that_is_not_four_numbers_is_named(self):
        cells = {"bl.a": self._field(norm=[0.05, 0.07, 0.15]), "bl.b": self._field(norm="30,60")}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"norm_shape": 2})
        self.assertEqual({item["path"] for item in result["offenders"]}, {"bl.a", "bl.b"})

    def test_a_norm_that_lands_off_the_page_is_named(self):
        # 页面宽度被当成 0 除过、或按别的页幅钳制：框画到页面外面，前台仍然一句话不说。
        # 原始框贴着页面边缘，所以两套坐标那条推不出结论——这一格只报「画到页面外」。
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, {"bl.a": self._field(box=(0.0, 0.0, 1420.0, 500.0),
                                                          norm=[0.0, 0.0, 1.42, 0.5])})
        self.assertEqual(result["verdicts"], {"norm_out_of_bounds": 1})

    def test_a_frame_with_no_area_is_named(self):
        box = (30.0, 60.0, 30.0, 78.0)
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, {"bl.a": self._field(box=box)})
        # 两套坐标仍然同尺度（点框也是同一个尺度除出来的），所以只报「点不到」这一条。
        self.assertEqual(result["verdicts"], {"norm_degenerate": 1})

    def test_a_page_number_outside_the_document_is_named(self):
        cells = {"bl.a": self._field(page=3), "bl.b": self._field(page=0),
                 "bl.c": self._field(page=2)}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells, page_count=2)
        self.assertEqual(result["verdicts"], {"page_out_of_range": 2})
        self.assertEqual({item["path"] for item in result["offenders"]}, {"bl.a", "bl.b"})
        self.assertIn("1..2", str(result["offenders"]))

    def test_a_norm_recomputed_on_another_page_scale_is_named(self):
        # 改了 `bbox` 没重算 `bbox_norm`（或反过来）：纵边各自推出的尺度再也碰不到一起。
        cells = {"bl.a": self._field(box=(100.0, 100.0, 300.0, 200.0),
                                     norm=[0.1681, 0.5, 0.5042, 0.6])}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"scale_conflict": 1})
        self.assertIn("bbox", result["offenders"][0]["actual"])
        self.assertIn("bbox_norm", result["offenders"][0]["actual"])

    def test_a_frame_touching_the_page_edge_is_not_called_a_conflict(self):
        cells = {"bl.a": self._field(box=(0.0, 0.0, 595.0, 842.0), norm=[0.0, 0.0, 1.0, 1.0]),
                 "bl.b": self._field(box=(0.0, 60.0, 120.0, 78.0))}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        # 贴 0 / 贴页面边缘时推不出唯一尺度——看不见证据不等于有矛盾，不能整批假报警。
        self.assertEqual(result["offenders"], [])

    def test_a_missing_page_count_draws_no_conclusion_about_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, {"bl.a": self._field(page=7)}, page_count=None)
        self.assertEqual(result["offenders"], [])

    def test_a_cell_with_no_value_is_out_of_this_question(self):
        cells = {"bl.a": self._field(""), "bl.b": self._field(None), "bl.c": self._field([]),
                 "bl.d": self._field("有值")}
        for key in ("bl.a", "bl.b", "bl.c"):
            cells[key]["grounding"] = {}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        # 前台没有东西可画，看不见证据这条不成立；空格子归 VIII 的判决痕迹那一问管。
        self.assertEqual(result["counts"], {"cells": 1, "audited": 1, "accepted": 1})
        self.assertEqual(result["offenders"], [])

    def test_which_coordinates_get_drawn_follows_the_config(self):
        cells = {"bl.a": self._field(), "bl.b": self._field()}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells, drawn_coordinate_keys=["bbox_display"])
        self.assertEqual(result["verdicts"], {"norm_absent": 2})
        self.assertEqual(result["drawn_keys"], ["bbox_display"])

    def test_the_coordinate_tolerance_is_the_one_from_the_config(self):
        cells = {"bl.a": self._field(), "bl.b": self._field("金额")}
        with tempfile.TemporaryDirectory() as temp:
            # 出厂那半个刻度是「四舍五入写进产物」的唯一误差来源：抽干它，自洽的格子也会全红。
            strict = self._run(temp, cells, coordinate_tolerance=0.0)
            loose = self._run(temp, cells)
        self.assertEqual(strict["verdicts"], {"scale_conflict": 2})
        self.assertEqual(loose["offenders"], [])
        self.assertEqual(loose["tolerance"], DELIVERY_DEFAULTS["coordinate_tolerance"])

    def test_the_cli_bites_when_a_frame_cannot_be_drawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = DeliveryAuditTests._delivery(Path(temp) / "ws", {
                f"D00{i}": [(f"bl.f{j}", "accept") for j in range(200)] for i in range(1, 6)})
            payload = json.loads((root / "D001" / "D001.json").read_text(encoding="utf-8"))
            payload["fields"]["bl.f0"]["grounding"]["bbox_norm"] = None
            (root / "D001" / "D001.json").write_text(json.dumps(payload, ensure_ascii=False),
                                                     encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = deliveryaudit_main(["--out", str(root)])
        self.assertEqual(code, 1)
        self.assertIn("坐标可核性", out.getvalue())
        self.assertIn("画不出来或两套坐标打架: 1", out.getvalue())
        self.assertIn("norm_absent", out.getvalue())

    def test_a_delivery_with_no_values_says_so_instead_of_passing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self._tree(temp, {"bl.a": self._field("")})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                deliveryaudit_main(["--out", str(root)])
        self.assertIn("这一问没有内容可审", out.getvalue())


from docpipe.deliveryaudit import gates as delivery_gates


class DeliveryGateTests(unittest.TestCase):
    """铁律审计 XII：每一格「自动接受」，出厂配置里的门槛解释得通吗。

    判决层拿 `confidence` 跟 `thresholds` 的两个数比过才写 `accept`，可产物里只剩
    `confidence` 与 `decision` 两个字段，中间没有人再对一遍。这里独立查配置（不叫
    `config.risk_threshold`），每个用例只种一个破口——于是一条 FAIL 只指向一处。
    """

    LOW = "optional_low_risk"
    MONEY = "monetary"

    def setUp(self):
        self.config = load_config()
        self.floor = float(self.config["thresholds"]["global_min_confidence"])
        self.tiers = {key: float(value) for key, value in
                      self.config["thresholds"]["risk_tiers"].items()}

    @classmethod
    def _field(cls, value="编号", decision="accept", confidence=0.95, risk=LOW, path=None,
               reasons=None):
        return {"value": value, "path": path if path is not None else "bl.field",
                "risk": risk, "decision": decision, "confidence": confidence,
                "review_reasons": list(reasons or [])}

    @staticmethod
    def _tree(temp, cells):
        root = Path(temp) / "out"
        folder = root / "D001"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "D001.json").write_text(
            json.dumps({"document_id": "D001", "source": {"page_count": 1}, "fields": cells},
                       ensure_ascii=False), encoding="utf-8")
        return root

    def _run(self, temp, cells, config=None, **opts):
        root = self._tree(temp, cells)
        return delivery_gates(root, dict(DELIVERY_DEFAULTS, **opts), config or self.config)

    def test_a_gate_supported_release_is_clean_and_review_is_exempt(self):
        cells = {"bl.field": self._field(confidence=self.tiers[self.LOW] + 0.01),
                 "bl.low": self._field("金额", decision="review", confidence=self.floor - 0.1,
                                       reasons=["LOW_CONFIDENCE"])}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["offenders"], [])
        self.assertEqual(result["states"], ["accept"])
        self.assertEqual(result["global_min"], self.floor)
        self.assertEqual(result["no_gate"], False)
        # 只有 `gate_states` 里的状态有解释义务：review 那一格连数都不参加核对。
        self.assertEqual(result["counts"], {"cells": 1, "gated": 1})

    def test_the_gate_numbers_come_from_the_shipped_config(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, {"bl.field": self._field()})
        self.assertEqual(result["tiers"], self.tiers)
        self.assertEqual(sorted(result["tiers"]), sorted(self.config["thresholds"]["risk_tiers"]))
        self.assertEqual(result["overrides"], [])

    def test_a_release_below_its_own_risk_tier_is_named(self):
        # 高于全局下限、低于 `monetary` 档：只有档这一条能红。
        cells = {"bl.field": self._field("1,000.00", risk=self.MONEY,
                                        confidence=self.tiers[self.MONEY] - 0.02)}
        self.assertGreater(cells["bl.field"]["confidence"], self.floor)
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"below_risk_tier": 1})
        self.assertIn("monetary", str(result["offenders"]))

    def test_a_release_below_the_global_floor_is_named_once(self):
        cells = {"bl.field": self._field("1,000.00", risk=self.MONEY,
                                        confidence=self.floor - 0.4)}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        # 全局下限先挡住，就不该再叠一条档位违规：一条格子只该有一句话。
        self.assertEqual(result["verdicts"], {"below_global_min": 1})

    def test_a_risk_class_the_config_never_got_a_gate_is_named(self):
        cells = {"bl.field": self._field(risk="secret_sauce")}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"unknown_risk_class": 1})
        self.assertIn("secret_sauce", str(result["offenders"]))

    def test_a_release_without_any_risk_is_not_guessed_into_the_low_tier(self):
        cells = {"bl.field": self._field()}
        del cells["bl.field"]["risk"]
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        # 判决层在 `risk` 缺失时退回 `optional_low_risk`；这里刻意不跟：按一个没人声明过的档
        # 放行，正是「放行没有出处」的那种事。
        self.assertEqual(result["verdicts"], {"unknown_risk_class": 1})

    def test_a_release_that_still_carries_review_reasons_is_named(self):
        cells = {"bl.field": self._field(reasons=["LOW_CONFIDENCE"])}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"accept_with_review_reasons": 1})
        self.assertIn("LOW_CONFIDENCE", str(result["offenders"]))

    def test_a_cell_key_that_disagrees_with_its_own_path_is_named(self):
        cells = {"bl.key": self._field(path="bl.somebody_else")}
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"path_identity_split": 1})
        self.assertIn("bl.key", str(result["offenders"]))

    def test_a_confidence_that_is_not_a_number_cannot_be_explained(self):
        cells = {"bl.a": self._field(confidence="0.95", path="bl.a"),
                 "bl.b": self._field("金额", confidence=True, path="bl.b"),
                 "bl.c": self._field("日期", confidence=self.tiers[self.LOW] + 0.01, path="bl.c")}
        # 两格拿不出能比的数（字符串、布尔），顺手把它们的 `risk` 也拿掉：两件事各报各的。
        for key in ("bl.a", "bl.b"):
            del cells[key]["risk"]
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, cells)
        self.assertEqual(result["verdicts"], {"confidence_absent": 2, "unknown_risk_class": 2})
        self.assertEqual(result["counts"], {"cells": 3, "gated": 3})
        self.assertEqual({item["problem"] for item in result["offenders"]},
                         {"confidence_absent", "unknown_risk_class"})

    def test_which_states_must_explain_follows_the_config(self):
        cells = {"bl.field": self._field(),
                 "bl.low": self._field("金额", decision="review", confidence=self.floor - 0.1,
                                       path="bl.low")}
        with tempfile.TemporaryDirectory() as temp:
            flipped = self._run(temp, cells, gate_states=["review"])
        # 词汇表一换，出厂那些放行格子立刻没有解释义务，而低置信度的复核格子反过来全红。
        # 期望里只有 `below_global_min` 一条，这一句同时也在证明两张表是分开的：那一格复核
        # 格子带着 `LOW_CONFIDENCE` 的理由，把「必须闭嘴」跟「有解释义务」合成一个键时，
        # 它会被顺手判成 `accept_with_review_reasons`——第一版就是这样。
        self.assertEqual(flipped["verdicts"], {"below_global_min": 1})
        self.assertEqual(flipped["states"], ["review"])

    def test_which_states_must_stay_silent_follows_the_config(self):
        cells = {"bl.field": self._field(reasons=["LOW_CONFIDENCE"]),
                 "bl.low": self._field("金额", decision="review", confidence=self.floor - 0.1,
                                       reasons=["SIGN_INFERRED"], path="bl.low")}
        with tempfile.TemporaryDirectory() as temp:
            # 两张表都出厂时，露头的是那一格「说放行却挂着理由」的；复核格子天生带理由，不算。
            shipped = self._run(temp, cells)
            # 静音表换成 `review`、解释表照旧：露头的立刻换成复核那一格，放行那格反而不再
            # 因为挂理由被点名——它的数字本身跟 `thresholds` 仍然对得上，解释那条照旧查。
            muted = self._run(temp, cells, reason_free_states=["review"])
        self.assertEqual(shipped["verdicts"], {"accept_with_review_reasons": 1})
        self.assertEqual(shipped["offenders"][0]["path"], "bl.field")
        self.assertEqual(muted["verdicts"], {"accept_with_review_reasons": 1})
        self.assertEqual(muted["offenders"][0]["path"], "bl.low")

    def test_a_path_override_replaces_the_tier_and_folds_row_indexes(self):
        cells = {"bs.transactions[3].amount": self._field("1,000.00", risk=self.MONEY,
                                                          confidence=self.floor + 0.02,
                                                          path="bs.transactions[3].amount")}
        tuned = copy.deepcopy(self.config)
        tuned["thresholds"]["field_overrides"] = {"bs.transactions[i].amount": self.floor + 0.01}
        tightened = copy.deepcopy(self.config)
        tightened["thresholds"]["field_overrides"] = {"bs.transactions[3].amount": 0.99}
        with tempfile.TemporaryDirectory() as temp:
            # 出厂是空覆盖表，所以这一支只有夹具证明：`[i]` 形状的路径覆盖必须命中带下标的格子。
            loose = self._run(temp, cells, config=tuned)
            strict = self._run(temp, cells, config=tightened)
        self.assertEqual(loose["offenders"], [])
        self.assertEqual(loose["overrides"], ["bs.transactions[i].amount"])
        self.assertEqual(strict["verdicts"], {"below_risk_tier": 1})

    def test_an_empty_threshold_section_has_nothing_to_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self._run(temp, {"bl.field": self._field(confidence=0.0)},
                               config={"thresholds": {}})
        self.assertTrue(result["no_gate"])
        self.assertEqual(result["offenders"], [])
        self.assertEqual(result["counts"], {"cells": 0, "gated": 1})

    def test_the_cli_bites_when_a_release_has_no_gate_behind_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = DeliveryAuditTests._delivery(Path(temp) / "ws", {
                f"D00{i}": [(f"bl.f{j}", "accept") for j in range(200)] for i in range(1, 6)})
            clean = io.StringIO()
            with contextlib.redirect_stdout(clean):
                clean_code = deliveryaudit_main(["--out", str(root)])
            payload = json.loads((root / "D001" / "D001.json").read_text(encoding="utf-8"))
            payload["fields"]["bl.f0"]["confidence"] = 0.5
            (root / "D001" / "D001.json").write_text(json.dumps(payload, ensure_ascii=False),
                                                     encoding="utf-8")
            bitten = io.StringIO()
            with contextlib.redirect_stdout(bitten):
                bitten_code = deliveryaudit_main(["--out", str(root)])
        self.assertEqual(clean_code, 0)
        self.assertIn("门槛自证", clean.getvalue())
        self.assertIn("放行解释不通: 0", clean.getvalue())
        self.assertEqual(bitten_code, 1)
        self.assertIn("below_global_min", bitten.getvalue())
        self.assertIn("放行解释不通: 1", bitten.getvalue())

    def test_the_cli_says_the_gate_leg_has_no_content_when_thresholds_are_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = DeliveryAuditTests._delivery(Path(temp) / "ws", {
                f"D00{i}": [(f"bl.f{j}", "accept") for j in range(200)] for i in range(1, 6)})
            stripped = copy.deepcopy(load_config())
            stripped["thresholds"] = {}
            out = io.StringIO()
            with mock.patch("docpipe.config.load_config", return_value=stripped), \
                    contextlib.redirect_stdout(out):
                code = deliveryaudit_main(["--out", str(root)])
        # 弃权不是通过，也不是失败：它得说一句「这一问没内容」，退出码仍由别的那些问答。
        self.assertIn("thresholds 段一个数都没读到", out.getvalue())
        self.assertIn("放行解释不通: 0", out.getvalue())
        self.assertEqual(code, 0)


class ReviewLedgerWriteTests(unittest.TestCase):
    """铁律审计 XIII：复核台说「已保存」之前，那条修正必须真的能从台账读回来。

    台账是逐行追加的明文文件。上一行缺换行结尾时（有人手改过、或上一次写被截断），下一次追加
    会把两条记录**粘成一行**，而三个读取侧对坏行的处理是「跳过」或「记账」——于是两条人工修正
    同时消失，而界面当时已经说过「已保存」。这里钉住写入侧的三步（补齐换行、落盘、写完用导出层
    同一套读法确认）与「读不出来的行必须数得出来」这条口径。
    """

    PATHS = {"bl.registered_capital": "100.00", "bl.legal_representative": "张三"}

    @classmethod
    def _store(cls, root: Path):
        from docpipe.serve import ReviewStore
        out = root / "out"
        document = out / "D900"
        document.mkdir(parents=True)
        fields = {path: {"path": path, "risk": "monetary", "value": value,
                         "decision": "review", "review_reasons": ["LOW_OCR"],
                         "grounding": {"page": 1, "bbox": [50.0, 100.0, 200.0, 118.0]}}
                  for path, value in cls.PATHS.items()}
        (document / "D900.json").write_text(json.dumps(
            {"document_id": "D900", "doc_type": "business_license", "fields": fields},
            ensure_ascii=False), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps({"documents": [
            {"document_id": "D900", "path": "D900/D900.json"}]}), encoding="utf-8")
        (out / "review").mkdir()
        return ReviewStore(out, copy.deepcopy(load_config())), out

    @staticmethod
    def _save(store, path, value=None):
        return store.append_review("D900", {"field_path": path,
                                            "action": "correct" if value else "accept",
                                            "corrected_value": value})

    @staticmethod
    def _record(path, value=None):
        record = {"document_id": "D900", "field_path": path, "timestamp": "T0",
                  "action": "correct" if value else "accept"}
        if value:
            record["corrected_value"] = value
        return record

    def test_a_saved_correction_is_read_back_before_the_ui_may_claim_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            result = self._save(store, "bl.registered_capital", value="120.00")
            ledger = read_review_ledger(out, "D900")
        self.assertEqual(result["record"]["corrected_value"], "120.00")
        self.assertEqual(result["ledger_unread"], 0)
        self.assertEqual(ledger["records"]["bl.registered_capital"]["corrected_value"], "120.00")

    def test_a_line_without_a_newline_does_not_swallow_the_previous_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            ledger_path = out / "review" / "D900.review.jsonl"
            # 已经签好的一格，故意不带换行结尾：这就是「上一次写被截断」或「手改过台账」的样子。
            ledger_path.write_text(json.dumps(self._record("bl.registered_capital", "120.00"),
                                              ensure_ascii=False), encoding="utf-8")
            self._save(store, "bl.legal_representative", value="李四")
            ledger = read_review_ledger(out, "D900")
        self.assertEqual(ledger["unread"], [])
        self.assertEqual(sorted(ledger["records"]), sorted(self.PATHS))
        self.assertEqual(ledger["records"]["bl.registered_capital"]["corrected_value"], "120.00")

    def test_two_records_glued_into_one_line_are_counted_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            glued = (json.dumps(self._record("bl.registered_capital", "120.00"),
                                ensure_ascii=False)
                     + json.dumps(self._record("bl.legal_representative", "李四"),
                                  ensure_ascii=False))
            (out / "review" / "D900.review.jsonl").write_text(glued + "\n", encoding="utf-8")
            ledger = read_review_ledger(out, "D900")
            reviews, unread = store.reviews("D900"), store.ledger_unread("D900")
        self.assertEqual(ledger["records"], {})
        self.assertEqual(len(ledger["unread"]), 1)
        self.assertEqual(reviews, [])
        self.assertEqual(unread, 1)

    def test_a_json_line_that_is_not_a_record_never_reaches_the_browser(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            good = json.dumps(self._record("bl.registered_capital"), ensure_ascii=False)
            (out / "review" / "D900.review.jsonl").write_text(
                f"null\n\"裸字符串\"\n[1, 2]\n{good}\n", encoding="utf-8")
            reviews = store.reviews("D900")
            unread = store.ledger_unread("D900")
        self.assertEqual([item["field_path"] for item in reviews], ["bl.registered_capital"])
        self.assertEqual(unread, 3)

    def test_an_object_without_a_field_path_counts_as_unread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            (out / "review" / "D900.review.jsonl").write_text(
                json.dumps({"action": "accept", "timestamp": "T0"}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            ledger = read_review_ledger(out, "D900")
            self._save(store, "bl.registered_capital")
        self.assertEqual(ledger["records"], {})
        self.assertEqual(len(ledger["unread"]), 1)

    def test_a_write_that_cannot_be_read_back_is_an_error_not_a_saved_toast(self):
        empty = {"records": {}, "all": [], "unread": [], "lines": 0}
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self._store(Path(tmp))
            with mock.patch("docpipe.serve.read_review_ledger", return_value=empty):
                with self.assertRaises(RuntimeError) as caught:
                    self._save(store, "bl.registered_capital", value="120.00")
        self.assertIn("读不回来", str(caught.exception))
        self.assertIn("bl.registered_capital", str(caught.exception))

    def test_the_append_is_flushed_to_disk_before_saving_is_claimed(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self._store(Path(tmp))
            with mock.patch("docpipe.serve.os.fsync", side_effect=calls.append):
                self._save(store, "bl.registered_capital", value="120.00")
        self.assertEqual(len(calls), 1)

    def test_an_empty_ledger_gets_no_leading_blank_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, out = self._store(Path(tmp))
            (out / "review" / "D900.review.jsonl").write_text("", encoding="utf-8")
            self._save(store, "bl.registered_capital", value="120.00")
            text = (out / "review" / "D900.review.jsonl").read_text(encoding="utf-8")
            ledger = read_review_ledger(out, "D900")
        self.assertTrue(text.startswith("{"))
        self.assertEqual(ledger["lines"], 1)

    def test_the_review_server_keeps_only_one_ledger_parser(self):
        source = (ROOT / "docpipe" / "serve.py").read_text(encoding="utf-8")
        self.assertIn("read_review_ledger", source)
        self.assertNotIn("json.loads(line)", source)


class ReviewDriftTests(unittest.TestCase):
    """铁律审计 XIV：折进校准池的人工判决，必须对着它判定过的那一版产物。

    结果目录重跑过（换解析器、改阈值、重新 OCR）之后，字段会换值、也会换判决，可台账还是
    原来那一行。把那条 accept 当成「这一版特征是对的」学进校准池，学的是一条没人看过的对照。
    """

    CAPITAL = "bl.registered_capital"
    REPRESENTATIVE = "bl.legal_representative"

    @staticmethod
    def _record(path, action, stamp, original_value, auto_decision="review"):
        record = {"document_id": "D900", "field_path": path, "action": action,
                  "auto_decision": auto_decision, "timestamp": stamp,
                  "original_value": original_value}
        if action == "correct":
            record["corrected_value"] = "重录值"
        return record

    @staticmethod
    def _workspace(tmp, records, mutate=None):
        """夹具字段出厂是 value=100.00 / decision=review，mutate 模拟产物重跑。"""
        root = ReviewLoopTests._fixture(Path(tmp), records)
        if mutate:
            path = root / "D900" / "D900.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            mutate(document["fields"])
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return root

    @staticmethod
    def _config(**overrides):
        config = copy.deepcopy(load_config())
        config["calibration"].update(overrides)
        return config

    def _drift_tags(self, root, config=None):
        samples = calibrate.review_samples([root], config or self._config())
        return [item["drift"] for item in samples]

    def test_a_product_the_reviewer_actually_saw_is_not_drift(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00"),
                   self._record(self.REPRESENTATIVE, "correct", "2026-10-26T09:00:01", "100.00")]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(tmp, records)
            tags = self._drift_tags(root)
            samples = calibrate.review_samples([root], self._config())
        self.assertEqual(tags, ["", ""])
        self.assertEqual({item["path"]: item["y"] for item in samples},
                         {self.CAPITAL: 1, self.REPRESENTATIVE: 0})

    def test_a_rewritten_value_is_tagged_value_drift(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00")]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            tags = self._drift_tags(root)
        self.assertEqual(tags, ["value"])

    def test_a_flipped_decision_is_tagged_too_and_both_axes_add_up(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00",
                                auto_decision="accept"),
                   self._record(self.REPRESENTATIVE, "accept", "2026-10-26T09:00:01", "100.00")]
        with tempfile.TemporaryDirectory() as tmp:
            def rerun(fields):
                fields[self.CAPITAL]["decision"] = "review"
                fields[self.REPRESENTATIVE]["value"] = "200.00"
                fields[self.REPRESENTATIVE]["decision"] = "accept"
            root = self._workspace(tmp, records, mutate=rerun)
            tags = self._drift_tags(root)
        self.assertEqual(tags, ["value+decision", "decision"])

    def test_a_ledger_without_anchors_is_not_judged(self):
        """看不见证据不等于有矛盾：缺 original_value 与 auto_decision 的老台账就不判。"""
        records = [ReviewLoopTests._record(self.CAPITAL, "accept", "2026-10-26T09:00:00")]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            tags = self._drift_tags(root)
        self.assertEqual(tags, [""])

    def test_null_and_empty_value_anchors_are_the_same_text(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "")]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(tmp, records,
                                   lambda fields: fields[self.CAPITAL].update(value=None))
            tags = self._drift_tags(root)
        self.assertEqual(tags, [""])

    def test_shipped_policy_drops_drifted_samples_and_counts_each_axis(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00"),
                   self._record(self.REPRESENTATIVE, "accept", "2026-10-26T09:00:01", "100.00")]
        config = self._config(review_min_samples=1)
        with tempfile.TemporaryDirectory() as tmp:
            def rerun(fields):
                fields[self.CAPITAL]["value"] = "重跑后的新值"
                fields[self.REPRESENTATIVE]["decision"] = "accept"
            root = self._workspace(tmp, records, mutate=rerun)
            samples = calibrate.review_samples([root], config)
            merged, stats = calibrate.merge_review_samples([], samples, config)
        self.assertEqual(load_config()["calibration"]["review_drift"], "drop")
        self.assertEqual((stats["records"], stats["drift"], stats["drift_value"],
                          stats["drift_decision"], stats["drift_dropped"], stats["usable"]),
                         (2, 2, 1, 1, 2, 0))
        self.assertEqual(merged, [])

    def test_keep_policy_folds_them_anyway_but_still_reports_them(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00")]
        config = self._config(review_drift="keep", review_min_samples=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            samples = calibrate.review_samples([root], config)
            merged, stats = calibrate.merge_review_samples([], samples, config)
        self.assertEqual(stats["drift_policy"], "keep")
        self.assertEqual((stats["drift"], stats["drift_dropped"], stats["drift_kept"],
                          stats["usable"]), (1, 0, 1, 1))
        self.assertEqual(len(merged), 1)

    def test_an_unreadable_policy_fails_closed_to_drop(self):
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00")]
        config = self._config(review_drift="driip", review_min_samples=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            merged, stats = calibrate.merge_review_samples(
                [], calibrate.review_samples([root], config), config)
        self.assertEqual(stats["drift_policy"], "drop")
        self.assertEqual(merged, [])

    def test_the_code_fallback_is_the_shipped_value(self):
        """删掉配置键也要拿到同一个答案——configaudit 的逐字一致靠这条兜住语义。"""
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00")]
        config = self._config(review_min_samples=1)
        del config["calibration"]["review_drift"]
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            merged, stats = calibrate.merge_review_samples(
                [], calibrate.review_samples([root], config), config)
        self.assertEqual(stats["drift_policy"], "drop")
        self.assertEqual(merged, [])

    def test_a_drifted_record_shadowed_by_labels_is_not_counted_as_drift(self):
        """分母要说清是谁：被标注压住（shadowed）的漂移记录不再进池，就不该同时算进 drift。"""
        records = [self._record(self.CAPITAL, "accept", "2026-10-26T09:00:00", "100.00")]
        config = self._config(review_min_samples=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = self._workspace(
                tmp, records,
                lambda fields: fields[self.CAPITAL].update(value="重跑后的新值"))
            samples = calibrate.review_samples([root], config)
            labels = [{"document_id": "D900", "path": self.CAPITAL}]
            merged, stats = calibrate.merge_review_samples(labels, samples, config)
        self.assertEqual((stats["records"], stats["shadowed_by_labels"], stats["drift"],
                          stats["usable"]), (1, 1, 0, 0))
        self.assertEqual(merged, labels)


class DocTypeLabelTests(unittest.TestCase):
    """文档类型的中文名只许有一个主人：配置 `doc_types.*.label`。

    复核台原先自带一份 `DOC_LABELS`，已经与配置漂开（`CUSTOMS_VAT_PAYMENT` 少写了「专用」），
    而配置里没有的类型在单文件网页版上会直接印出英文码 `UNKNOWN`。现在两个出口都读
    `report.doc_type_labels`，界面那份常量只在载荷还没到位时兜底。
    """

    def _script(self):
        return (Path(__file__).resolve().parents[1] / "docpipe" / "web" / "app.js"
                ).read_text(encoding="utf-8")

    def _shipped_fallback(self):
        script = self._script()
        block = re.search(r"const SHIPPED_DOC_LABELS = \{(.*?)\};", script, re.S)
        self.assertIsNotNone(block)
        return dict(re.findall(r'([A-Z_]+): "([^"]*)"', block.group(1)))

    def _workspace(self, root, doc_type):
        out = root / "out"
        (out / "T1").mkdir(parents=True)
        (out / "review").mkdir()
        (out / "summary.json").write_text(json.dumps({
            "run": {"pipeline_version": "0.1.0", "started_at": "2026-10-24T12:00:00+08:00"},
            "documents": [{"document_id": "T1", "doc_type": doc_type,
                           "path": "T1/T1.json", "review_count": 0}],
        }), encoding="utf-8")
        (out / "T1" / "T1.json").write_text(json.dumps({
            "document_id": "T1", "doc_type": doc_type,
            "source": {"path": "", "file_name": "示例.pdf", "page_count": 1},
            "fields": {"party.name": {
                "path": "party.name", "label": "名称", "risk": "party_name",
                "value": "示例公司", "value_type": "string", "confidence": 0.99,
                "decision": "accept",
                "grounding": {"page": 1, "bbox": [1, 2, 3, 4],
                              "bbox_norm": [0.1, 0.1, 0.2, 0.2]},
            }},
        }, ensure_ascii=False), encoding="utf-8")
        return out

    def _report_payload(self, out, config):
        html = Path(build_report(out, config=config)["path"]).read_text(encoding="utf-8")
        payload = json.loads(re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S
            ).group(1).replace("<\\/", "</"))
        return payload, html

    def test_the_console_fallback_is_the_config_verbatim(self):
        from docpipe.report import doc_type_labels
        labels = doc_type_labels(load_config())
        self.assertEqual(self._shipped_fallback(), labels)
        # 历史上漂过的那一格单独钉一次：少一个「专用」就是两个出口各叫一个名字。
        self.assertEqual(labels["CUSTOMS_VAT_PAYMENT"], "海关进口增值税专用缴款书")

    def test_unknown_type_is_named_the_same_in_both_web_exits(self):
        from docpipe.report import UNKNOWN_DOC_LABEL, doc_type_label
        self.assertEqual(self._shipped_fallback()["UNKNOWN"], UNKNOWN_DOC_LABEL)
        self.assertEqual(doc_type_label("UNKNOWN", load_config()), UNKNOWN_DOC_LABEL)
        with tempfile.TemporaryDirectory() as temp:
            out = self._workspace(Path(temp), "UNKNOWN")
            payload, _ = self._report_payload(out, load_config())
        self.assertEqual(payload["documents"][0]["doc_type_label"], UNKNOWN_DOC_LABEL)
        self.assertEqual(payload["groups"][0]["label"], UNKNOWN_DOC_LABEL)

    def test_renaming_a_label_moves_both_web_exits(self):
        from docpipe.report import doc_type_labels
        from docpipe.serve import ReviewStore
        config = copy.deepcopy(load_config())
        config["doc_types"]["CUSTOMS_VAT_PAYMENT"]["label"] = "缴款书（改名后）"
        with tempfile.TemporaryDirectory() as temp:
            out = self._workspace(Path(temp), "CUSTOMS_VAT_PAYMENT")
            payload, html = self._report_payload(out, config)
            shipped = ReviewStore(out, config).workspace_payload()["doc_type_labels"]
        self.assertEqual(payload["documents"][0]["doc_type_label"], "缴款书（改名后）")
        self.assertEqual(shipped, doc_type_labels(config))
        self.assertNotIn("海关进口增值税专用缴款书", html)

    def test_a_type_without_a_label_falls_back_to_its_code(self):
        from docpipe.report import doc_type_labels
        config = copy.deepcopy(load_config())
        config["doc_types"]["NEW_TYPE"] = {"keywords": ["示例"]}
        self.assertEqual(doc_type_labels(config)["NEW_TYPE"], "NEW_TYPE")
        self.assertNotIn("", doc_type_labels(config).values())

    def test_the_console_reads_the_labels_from_the_payload(self):
        script = self._script()
        self.assertIn("payload.doc_type_labels", script)
        self.assertIn("state.docLabels || {}", script)
        self.assertIn("SHIPPED_DOC_LABELS[type]", script)

class NonCnyStatementTests(unittest.TestCase):
    """§4.4.1：第一版只自动接受 CNY，其他币种「全部金额复核」。

    只拦币种那一格时，美元流水的余额、合计、交易金额会自动接受并被当成人民币
    交付——币种是整份流水的判决输入，不是某一格自己的事。
    """

    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.config["calibration"]["artifact"] = None
        self.parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                       "pages": [{"page": 1, "width": 595, "height": 842,
                                  "quality_flags": [], "raster": None}]}

    @staticmethod
    def _field(path: str, risk: str, value: str = "100.00") -> Dict:
        return {
            "path": path, "label": path, "value": value, "value_type": "decimal",
            "required": True, "risk": risk,
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0, "grounding_present": 1},
            "meta": {}, "decision": "review", "validators": [], "review_reasons": [],
        }

    @staticmethod
    def _doc(currency) -> Dict:
        return {"document_id": "FX1", "doc_type": "BANK_STATEMENT",
                "fields": {"bs.currency": {"path": "bs.currency", "value": currency,
                                           "risk": "enum_flag", "required": True}}}

    def _decide(self, field, doc):
        return decide_field(field, doc, self.parsed, self.config, [], Calibrator(self.config))

    def test_every_monetary_field_of_a_non_cny_statement_is_held(self):
        doc = self._doc("USD")
        for path in ("bs.opening_balance", "bs.closing_balance", "bs.total_income",
                     "bs.total_expense", "bs.transactions[3].amount",
                     "bs.transactions[3].balance"):
            decided = self._decide(self._field(path, "monetary"), doc)
            self.assertEqual(decided["decision"], "review", path)
            self.assertIn("NON_CNY_CURRENCY", decided["review_reasons"], path)

    def test_a_cny_statement_keeps_its_amounts(self):
        for currency in ("CNY", "人民币", "RMB"):
            doc = self._doc(currency)
            decided = self._decide(self._field("bs.closing_balance", "monetary"), doc)
            self.assertEqual(decided["decision"], "accept", currency)
            self.assertNotIn("NON_CNY_CURRENCY", decided["review_reasons"], currency)

    def test_unreadable_currency_is_not_assumed_cny(self):
        decided = self._decide(self._field("bs.closing_balance", "monetary"), self._doc(None))
        self.assertEqual(decided["decision"], "review")
        self.assertIn("NON_CNY_CURRENCY", decided["review_reasons"])

    def test_the_currency_cell_itself_still_gets_the_reason(self):
        decided = self._decide(self._field("bs.currency", "enum_flag", "美元"), self._doc("美元"))
        self.assertIn("NON_CNY_CURRENCY", decided["review_reasons"])

    def test_non_monetary_fields_are_not_dragged_in(self):
        decided = self._decide(self._field("bs.account_name", "party_name", "示例公司"),
                               self._doc("USD"))
        self.assertNotIn("NON_CNY_CURRENCY", decided["review_reasons"])

    def test_a_doc_object_without_a_field_table_is_left_alone(self):
        # 单格判决的夹具只给一个字段；看不见币种时这一条不参与，免得在
        # 「没有整份文档」的对象上凭猜测改判。
        bare = {"document_id": "FX2", "doc_type": "BANK_STATEMENT"}
        decided = self._decide(self._field("bs.closing_balance", "monetary"), bare)
        self.assertNotIn("NON_CNY_CURRENCY", decided["review_reasons"])

class SchemaPathTests(unittest.TestCase):
    """SCENARIO 第 102 行：已发布字段路径只许追加，改名/删除必须当场红。"""

    def test_the_shipped_config_matches_the_frozen_baseline(self):
        result = schemapaths.audit(load_config())
        self.assertEqual(result["problem"], "")
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["added"], [])
        self.assertGreater(len(result["shipped"]), 100)

    def test_a_renamed_path_is_named_with_both_ends(self):
        bitten = schemapaths.compare(["bl.new_name"], ["bl.old_name"])
        self.assertEqual(bitten["missing"], ["bl.old_name"])
        self.assertEqual(bitten["added"], ["bl.new_name"])

    def test_the_config_index_and_the_produced_index_are_one_name(self):
        self.assertEqual(
            schemapaths.compare(["bs.transactions[i].amount"],
                                ["bs.transactions[7].amount"]),
            {"missing": [], "added": []})

    def test_appending_is_legal_but_must_be_registered(self):
        bitten = schemapaths.compare(["bl.company_name", "bl.new_field"],
                                     ["bl.company_name"])
        self.assertEqual(bitten["missing"], [])
        self.assertEqual(bitten["added"], ["bl.new_field"])

    def test_a_missing_baseline_is_reported_not_silently_green(self):
        with tempfile.TemporaryDirectory() as temp:
            result = schemapaths.audit(load_config(), Path(temp) / "absent.json")
        self.assertIn("基线不存在", result["problem"])

    def test_the_cli_writes_the_baseline_and_then_passes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            target = Path(temp) / "schema_paths.json"
            self.assertEqual(schemapaths.main(["--baseline", str(target), "--write"]), 0)
            self.assertTrue(target.exists())
            self.assertEqual(schemapaths.main(["--baseline", str(target)]), 0)
            data = json.loads(target.read_text(encoding="utf-8"))
            data["paths"].remove("bl.company_name")
            target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(schemapaths.main(["--baseline", str(target)]), 1)

class HoldoutIsolationTests(unittest.TestCase):
    """SCENARIO 第 108 行：留出集标签只许评估脚本读，调参侧必须拒读。"""

    def setUp(self):
        self.config = load_config()

    def test_the_shipped_config_marks_the_holdout_split(self):
        self.assertIn("holdout", evaluation_only_splits(self.config))
        self.assertNotIn("dev", evaluation_only_splits(self.config))

    def test_fitting_on_the_holdout_split_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            calibrate.build_samples("holdout", self.config)
        self.assertIn("留出集", str(caught.exception))

    def test_the_refusal_follows_the_config_not_the_name(self):
        moved = copy.deepcopy(self.config)
        moved["paths"]["evaluation_only_splits"] = ["dev"]
        with self.assertRaises(SystemExit) as caught:
            calibrate.build_samples("dev", moved)
        self.assertIn("留出集", str(caught.exception))

    def test_an_unguarded_split_still_reaches_the_loader(self):
        loose = copy.deepcopy(self.config)
        loose["paths"]["evaluation_only_splits"] = []
        with mock.patch.object(calibrate, "load_split", return_value=([], {})) as patched:
            self.assertEqual(calibrate.build_samples("holdout", loose), [])
        patched.assert_called_once()

    def test_the_tuning_module_does_not_hardcode_the_split_name(self):
        source = (ROOT / "docpipe" / "calibrate.py").read_text(encoding="utf-8")
        self.assertNotIn('"holdout"', source)
        self.assertIn("evaluation_only_splits", source)

    def test_the_labels_root_is_read_from_the_config(self):
        source = (ROOT / "docpipe" / "calibrate.py").read_text(encoding="utf-8")
        self.assertNotIn('ROOT / "labels"', source)
        self.assertIn('config["paths"]["labels"]', source)


class SampleFreezeTests(unittest.TestCase):
    """SCENARIO 第 109 行：样本清单冻结后不得原地修改；需要修改就新建版本号。"""

    def test_the_shipped_samples_match_their_manifest(self):
        from docpipe import samplefreeze
        result = samplefreeze.audit(load_config())
        self.assertEqual(result["problems"], [])
        self.assertGreater(result["splits"]["dev"]["documents"], 0)

    def test_a_tampered_manifest_names_the_document(self):
        from docpipe import samplefreeze
        config = load_config()
        payload = json.loads(samplefreeze.manifest_path("dev", config).read_text(encoding="utf-8"))
        first = next(item for item in payload["documents"] if item.get("split") in (None, "dev"))
        for item in payload["documents"]:
            if item["document_id"] == first["document_id"]:
                item["sha256"] = "0" * 64
        bitten = samplefreeze.audit_split("dev", config, manifest=payload)["hash_mismatches"]
        self.assertEqual([row[0] for row in bitten], [first["document_id"]])

    def test_content_change_without_version_bump_is_refused(self):
        from docpipe import samplefreeze
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "freeze.json"
            current = samplefreeze.audit(load_config())["splits"]
            frozen = {k: {"version": v["version"], "digest": "0" * 64} for k, v in current.items()}
            samplefreeze.write_baseline(frozen, base)
            problems = samplefreeze.audit(load_config(), base)["problems"]
        self.assertTrue(any("内容变了但 version 没升" in item for item in problems))

    def test_a_version_bump_allows_a_new_digest(self):
        from docpipe import samplefreeze
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "freeze.json"
            frozen = {k: {"version": 0, "digest": "0" * 64}
                      for k in samplefreeze.audit(load_config())["splits"]}
            samplefreeze.write_baseline(frozen, base)
            problems = samplefreeze.audit(load_config(), base)["problems"]
        self.assertEqual(problems, [])

    def test_an_unlisted_file_is_named(self):
        from docpipe import samplefreeze
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "samples" / "dev").mkdir(parents=True)
            (root / "labels" / "dev").mkdir(parents=True)
            sample = root / "samples" / "dev" / "D001.pdf"
            label = root / "labels" / "dev" / "D001.json"
            sample.write_bytes(b"pdf")
            label.write_text("{}", encoding="utf-8")
            manifest = {"version": 1, "documents": [{
                "document_id": "D001", "split": "dev", "path": sample.as_posix(),
                "label_path": label.as_posix(),
                "sha256": hashlib.sha256(b"pdf").hexdigest()}]}
            (root / "samples" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "samples" / "dev" / "extra.pdf").write_bytes(b"x")
            config = {"paths": {"samples": str(root / "samples"), "labels": str(root / "labels")}}
            info = samplefreeze.audit_split("dev", config)
        self.assertTrue(any(name.endswith("extra.pdf") for name in info["unlisted"]), info["unlisted"])
class EvaluationProvenanceTests(unittest.TestCase):
    """SCENARIO §9.3/§9.4：评估报告要带代码版本、配置哈希、样本清单哈希，并写明合成集局限。

    这三样此前一个都没记：报告里只有指标，换一版代码或改一个阈值重跑，两份报告分不清
    谁是谁；三条合成集局限只活在 SCENARIO 里，报告上看不到，数字就容易被当成真实
    业务准确率。
    """

    def setUp(self):
        self.config = load_config()

    def test_the_shipped_evaluation_records_three_hashes(self):
        from docpipe import evaluate
        provenance = evaluate.report_provenance("dev", self.config)
        for key in ("code_sha256", "config_sha256", "sample_manifest_sha256"):
            self.assertRegex(provenance["provenance"][key], r"^[0-9a-f]{64}$")
        self.assertEqual(provenance["provenance"]["split"], "dev")

    def test_scope_only_flips_at_the_configured_floor(self):
        from docpipe import evaluate
        floor = self.config["evaluation"]["provenance"]["min_real_samples"]
        below = evaluate.report_provenance("dev", self.config, real_samples=floor - 1)
        at = evaluate.report_provenance("dev", self.config, real_samples=floor)
        self.assertEqual((below["claim_scope"], below["production_ready"]),
                         ("synthetic_only", False))
        self.assertEqual((at["claim_scope"], at["production_ready"]),
                         ("real_calibrated", True))
        moved = copy.deepcopy(self.config)
        moved["evaluation"]["provenance"]["min_real_samples"] = 5
        self.assertEqual(
            evaluate.report_provenance("dev", moved, real_samples=5)["claim_scope"],
            "real_calibrated")

    def test_the_limitations_are_read_from_the_config(self):
        from docpipe import evaluate
        moved = copy.deepcopy(self.config)
        moved["evaluation"]["provenance"]["limitations"] = ["哨兵局限"]
        self.assertEqual(evaluate.report_provenance("dev", moved)["limitations"], ["哨兵局限"])
        moved["evaluation"]["provenance"]["limitations"] = []
        self.assertEqual(evaluate.report_provenance("dev", moved)["limitations"], [])

    def test_any_threshold_change_moves_the_config_hash(self):
        from docpipe import evaluate
        moved = copy.deepcopy(self.config)
        self.assertEqual(evaluate.config_digest(moved), evaluate.config_digest(self.config))
        moved["acceptance"]["min_auto_accept_rate"] = 0.61
        self.assertNotEqual(evaluate.config_digest(moved), evaluate.config_digest(self.config))

    def test_the_written_report_carries_the_provenance(self):
        from docpipe import evaluate
        with tempfile.TemporaryDirectory() as temp:
            pred = Path(temp) / "out"
            labels = Path(temp) / "holdout"
            pred.mkdir()
            labels.mkdir()
            result = evaluate.evaluate_predictions(pred, labels, self.config)
        self.assertEqual(result["provenance"]["split"], "holdout")
        self.assertEqual(result["claim_scope"], "synthetic_only")
        self.assertFalse(result["production_ready"])
        self.assertTrue(result["limitations"])
class CrossDocumentContractTests(unittest.TestCase):
    """SCENARIO §5.4：跨文档一致性规则的词、字段与复核义务都来自配置里的规则表。

    流水户名与利润表编制单位以前不在 V-X-01 的比对范围里（代码只收了执照与征信两张名），
    而 V-X-02 的失败打在名称字段上、两个 USCC 自己反而能自动接受——两条都不是「测不到」，
    是判决落在了契约没点名的格子上。
    """

    USCC = "91310000MA1FL0AB3C"
    OTHER = "91110000MA01XY2Z8P"

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())
        self.config["calibration"]["artifact"] = None
        self.rules = trust.cross_document_rules(self.config)

    def _doc(self, doc_id, doc_type, values):
        return {"document_id": doc_id, "doc_type": doc_type,
                "fields": {path: {"path": path, "value": value} for path, value in values.items()}}

    def _validate(self, docs, config=None):
        validators = {doc["document_id"]: {} for doc in docs}
        self.trust.validate_cross_document(docs, config or self.config, validators)
        return {(doc_id, path): [f"{item['id']}/{item['status']}" for item in items]
                for doc_id, by_path in validators.items()
                for path, items in by_path.items()}

    def _field(self, path, value, risk="party_name"):
        return {"path": path, "label": path, "value": value, "value_type": "string",
                "required": True, "risk": risk,
                "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
                "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                            "table_score": 1.0, "cross_field_score": 1.0, "grounding_present": 1},
                "meta": {}, "decision": "review", "validators": [], "review_reasons": []}

    def _parsed(self):
        return {"page_count": 1, "quality": {"parse_score": 0.99},
                "pages": [{"page": 1, "width": 595, "height": 842, "quality_flags": [],
                           "raster": None}]}

    def test_the_statement_account_name_and_the_income_unit_are_compared(self):
        hits = self._validate([
            self._doc("X1", "BUSINESS_LICENSE", {"bl.company_name": "示例信息技术有限公司",
                                                 "bl.unified_social_credit_code": self.USCC}),
            self._doc("X2", "BANK_STATEMENT", {"bs.account_name": "示例信息技术公司"}),
            self._doc("X3", "INCOME_STATEMENT", {"is.company_name": "示例信息技术有限公司"})])
        self.assertEqual(sorted(path for _, path in hits),
                         ["bl.company_name", "bs.account_name", "is.company_name"])
        self.assertTrue(all(item == "V-X-01/warn" for values in hits.values() for item in values))

    def test_a_batch_with_several_subjects_is_not_compared_across_companies(self):
        hits = self._validate([
            self._doc("X1", "BUSINESS_LICENSE", {"bl.company_name": "甲公司",
                                                 "bl.unified_social_credit_code": self.USCC}),
            self._doc("X2", "BANK_STATEMENT", {"bs.account_name": "乙公司"}),
            self._doc("X3", "CREDIT_REPORT_ENTERPRISE", {"cr.company_name": "丙公司",
                                                          "cr.unified_social_credit_code": self.OTHER})])
        self.assertEqual(hits, {})

    def test_the_uscc_rule_fails_on_the_credit_codes_not_on_the_names(self):
        hits = self._validate([
            self._doc("Y1", "BUSINESS_LICENSE", {"bl.company_name": "同一名称有限公司",
                                                 "bl.unified_social_credit_code": self.USCC}),
            self._doc("Y2", "CREDIT_REPORT_ENTERPRISE", {"cr.company_name": "同一名称有限公司",
                                                         "cr.unified_social_credit_code": self.OTHER})])
        self.assertEqual(sorted(hits),
                         [("Y1", "bl.unified_social_credit_code"),
                          ("Y2", "cr.unified_social_credit_code")])

    def test_the_severity_word_comes_from_the_registry_not_from_the_code(self):
        docs = [self._doc("X1", "BUSINESS_LICENSE", {"bl.company_name": "甲公司",
                                                     "bl.unified_social_credit_code": self.USCC}),
                self._doc("X2", "BANK_STATEMENT", {"bs.account_name": "乙公司"})]
        self.assertTrue(all("V-X-01/warn" in items for items in self._validate(docs).values()))
        moved = copy.deepcopy(self.config)
        moved["cross_document"]["rules"]["V-X-01"]["severity"] = "fail"
        hits = self._validate(docs, moved)
        self.assertTrue(all("V-X-01/fail" in items for items in hits.values()))

    def test_a_contract_warn_still_forces_review_but_a_declined_one_does_not(self):
        doc = self._doc("X1", "BUSINESS_LICENSE", {"bl.company_name": "示例信息技术有限公司",
                                                   "bl.unified_social_credit_code": self.USCC})
        items = [{"id": "V-X-01", "status": self.rules["V-X-01"]["severity"],
                  "detail": "same USCC but different company name"}]
        armed = decide_field(copy.deepcopy(self._field("bl.company_name", "示例信息技术有限公司")),
                             doc, self._parsed(), self.config, copy.deepcopy(items),
                             Calibrator(self.config))
        self.assertEqual(armed["decision"], "review")
        self.assertIn("CROSS_FIELD_FAIL", armed["review_reasons"])
        self.assertEqual(armed["signals"]["cross_field_score"], 0.0)

        moved = copy.deepcopy(self.config)
        moved["cross_document"]["rules"]["V-X-01"]["review"] = False
        relaxed = decide_field(copy.deepcopy(self._field("bl.company_name", "示例信息技术有限公司")),
                               doc, self._parsed(), moved, copy.deepcopy(items),
                               Calibrator(moved))
        self.assertNotIn("CROSS_FIELD_FAIL", relaxed["review_reasons"])

    def test_the_emitted_set_and_the_registry_enforced_flags_agree(self):
        self.assertEqual(sorted(rid for rid, entry in self.rules.items() if entry.get("enforced")),
                         sorted(self.trust.CROSS_DOC_EMITTED))
        expected = {"V-X-01": sorted(set(self.trust.CROSS_DOC_NAME_PATHS.values())),
                    "V-X-02": sorted(self.trust.CROSS_DOC_USCC_PATHS),
                    "V-X-04": sorted(self.trust.STATEMENT_RECENCY_PATHS)}
        for rule_id in self.trust.CROSS_DOC_EMITTED:
            self.assertEqual(sorted(self.rules[rule_id]["paths"]), expected[rule_id])

    def test_a_missing_registry_entry_fails_loudly_instead_of_guessing_a_word(self):
        stripped = copy.deepcopy(self.config)
        del stripped["cross_document"]["rules"]["V-X-01"]
        docs = [self._doc("X1", "BUSINESS_LICENSE", {"bl.company_name": "甲公司",
                                                     "bl.unified_social_credit_code": self.USCC}),
                self._doc("X2", "BANK_STATEMENT", {"bs.account_name": "乙公司"})]
        with self.assertRaises(ValueError):
            self._validate(docs, stripped)


class GroundingContractTests(unittest.TestCase):
    """SCENARIO §7.1（line 377）与 §3.1：解析层自己负责丢掉没有页内坐标的块/单元格。

    这条铁律以前只在字段层执行（trust._grounding_ok，页边容差还硬写成 ±1 点）：
    解析产物里既没有丢弃动作，也没有被丢弃结果的记录，`GROUNDING_MISSING` 要等
    某个字段恰好抽到那个块才会出现——没抽到的坏块就静默留在 lines/tables 里。
    """

    def setUp(self):
        from docpipe import parse, trust
        self.parse = parse
        self.trust = trust
        self.config = copy.deepcopy(load_config())
        self.config["calibration"]["artifact"] = None

    def _tuned(self, tolerance):
        tuned = copy.deepcopy(self.config)
        tuned["parsing"]["grounding"]["tolerance_pt"] = tolerance
        return tuned

    def test_text_block_without_bbox_is_dropped_and_named(self):
        parts = [{"text": "正常", "bbox": [1, 2, 30, 40]}, {"text": "没有框"}]
        kept, dropped = self.parse.enforce_part_grounding(2, 595, 842, parts, self.config)
        self.assertEqual([item["text"] for item in kept], ["正常"])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["code"], "GROUNDING_MISSING")
        self.assertEqual(dropped[0]["reason"], "bbox_missing")
        self.assertEqual(dropped[0]["page"], 2)
        self.assertEqual(dropped[0]["kind"], "text")

    def test_zero_area_and_out_of_page_are_named_separately(self):
        parts = [{"text": "零宽", "bbox": [10, 10, 10, 20]},
                 {"text": "出页", "bbox": [0, 0, 999, 999]},
                 {"text": "形状", "bbox": [1, 2, 3]}]
        kept, dropped = self.parse.enforce_part_grounding(1, 595, 842, parts, self.config)
        self.assertEqual(kept, [])
        self.assertEqual([item["reason"] for item in dropped],
                         ["bbox_degenerate", "bbox_outside", "bbox_missing"])

    def test_tolerance_is_the_number_in_the_config(self):
        edge = [{"text": "压边", "bbox": [100, 100, 596.5, 120]}]
        kept, dropped = self.parse.enforce_part_grounding(
            1, 595, 842, copy.deepcopy(edge), self._tuned(0.0))
        self.assertEqual(kept, [])
        self.assertEqual([item["reason"] for item in dropped], ["bbox_outside"])
        kept, dropped = self.parse.enforce_part_grounding(
            1, 595, 842, copy.deepcopy(edge), self._tuned(2.0))
        self.assertEqual([item["text"] for item in kept], ["压边"])
        self.assertEqual(dropped, [])

    def test_table_cell_without_bbox_is_dropped_but_the_row_stays(self):
        table = {"table_id": "p1-transactions", "name": "transactions", "page": 1,
                 "bbox": [0, 0, 500, 500], "headers": ["日期", "金额"],
                 "rows": [{"row_index": 0, "page": 1, "cells": {
                     "日期": {"text": "2025-01-01", "bbox": [1, 2, 50, 20]},
                     "金额": {"text": "1.00"}}}]}
        kept, dropped = self.parse.enforce_table_grounding(
            1, 595, 842, [table], self.config)
        self.assertEqual(sorted(kept[0]["rows"][0]["cells"]), ["日期"])
        self.assertEqual(len(kept[0]["rows"]), 1)
        self.assertEqual([(item["kind"], item["reason"]) for item in dropped],
                         [("cell", "bbox_missing")])

    def test_table_booked_to_the_wrong_page_is_dropped_whole(self):
        table = {"name": "transactions", "page": 2, "bbox": [0, 0, 500, 500], "rows": []}
        kept, dropped = self.parse.enforce_table_grounding(
            1, 595, 842, [table], self.config)
        self.assertEqual(kept, [])
        self.assertEqual([(item["kind"], item["reason"]) for item in dropped],
                         [("table", "page_missing")])

    def test_both_layers_read_one_tolerance(self):
        field = {"path": "bl.company_name", "value": "示例", "required": True,
                 "risk": "party_name", "grounding": {"page": 1,
                 "bbox": [100, 100, 596.5, 120]}}
        pages = {1: {"page": 1, "width": 595, "height": 842}}
        tight, loose = self._tuned(0.0), self._tuned(2.0)
        self.assertEqual(self.trust._grounding_ok(
            field, 1, pages, self.trust.grounding_tolerance(tight)),
            (False, "bbox outside page"))
        self.assertEqual(self.trust._grounding_ok(
            field, 1, pages, self.trust.grounding_tolerance(loose)), (True, None))
        self.assertEqual(self.parse.grounding_tolerance(loose),
                         self.trust.grounding_tolerance(loose))

    def test_payload_and_batch_summary_carry_the_parse_drop_count(self):
        from docpipe.run import _document_payload, _stats
        parsed = {"page_count": 1, "sha256": "0" * 64,
                  "quality": {"parse_score": 0.99, "grounding_dropped": 2,
                              "grounding_dropped_by_reason": {"bbox_missing": 2}},
                  "pages": [{"page": 1, "width": 595, "height": 842,
                             "quality_flags": [], "preprocess": {}, "raster": None}]}
        extracted = {"document_id": "G1", "doc_type": "BUSINESS_LICENSE",
                     "doc_type_confidence": 0.9, "fields": {}}
        payload = _document_payload(extracted, parsed, Path("g.pdf"), config=self.config)
        self.assertEqual(payload["document_quality"]["grounding_dropped"], 2)
        self.assertEqual(_stats([payload])["parse_grounding_dropped_count"], 2)



class DocContractTests(unittest.TestCase):
    """SCENARIO §8.1/§6.4：原因码三张表同源，单文档 JSON 也必须长成契约里的样子。

    三张表 = `trust.REVIEW_REASON_CODES`（谁在发）/ `hard_abstain_reasons`（谁强制拦）/
    `validation.review_actions`（谁带「怎么办」）。漂移是静默的：LOW_CONFIDENCE 曾经被
    发出来却既没文案也不在强制表里，界面上只剩一串英文；RAW_UNVERIFIED 反过来有文案却
    双重惰性。每条反向夹具都只破一个口，报出来的问题必须点名那个码。
    """

    @classmethod
    def setUpClass(cls):
        from docpipe import doccontract, trust
        cls.dc = doccontract
        cls.trust = trust
        cls.config = load_config()

    def _problems(self, config=None):
        return " | ".join(self.dc.check_reason_tables(config or self.config))

    def _payload(self, config=None, **overrides):
        from docpipe.run import _document_payload
        config = config or self.config
        parsed = RawTextAuditTests._parsed([RawTextAuditTests._page(1, "1,000.00")])
        extracted = extract_document(parsed, config)
        assessed = assess([extracted], {"RAW1": parsed}, config, cross_document=False)[0]
        doc = _document_payload(assessed, parsed, Path("x.pdf"), config=config)
        doc.update(overrides)
        return doc

    def test_shipped_config_keeps_the_three_tables_in_step(self):
        self.assertEqual(self._problems(), "")
        emitted = set(self.trust.REVIEW_REASON_CODES)
        forced = set(self.config["hard_abstain_reasons"])
        # 唯一允许「发得出来但不强制」的是与决策同时写下的 LOW_CONFIDENCE。
        self.assertEqual(emitted - forced, set(self.trust.SELF_ABSTAIN_CODES))
        self.assertEqual(set(self.dc.SCENARIO_ABSTAIN_CODES) - forced, set())

    def test_a_code_without_guidance_is_reported(self):
        config = copy.deepcopy(self.config)
        config["validation"]["review_actions"].pop("LOW_CONFIDENCE")
        self.assertIn("LOW_CONFIDENCE", self._problems(config))
        self.assertIn("没文案", self._problems(config))

    def test_guidance_for_a_code_nobody_emits_is_reported(self):
        config = copy.deepcopy(self.config)
        config["validation"]["review_actions"]["GHOST_CODE"] = "这句永远不会被看到"
        self.assertIn("GHOST_CODE", self._problems(config))
        self.assertIn("永远发不出来", self._problems(config))

    def test_a_scenario_code_dropped_from_the_forced_table_is_reported(self):
        config = copy.deepcopy(self.config)
        config["hard_abstain_reasons"].remove("CHAIN_UNRELIABLE")
        self.assertIn("CHAIN_UNRELIABLE", self._problems(config))
        self.assertIn("§6.4", self._problems(config))

    def test_an_emitted_code_that_forces_nothing_is_reported(self):
        config = copy.deepcopy(self.config)
        config["hard_abstain_reasons"].remove("RAW_UNVERIFIED")
        self.assertIn("RAW_UNVERIFIED", self._problems(config))

    def test_a_self_abstain_code_may_not_be_forced(self):
        config = copy.deepcopy(self.config)
        config["hard_abstain_reasons"].append("LOW_CONFIDENCE")
        self.assertIn("LOW_CONFIDENCE", self._problems(config))

    def test_a_clean_payload_passes_the_shape_check(self):
        doc = self._payload()
        self.assertEqual(self.dc.check_document(doc), [])

    def test_each_shape_violation_is_named(self):
        from collections import OrderedDict
        base = self._payload()
        field_id = next(key for key, field in base["fields"].items()
                        if isinstance(field.get("grounding"), dict))
        cases = OrderedDict()

        def mutated(fn):
            doc = json.loads(json.dumps(base))
            fn(doc)
            return doc

        cases["decimal 值不是字符串"] = mutated(
            lambda doc: doc["fields"][field_id].update(value_type="decimal", value=1000.0))
        cases["confidence 4 位小数"] = mutated(
            lambda doc: doc["fields"][field_id].update(confidence=0.9876))
        cases["未知原因码"] = mutated(
            lambda doc: doc["fields"][field_id].update(decision="review",
                                                       review_reasons=["MADE_UP"]))
        cases["已接受却挂理由"] = mutated(
            lambda doc: doc["fields"][field_id].update(decision="accept",
                                                       review_reasons=["LOW_OCR"]))
        cases["grounding 缺 bbox_norm"] = mutated(
            lambda doc: doc["fields"][field_id]["grounding"].pop("bbox_norm"))
        cases["必填为空却没有 REQUIRED_EMPTY"] = mutated(
            lambda doc: doc["fields"][field_id].update(value=None, required=True,
                                                       review_reasons=[]))
        cases["时间戳不带 +08:00"] = mutated(
            lambda doc: doc.update(extraction_timestamp="2026-10-28T21:27:12"))
        cases["sha256 不是 64 位"] = mutated(
            lambda doc: doc["source"].update(sha256="abc"))
        cases["summary 计数对不上"] = mutated(
            lambda doc: doc["summary"].update(accept_count=99))
        cases["顶层缺键"] = mutated(lambda doc: doc.pop("validation_summary"))
        for label, doc in cases.items():
            with self.subTest(label=label):
                self.assertTrue(self.dc.check_document(doc), f"{label} 没被报出来")

    def test_the_cli_flags_a_thin_tree_and_a_clean_one_passes(self):
        import contextlib
        import io
        import tempfile
        with contextlib.redirect_stdout(io.StringIO()),                 tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(self.dc.main(["--out", str(root)]), 1)  # 空树 = 覆盖不足
            doc = self._payload()
            (root / "D900").mkdir()
            (root / "D900" / "D900.json").write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(self.dc.main(["--out", str(root), "--min-documents", "1"]), 0)
            self.assertEqual(self.dc.main(["--out", str(root), "--min-documents", "5"]), 1)
            doc["summary"]["field_count"] = 999
            (root / "D900" / "D900.json").write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(self.dc.main(["--out", str(root), "--min-documents", "1"]), 1)

    def test_low_confidence_is_emitted_with_review_and_guidance(self):
        from docpipe.review_actions import field_actions, review_action_table
        from docpipe.trust import Calibrator, decide_field
        config = copy.deepcopy(self.config)
        config["thresholds"]["global_min_confidence"] = 0.995
        page = {"page": 1, "width": 595, "height": 842, "quality_flags": [],
                "preprocess": {}, "lines": [], "tables": []}
        parsed = {"document_id": "L1", "doc_type": "BUSINESS_LICENSE", "page_count": 1,
                  "quality": {"parse_score": 0.99}, "pages": [page]}
        field = {"path": "bl.company_name", "value": "示例有限公司", "raw_text": "示例有限公司",
                 "value_type": "string", "required": True, "risk": "party_name",
                 "extractor_id": "label_value:企业名称",
                 "grounding": {"page": 1, "bbox": [100, 100, 200, 120]},
                 "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                             "table_score": 1.0, "cross_field_score": 1.0},
                 "meta": {}}
        decided = decide_field(field, {"doc_type": "BUSINESS_LICENSE", "fields": {}}, parsed,
                               config, [], Calibrator(config))
        self.assertEqual(decided["decision"], "review")
        self.assertIn("LOW_CONFIDENCE", decided["review_reasons"])
        # 这个码以前没有文案：界面与 CSV 只会给复核员一串英文。
        self.assertTrue(field_actions(decided, review_action_table(config)))

class PlaceholderTests(unittest.TestCase):
    """SCENARIO §4.1 规则 2：「必填字段为空、为 -- 、*** 、无 等占位符，判错」。

    占位符不是值：它先说「这一格没有数」，再由人判断这句「没有」是真是假。
    整格相等比较，词内出现不动；清单在 `validation.placeholders`，代码不认字面量。
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def _string(self, raw, config=None, **spec):
        from docpipe.extract import _normalize
        return _normalize(raw, {"type": "string", **spec}, config or self.config, {})

    def _decided(self, revenue, config=None):
        parsed = RawTextAuditTests._parsed([RawTextAuditTests._page(1, revenue)])
        extracted = extract_document(parsed, config or self.config)
        assessed = assess([extracted], {"RAW1": parsed}, config or self.config,
                          cross_document=False)[0]
        return extracted["fields"]["is.revenue"], assessed["fields"]["is.revenue"]

    def test_every_configured_placeholder_becomes_empty(self):
        from docpipe.extract import placeholders
        for token in placeholders(self.config):
            value, status, meta = self._string(token)
            self.assertIsNone(value, token)
            self.assertEqual(status, "placeholder", token)
            self.assertEqual(meta["placeholder"], token)

    def test_a_placeholder_inside_a_word_is_not_a_value_gap(self):
        self.assertEqual(self._string("生产--销售"), ("生产--销售", "ok", {}))

    def test_the_list_is_config_not_code(self):
        config = copy.deepcopy(self.config)
        config["validation"]["placeholders"] = ["NIL"]
        self.assertEqual(self._string("NIL", config)[1], "placeholder")
        # 代码里没有硬编码 "--"：换成别的清单之后它就只是个普通字符串。
        self.assertEqual(self._string("--", config)[1], "ok")

    def test_a_placeholder_survives_no_value_type(self):
        """日期/金额/整数/枚举曾经各自解析失败退化成 FORMAT_FAIL，现在统一是「不是值」。"""
        from docpipe.extract import _normalize
        for kind in ("date", "decimal", "integer", "enum"):
            spec = {"type": kind, "enum": "unit"}
            value, status, meta = _normalize("--", spec, self.config, {})
            self.assertIsNone(value, kind)
            self.assertEqual(status, "placeholder", kind)
            self.assertEqual(meta["placeholder"], "--")

    def test_a_required_placeholder_goes_to_review_with_guidance(self):
        raw, decided = self._decided("--")
        self.assertIsNone(raw["value"])
        self.assertEqual(raw["status"], "placeholder")
        self.assertEqual(decided["decision"], "review")
        self.assertIn("PLACEHOLDER_EMPTY", decided["review_reasons"])
        self.assertIn("REQUIRED_EMPTY", decided["review_reasons"])
        self.assertTrue(field_actions(decided, review_action_table(self.config)))

    def test_the_gate_can_silence_the_code_but_not_the_emptiness(self):
        config = copy.deepcopy(self.config)
        config["validation"]["review_on_placeholder"] = False
        raw, decided = self._decided("--", config)
        self.assertIsNone(raw["value"])
        self.assertNotIn("PLACEHOLDER_EMPTY", decided["review_reasons"])
        # 开关关掉的只是这条码；「必填为空」仍然把它拦在复核。
        self.assertIn("REQUIRED_EMPTY", decided["review_reasons"])
        self.assertEqual(decided["decision"], "review")

class EnumContractTests(unittest.TestCase):
    """SCENARIO §4.2：「只接受配置中的枚举；同义写法按配置映射，映射不唯一则复核」。

    两件事以前都没人管：`status=unmapped` 只写进 payload、判决层不读它，不在词表里的
    原文照样自动接受；同一个别名挂在两个规范值下时按字典顺序静默挑第一个。
    """

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def _enum(self, raw, config=None, enum="currency"):
        from docpipe.extract import _normalize
        return _normalize(raw, {"type": "enum", "enum": enum}, config or self.config, {})

    def _decided(self, raw, config=None, path="bs.currency", doc_type="BANK_STATEMENT",
                 enum="currency"):
        from docpipe.trust import validate_document_fields
        config = config or self.config
        value, status, meta = self._enum(raw, config, enum)
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
        doc = {"doc_id": "E001", "doc_type": doc_type, "fields": {path: field}}
        validators = {}
        validate_document_fields(doc, parsed, config, validators)
        decided = decide_field(field, doc, parsed, config, validators.get(path, []),
                               Calibrator(config))
        return field, decided, validators.get(path, [])

    def test_a_value_outside_the_enum_is_never_accepted(self):
        field, decided, validators = self._decided("美元")
        self.assertEqual(field["status"], "unmapped")
        self.assertEqual(decided["decision"], "review")
        self.assertIn("FORMAT_FAIL", decided["review_reasons"])
        self.assertEqual([v["detail"] for v in validators],
                         ["value not in the configured enum"])
        # 值仍是原文：复核员要看的就是票面上印的那三个字。
        self.assertEqual(decided["value"], "美元")

    def test_a_unique_alias_still_maps_and_stays_accepted(self):
        field, decided, validators = self._decided("人民币")
        self.assertEqual(field["value"], "CNY")
        self.assertEqual(decided["decision"], "accept")
        self.assertEqual(decided["review_reasons"], [])
        self.assertEqual(validators, [])

    def test_an_alias_owned_by_two_values_is_not_guessed(self):
        config = copy.deepcopy(self.config)
        config["enums"]["currency"] = {"CNY": ["人民币", "元"], "USD": ["美元", "元"]}
        value, status, meta = self._enum("元", config)
        self.assertEqual(value, "元")
        self.assertEqual(status, "unmapped")
        self.assertEqual(meta["enum_ambiguous"], ["CNY", "USD"])
        field, decided, validators = self._decided("元", config)
        self.assertEqual(decided["decision"], "review")
        self.assertIn("FORMAT_FAIL", decided["review_reasons"])
        self.assertIn("CNY/USD", validators[0]["detail"])

    def test_the_rule_is_not_tied_to_one_enum_name(self):
        # 老代码只对 rating 做枚举校验（V-ID-07 + RATING_OPT），币种是漏的。
        _, currency, currency_rules = self._decided("美元")
        _, rating, rating_rules = self._decided("BBB", path="cr.credit_rating",
                                                doc_type="CREDIT_REPORT_ENTERPRISE",
                                                enum="rating")
        for decided, rules in ((currency, currency_rules), (rating, rating_rules)):
            self.assertEqual(decided["decision"], "review")
            self.assertEqual([v["id"] for v in rules], ["V-ID-07"])

    def test_an_enum_alias_outranks_the_placeholder_fold(self):
        """§4.2 的词表映射优先于 §4.1 的占位符折叠：`flag` 把 `-`/`无` 当成答案。"""
        self.assertEqual(self._enum("-", enum="flag"), ("unknown", "ok", {}))
        self.assertEqual(self._enum("无", enum="flag"), ("no", "ok", {}))
        # 词表里没有的符号仍然是「不是值」。
        self.assertEqual(self._enum("--", enum="flag")[1], "placeholder")

    def test_a_blank_enum_field_is_empty_not_a_format_failure(self):
        _, decided, validators = self._decided("")
        self.assertEqual(validators, [])
        self.assertNotIn("FORMAT_FAIL", decided["review_reasons"])


class MatchingContractTests(unittest.TestCase):
    """SCENARIO §10.1 的三条匹配规则：值完全相等、溯源「中心在框内 或 IoU≥0.5」、
    交易行内容键只含日期/金额/余额。每条都由一份反例单独点亮。
    """

    @classmethod
    def setUpClass(cls):
        from docpipe import evaluate as ev
        cls.ev = ev

    GT = {"page": 1, "bbox": [100.0, 100.0, 200.0, 200.0]}

    def test_the_centre_rule_rescues_a_low_iou_but_centred_box(self):
        # 整页大小的预测框：中心 (150,150) 在标注框内，IoU 只有 0.111。
        pred = {"page": 1, "bbox": [0.0, 0.0, 300.0, 300.0]}
        self.assertLess(self.ev._iou(self.GT["bbox"], pred["bbox"]), 0.5)
        self.assertTrue(self.ev.grounding_ok(self.GT, pred))
        self.assertFalse(self.ev.grounding_ok(self.GT, pred, {"center_inside": False}))

    def test_the_iou_rule_holds_with_the_centre_rule_switched_off(self):
        # 平移 10 点的紧框：IoU 0.68。关掉中心规则后，只有 min_iou 这道门在判。
        pred = {"page": 1, "bbox": [110.0, 110.0, 210.0, 210.0]}
        loose = {"center_inside": False}
        self.assertGreaterEqual(self.ev._iou(self.GT["bbox"], pred["bbox"]), 0.5)
        self.assertTrue(self.ev.grounding_ok(self.GT, pred, loose))
        self.assertFalse(self.ev.grounding_ok(self.GT, pred, {"center_inside": False,
                                                           "min_iou": 0.8}))

    def test_a_sliver_overlap_is_not_a_grounding_hit(self):
        # 旧口径下这种格子算命中：相交 100 平方点、中心 (210,210) 在框外。
        pred = {"page": 1, "bbox": [190.0, 190.0, 230.0, 230.0]}
        self.assertGreater(self.ev._iou(self.GT["bbox"], pred["bbox"]), 0.0)
        self.assertFalse(self.ev.grounding_ok(self.GT, pred))

    def test_page_and_missing_bbox_still_miss(self):
        inside = {"page": 1, "bbox": [100.0, 100.0, 200.0, 200.0]}
        self.assertFalse(self.ev.grounding_ok(self.GT, dict(inside, page=2)))
        self.assertFalse(self.ev.grounding_ok(self.GT, {"page": 1}))

    def test_decimal_tolerance_is_off_by_default(self):
        # 归一化会把金额四舍五入到分，所以差一分是能区分的最小单位。
        spec = {"type": "decimal", "risk": "monetary"}
        self.assertTrue(self.ev.values_match(spec, "100.00", "100"))
        self.assertFalse(self.ev.values_match(spec, "100.00", "100.01"))
        self.assertTrue(self.ev.values_match(spec, "100.00", "100.01",
                                              {"decimal_tolerance": 0.01}))

    @staticmethod
    def _statement(direction_override):
        labels = {"doc_type": "BANK_STATEMENT", "fields": {}}
        pred = {}
        row = {"txn_date": "2025-07-01", "amount": "100.00", "balance": "1100.00",
               "direction": "income"}
        for column, value in row.items():
            path = f"bs.transactions[0].{column}"
            labels["fields"][path] = {"value": value}
            pred[path] = {"value": direction_override if column == "direction" else value}
        return labels, pred

    def test_transaction_content_key_excludes_direction_by_default(self):
        config = load_config()
        labels, pred = self._statement("expense")
        loose = self.ev.transaction_comparison(labels, pred, "BANK_STATEMENT", config)
        self.assertEqual(loose["keys"], ["txn_date", "amount", "balance"])
        self.assertEqual(loose["content_matched"], 1)
        strict = copy.deepcopy(config)
        strict["evaluation"]["match"]["transaction_keys"] = [
            "txn_date", "amount", "balance", "direction"]
        other = self.ev.transaction_comparison(labels, pred, "BANK_STATEMENT", strict)
        self.assertEqual(other["content_matched"], 0)
        self.assertEqual(other["keys"][-1], "direction")

    def test_an_unknown_transaction_key_is_loud(self):
        config = copy.deepcopy(load_config())
        config["evaluation"]["match"]["transaction_keys"] = ["txn_date", "bogus"]
        with self.assertRaises(ValueError):
            self.ev.transaction_keys(config)

    def test_shipped_match_section_declares_the_contract(self):
        shipped = self.ev.match_options(load_config())
        self.assertEqual(float(shipped["decimal_tolerance"]), 0.0)
        self.assertEqual(float(shipped["min_iou"]), 0.5)
        self.assertIs(shipped["center_inside"], True)
        self.assertEqual(list(shipped["transaction_keys"]),
                         ["txn_date", "amount", "balance"])



class CorpusShapeTests(unittest.TestCase):
    """SCENARIO §9.1 的语料形状：少两格、整 split 全用文本层、流水砍短都必须点名。

    形状是所有指标的共同分母，薄分母上的指标和满分母上的指标长得一样；这几条
    都种在内存清单与内存配置上，不改磁盘上的 samples/ 与 labels/。
    """

    def setUp(self):
        self.config = load_config()
        self.documents = corpusaudit.split_documents("dev", self.config)

    def _codes(self, documents, config=None):
        shape = corpusaudit.audit_split("dev", config or self.config, documents)
        return shape, {item["code"] for item in shape["problems"]}

    def test_shipped_splits_match_the_declared_shape(self):
        for split in ("dev", "holdout"):
            shape = corpusaudit.audit_split(split, self.config)
            self.assertEqual(shape["problems"], [], split)
            self.assertEqual(shape["documents"], 10)
            self.assertGreaterEqual(shape["transactions"], 120)
        self.assertEqual(corpusaudit.audit(self.config)["problems"], 0)

    def test_a_thin_corpus_is_named(self):
        shape, codes = self._codes(self.documents[:5])
        self.assertEqual(shape["documents"], 5)
        self.assertIn("document_count", codes)
        self.assertIn("doc_type_count", codes)

    def test_a_text_only_split_is_named(self):
        documents = [dict(doc, format="text") for doc in self.documents]
        _, codes = self._codes(documents)
        self.assertIn("format_count", codes)
        self.assertIn("scan_coverage", codes)

    def test_statement_bounds_come_from_the_config(self):
        shape = corpusaudit.audit_split("dev", self.config)
        rows = [item["rows"] for item in shape["statements"]]
        tight = copy.deepcopy(self.config)
        tight["evaluation"]["corpus"]["statement_rows"] = {"min": 0, "max": min(rows) - 1}
        _, codes = self._codes(self.documents, tight)
        self.assertEqual(codes, {"statement_rows"})

    def test_transaction_floor_comes_from_the_config(self):
        total = corpusaudit.audit_split("dev", self.config)["transactions"]
        raised = copy.deepcopy(self.config)
        raised["evaluation"]["corpus"]["min_transactions_per_split"] = total + 1
        _, codes = self._codes(self.documents, raised)
        self.assertEqual(codes, {"transaction_floor"})

    def test_an_unreadable_label_is_named(self):
        documents = [dict(doc) for doc in self.documents]
        target = next(doc for doc in documents
                      if doc.get("doc_type") == "BANK_STATEMENT")
        target["label_path"] = "labels/dev/__missing__.json"
        shape, codes = self._codes(documents)
        self.assertIn("label_missing", codes)
        self.assertIn("transaction_floor", codes)
        counted = {item["document_id"] for item in shape["statements"]}
        self.assertNotIn(target["document_id"], counted)


class StatementRowAccountingTests(unittest.TestCase):
    """SCENARIO §5.2 V-BS-07：交易笔数与页面数据行数一致，重复表头/页脚不得计入。

    解析层过去把「重复表头」「页脚」「首列日期读坏」的行一起静默丢掉：55 笔交易和
    48 笔交易的产物长得一模一样，信任层的 V-BS-07 只在名单里、从没发过言。现在每条
    丢弃行都带原因与原文，这组用例钉住四件事——归类按行内容分（页脚是稀疏合并行，
    整行填满还带「合计」的真交易不许被放行）、良性原因只认配置、认不出的丢弃让该页
    交易字段复核、保留下来的行数与抽出的笔数对不上同样点名到页。
    """

    SPEC = {"headers": ["交易日期", "摘要", "对方户名", "收支方向", "交易金额", "余额"],
            "row_required": ["交易日期", "交易金额"]}
    COLUMNS = ("amount", "balance", "txn_date")

    @staticmethod
    def _cell(text):
        return {"text": text, "bbox": [10.0, 10.0, 60.0, 20.0], "confidence": 0.99}

    def _row(self, **values):
        return {"cells": {label: self._cell(value) for label, value in values.items()}}

    def _keep(self, rows, config=None):
        from docpipe import parse
        config = config or load_config()
        skipped = []
        kept = parse._rows_from_cells(rows, self.SPEC, "BANK_STATEMENT", 1, "transactions",
                                      skipped=skipped,
                                      footer_options=parse.footer_rule(config))
        return kept, skipped

    def _statement(self, kept_pages, extracted_pages=None, skips=None):
        extracted_pages = kept_pages if extracted_pages is None else extracted_pages
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
                for column in self.COLUMNS:
                    path = f"bs.transactions[{index}].{column}"
                    fields[path] = {"path": path, "label": column,
                                    "value": "2026-03-01" if column == "txn_date" else "1.00",
                                    "required": True, "risk": "monetary",
                                    "grounding": {"page": page_no}, "signals": {},
                                    "meta": {}, "validators": [], "review_reasons": []}
                index += 1
        doc = {"document_id": "RA1", "doc_type": "BANK_STATEMENT", "fields": fields}
        return doc, {"document_id": "RA1", "doc_type": "BANK_STATEMENT", "pages": pages}

    @staticmethod
    def _flagged(doc, parsed, config):
        from docpipe.trust import validate_bank_statement
        validators = {}
        validate_bank_statement(doc, parsed, config, validators)
        return sorted(path for path, items in validators.items()
                      for item in items if item["id"] == "V-BS-07")

    def test_each_dropped_row_is_named_with_a_reason_and_its_text(self):
        from docpipe import parse
        header = self._row(交易日期="交易日期", 摘要="摘要", 对方户名="对方户名",
                           收支方向="收支方向", 交易金额="交易金额", 余额="余额")
        footer = self._row(交易日期="第1页共2页", 交易金额="—")
        real = self._row(交易日期="2026-03-01", 摘要="货款", 对方户名="甲公司",
                         收支方向="收入", 交易金额="100.00", 余额="1100.00")
        kept, skipped = self._keep([header, real, footer])
        self.assertEqual(len(kept), 1)
        self.assertEqual([item["reason"] for item in skipped], ["header", "footer"])
        self.assertIn("第1页共2页", skipped[1]["text"])
        for item in skipped:
            self.assertIn(item["reason"], parse.ROW_SKIP_REASONS)

    def test_a_full_row_that_mentions_a_footer_word_is_not_waved_through(self):
        full = self._row(交易日期="2O26-03-01", 摘要="货款合计", 对方户名="甲公司",
                         收支方向="收入", 交易金额="100.00", 余额="1100.00")
        kept, skipped = self._keep([full])
        self.assertEqual(kept, [])
        self.assertEqual([item["reason"] for item in skipped], ["unaccounted"])

    def test_the_footer_cell_bound_is_config_not_a_constant(self):
        footer = self._row(交易日期="合计", 交易金额="100.00")
        self.assertEqual([item["reason"] for item in self._keep([footer])[1]], ["footer"])
        loose = load_config()
        loose["validation"]["row_accounting"]["footer_max_cells"] = 0
        self.assertEqual([item["reason"] for item in self._keep([footer], loose)[1]],
                         ["unaccounted"])

    def test_a_benign_drop_does_not_flag_the_page(self):
        doc, parsed = self._statement(
            [(1, 2), (2, 1)], None,
            {1: [{"row_index": 2, "reason": "footer", "text": "第1页共2页"}]})
        self.assertEqual(self._flagged(doc, parsed, load_config()), [])

    def test_an_unknown_drop_flags_only_that_page(self):
        doc, parsed = self._statement(
            [(1, 2), (2, 1)], None,
            {2: [{"row_index": 1, "reason": "unaccounted", "text": "2O26-03-01 货款"}]})
        self.assertEqual(self._flagged(doc, parsed, load_config()),
                         [f"bs.transactions[2].{column}" for column in self.COLUMNS])

    def test_the_benign_list_comes_from_the_config(self):
        config = load_config()
        config["validation"]["row_accounting"]["benign_skip_reasons"].append("unaccounted")
        doc, parsed = self._statement(
            [(1, 2), (2, 1)], None,
            {2: [{"row_index": 1, "reason": "unaccounted", "text": "x"}]})
        self.assertEqual(self._flagged(doc, parsed, config), [])

    def test_kept_rows_must_equal_extracted_rows(self):
        doc, parsed = self._statement([(1, 2)], [(1, 1)])
        hits = self._flagged(doc, parsed, load_config())
        self.assertEqual(hits, [f"bs.transactions[0].{column}" for column in self.COLUMNS])

    def test_losing_a_whole_page_flags_every_transaction(self):
        doc, parsed = self._statement([(1, 1), (2, 2)], [(1, 1)])
        hits = self._flagged(doc, parsed, load_config())
        self.assertEqual(hits, [f"bs.transactions[0].{column}" for column in self.COLUMNS])


class StatementRecencyTests(unittest.TestCase):
    """SCENARIO §5.3 V-IS-05 / §5.4 V-X-04：同一对日期，两条规则，阈值与降级对象都读配置。

    V-IS-05 在代码里从来没有对应物，V-X-04 登记成未实现；而两条规则比的都是利润表期末与
    征信查询日。实现时一旦在代码里另挑字段（比如只打在净利润上、或对流水也发一份），配置
    里的规则表就变成了装饰。
    """

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())
        self.config["calibration"]["artifact"] = None

    def _doc(self, doc_id, doc_type, values):
        return {"document_id": doc_id, "doc_type": doc_type,
                "fields": {path: {"path": path, "value": value} for path, value in values.items()}}

    def _pair(self, period_end, query, period_start="2023-01-01",
              statement_name="示例科技有限公司", report_name="示例科技有限公司",
              drop_query=False):
        statement = self._doc("S1", "INCOME_STATEMENT",
                              {"is.company_name": statement_name,
                               "is.statement_period_start": period_start,
                               "is.statement_period_end": period_end,
                               "is.net_profit": "100.00"})
        values = {"cr.company_name": report_name}
        if not drop_query:
            values["cr.query_date"] = query
        return [statement, self._doc("R1", "CREDIT_REPORT_ENTERPRISE", values)]

    def _validate(self, docs, config=None):
        validators = {doc["document_id"]: {} for doc in docs}
        self.trust.validate_cross_document(docs, config or self.config, validators)
        return {(doc_id, path): [f"{item['id']}/{item['status']}" for item in items]
                for doc_id, by_path in validators.items()
                for path, items in by_path.items()}

    def _field(self):
        return {"path": "is.net_profit", "label": "净利润", "value": "100.00",
                "value_type": "decimal", "required": True, "risk": "monetary",
                "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
                "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                            "table_score": 1.0, "cross_field_score": 1.0, "grounding_present": 1},
                "meta": {}, "decision": "review", "validators": [], "review_reasons": []}

    def _parsed(self):
        return {"page_count": 1, "quality": {"parse_score": 0.99},
                "pages": [{"page": 1, "width": 595, "height": 842, "quality_flags": [],
                           "raster": None}]}

    def test_a_fresh_pair_is_silent(self):
        self.assertEqual(self._validate(self._pair("2025-12-31", "2026-01-05")), {})

    def test_a_period_ending_after_the_query_warns_both_contract_dates(self):
        hits = self._validate(self._pair("2026-03-31", "2025-12-13", period_start="2025-01-01"))
        self.assertEqual(hits, {("S1", "is.statement_period_end"): ["V-X-04/warn"],
                                ("R1", "cr.query_date"): ["V-X-04/warn"]})

    def test_a_stale_period_degrades_exactly_the_three_contract_paths(self):
        hits = self._validate(self._pair("2023-12-31", "2026-01-05"))
        self.assertEqual(hits, {("S1", "is.statement_period_start"): ["V-IS-05/warn"],
                                ("S1", "is.statement_period_end"): ["V-IS-05/warn"],
                                ("S1", "is.net_profit"): ["V-IS-05/warn"]})

    def test_the_eighteen_month_boundary_is_the_config_number(self):
        self.assertEqual(self._validate(self._pair("2024-07-20", "2026-01-20")), {})
        hits = self._validate(self._pair("2024-06-20", "2026-01-20"))
        self.assertIn(("S1", "is.net_profit"), hits)

    def test_moving_only_the_threshold_silences_the_rule(self):
        moved = copy.deepcopy(self.config)
        moved["validation"]["statement_recency"]["stale_months"] = 30
        self.assertEqual(self._validate(self._pair("2023-12-31", "2026-01-05"), moved), {})

    def test_the_degraded_paths_come_from_the_config(self):
        moved = copy.deepcopy(self.config)
        moved["validation"]["statement_recency"]["degraded_paths"] = ["is.net_profit"]
        hits = self._validate(self._pair("2023-12-31", "2026-01-05"), moved)
        self.assertEqual(hits, {("S1", "is.net_profit"): ["V-IS-05/warn"]})

    def test_documents_of_different_subjects_are_not_paired(self):
        self.assertEqual(self._validate(
            self._pair("2023-12-31", "2026-01-05", report_name="另一家公司")), {})

    def test_an_undated_report_is_not_guessed(self):
        self.assertEqual(self._validate(self._pair("2023-12-31", "2026-01-05",
                                                   drop_query=True)), {})

    def test_a_single_month_number_breaks_the_config_loudly(self):
        broken = copy.deepcopy(self.config)
        del broken["validation"]["statement_recency"]["stale_months"]
        with self.assertRaises(ValueError):
            self._validate(self._pair("2023-12-31", "2026-01-05"), broken)

    def test_a_warn_lowers_the_signal_but_does_not_fake_a_failure(self):
        doc = self._doc("S1", "INCOME_STATEMENT",
                        {"is.company_name": "示例科技有限公司",
                         "is.statement_period_start": "2023-01-01",
                         "is.statement_period_end": "2023-12-31",
                         "is.net_profit": "100.00"})
        items = [{"id": "V-IS-05", "status": "warn", "detail": "25 months"}]
        armed = decide_field(self._field(), doc, self._parsed(), self.config,
                             copy.deepcopy(items), Calibrator(self.config))
        quiet = decide_field(self._field(), doc, self._parsed(), self.config, [],
                             Calibrator(self.config))
        self.assertEqual(armed["signals"]["cross_field_score"], 0.3)
        self.assertEqual(quiet["signals"]["cross_field_score"], 1.0)
        self.assertNotIn("CROSS_FIELD_FAIL", armed["review_reasons"])
        self.assertLess(self.trust.heuristic_confidence(armed, self.config),
                        self.trust.heuristic_confidence(quiet, self.config))

class IncomeColumnTests(unittest.TestCase):
    """SCENARIO §5.3 V-IS-04 后半句：本期列与上期列不得混列。

    前半句（币种确认不了 → 全部金额复核）早已落地；「混列」在代码里没有落点，而列身份
    唯一能自证的东西是金额框的横向位置。这里钉住：分得开的列沉默、落在另一列带里的数把
    全表金额置复核、两列分不开算无法确认、只有一列时条款自然满足，以及列名/阈值/开关
    全读配置、列带按页聚而不是整篇聚。
    """

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())

    def _doc(self, boxes, currency="CNY"):
        fields = {path: {"path": path, "value": "100.00", "value_type": "decimal",
                         "grounding": {"page": page, "bbox": bbox}}
                  for path, (page, bbox) in boxes.items()}
        return {"document_id": "S1", "doc_type": "INCOME_STATEMENT",
                "currency_unit": currency, "fields": fields}

    def _validators(self, doc, config=None):
        validators = {}
        self.trust.validate_income_statement(doc, config or self.config, validators)
        return validators

    def _hits(self, doc, config=None):
        return {path: sorted({f"{item['id']}/{item['status']}" for item in items})
                for path, items in self._validators(doc, config).items()}

    @staticmethod
    def _row(left, top=10, width=130):
        return (1, [left, top, left + width, top + 10])

    def test_a_separated_layout_stays_silent(self):
        doc = self._doc({"is.revenue": self._row(320), "is.cost_of_sales": self._row(320, 20),
                         "is.prior_revenue": self._row(450)})
        self.assertEqual(self._hits(doc), {})

    def test_a_field_in_the_other_column_fails_every_amount(self):
        boxes = {"is.revenue": self._row(320), "is.cost_of_sales": self._row(320, 20),
                 "is.prior_revenue": self._row(450)}
        boxes["is.cost_of_sales"] = self._row(450, 20)
        hits = self._hits(self._doc(boxes))
        self.assertEqual(set(hits), set(boxes))
        for items in hits.values():
            self.assertEqual(items, ["V-IS-04/fail"])

    def test_the_detail_names_the_declared_and_the_nearest_column(self):
        boxes = {"is.revenue": self._row(320), "is.cost_of_sales": self._row(450, 20),
                 "is.prior_revenue": self._row(450)}
        detail = self._validators(self._doc(boxes))["is.cost_of_sales"][0]["detail"]
        self.assertIn("is.cost_of_sales", detail)
        self.assertIn("本期金额", detail)
        self.assertIn("上期金额", detail)

    def test_columns_too_close_are_unconfirmed(self):
        close = {"is.revenue": self._row(320), "is.prior_revenue": self._row(330)}
        detail = self._validators(self._doc(close))["is.revenue"][0]["detail"]
        self.assertIn("unconfirmed", detail)

    def test_one_amount_column_has_nothing_to_mix(self):
        doc = self._doc({"is.revenue": self._row(320), "is.net_profit": self._row(320, 20)})
        self.assertEqual(self._hits(doc), {})

    def test_the_gap_threshold_is_config(self):
        close = {"is.revenue": self._row(320), "is.prior_revenue": self._row(330)}
        self.assertNotEqual(self._hits(self._doc(close)), {})
        narrow = copy.deepcopy(self.config)
        narrow["validation"]["income_columns"]["min_column_gap"] = 5.0
        self.assertEqual(self._hits(self._doc(close), narrow), {})

    def test_the_margin_decides_between_mixed_and_unconfirmed(self):
        boxes = {"is.revenue": self._row(320), "is.cost_of_sales": self._row(320, 20),
                 "is.prior_revenue": self._row(450),
                 "is.selling_expenses": self._row(390, 30)}
        detail = self._validators(self._doc(boxes))["is.selling_expenses"][0]["detail"]
        self.assertIn("mixed", detail)
        loose = copy.deepcopy(self.config)
        loose["validation"]["income_columns"]["assignment_margin"] = 40.0
        detail = self._validators(self._doc(boxes), loose)["is.selling_expenses"][0]["detail"]
        self.assertIn("unconfirmed", detail)
        self.assertNotIn("mixed", detail)

    def test_columns_are_clustered_per_page(self):
        boxes = {"is.revenue": (1, [320, 10, 450, 20]),
                 "is.cost_of_sales": (1, [320, 20, 450, 30]),
                 "is.prior_revenue": (1, [450, 10, 580, 20]),
                 "is.net_profit": (2, [700, 10, 830, 20]),
                 "is.income_tax": (2, [700, 20, 830, 30]),
                 "is.prior_net_profit": (2, [830, 10, 960, 20])}
        self.assertEqual(self._hits(self._doc(boxes)), {})

    def test_fields_without_coordinates_are_skipped_not_guessed(self):
        doc = self._doc({"is.revenue": self._row(320), "is.prior_revenue": self._row(450)})
        doc["fields"]["is.net_profit"] = {"path": "is.net_profit", "value": "100.00",
                                          "value_type": "decimal", "grounding": None}
        self.assertEqual(self._hits(doc), {})

    def test_disabling_the_switch_silences_it(self):
        boxes = {"is.revenue": self._row(320), "is.cost_of_sales": self._row(450, 20),
                 "is.prior_revenue": self._row(450)}
        off = copy.deepcopy(self.config)
        off["validation"]["income_columns"]["enabled"] = False
        self.assertEqual(self._hits(self._doc(boxes), off), {})

    def test_a_missing_threshold_breaks_the_config_loudly(self):
        broken = copy.deepcopy(self.config)
        del broken["validation"]["income_columns"]["min_column_gap"]
        with self.assertRaises(ValueError):
            self._validators(self._doc({"is.revenue": self._row(320)}), broken)

class IncomeAmountDomainTests(unittest.TestCase):
    """SCENARIO §4.5 利润表字段域：非负金额、期间顺序、所得税软上限。

    恒等式只证明「加减没算错」：把整张表的符号一起翻过来，V-IS-01/02/03 依然全部成立，
    每个金额仍然自动接受。字段表里那句「≥ 0」是拦住符号整体翻转的唯一证据；这些用例钉住
    它由字段自己的 rules（NON_NEGATIVE）声明而非代码里的路径清单、契约标了「可为负」的
    格子不受影响、期间倒置走 V-ID-04，以及所得税超过利润总额只降信号不阻断。
    """

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())

    def _doc(self, values, boxes=None):
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

    def _validators(self, doc, config=None):
        validators = {}
        self.trust.validate_document_fields(doc, {"pages": []}, config or self.config, validators)
        self.trust.validate_income_statement(doc, config or self.config, validators)
        return validators

    def _hits(self, doc, config=None):
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in self._validators(doc, config).items() if items}

    def test_a_negative_amount_fails_the_contract_rule(self):
        self.assertEqual(self._hits(self._doc({"is.revenue": "-100.00"})),
                         {"is.revenue": ["V-ID-05/fail"]})

    def test_zero_is_a_legal_amount(self):
        self.assertEqual(self._hits(self._doc({"is.revenue": "0.00"})), {})

    def test_fields_the_contract_allows_negative_stay_silent(self):
        values = {"is.finance_expenses": "-12.50", "is.other_income": "-3.00",
                  "is.prior_net_profit": "-900.00", "is.net_profit": "-900.00"}
        self.assertEqual(self._hits(self._doc(values, {"is.prior_net_profit": [580, 10, 710, 20]})),
                         {})

    def test_a_sign_flipped_statement_passes_identities_but_fails_the_domain(self):
        doc = self._doc({"is.revenue": "-1000.00", "is.cost_of_sales": "-600.00",
                         "is.taxes_surcharges": "0.00", "is.selling_expenses": "0.00",
                         "is.admin_expenses": "0.00", "is.finance_expenses": "0.00",
                         "is.operating_profit": "-400.00", "is.non_operating_income": "0.00",
                         "is.non_operating_expense": "0.00", "is.total_profit": "-400.00",
                         "is.income_tax": "0.00", "is.net_profit": "-400.00"})
        hits = self._hits(doc)
        identities = [item for items in hits.values() for item in items
                      if item.startswith("V-IS-0")]
        self.assertTrue(identities)
        self.assertFalse([item for item in identities if item.endswith("/fail")])
        self.assertIn("V-IS-01/pass", hits["is.revenue"])
        self.assertIn("V-ID-05/fail", hits["is.revenue"])
        self.assertIn("V-ID-05/fail", hits["is.cost_of_sales"])
        self.assertNotIn("V-ID-05/fail", hits["is.net_profit"])

    def test_the_non_negative_list_is_read_from_the_field_table(self):
        relaxed = copy.deepcopy(self.config)
        for spec in relaxed["doc_types"]["INCOME_STATEMENT"]["fields"]:
            if spec["path"] == "is.revenue":
                spec["rules"] = ["MONEY"]
        self.assertEqual(self._hits(self._doc({"is.revenue": "-100.00"}), relaxed), {})

    def test_a_reversed_period_fails_the_date_rule(self):
        hits = self._hits(self._doc({"is.statement_period_start": "2024-12-31",
                                     "is.statement_period_end": "2024-01-01"}))
        self.assertEqual(hits.get("is.statement_period_start"), ["V-ID-04/fail"])
        self.assertEqual(hits.get("is.statement_period_end"), ["V-ID-04/fail"])

    def test_income_tax_over_total_profit_only_warns(self):
        doc = self._doc({"is.total_profit": "100.00", "is.income_tax": "150.00",
                         "is.net_profit": "-50.00"})
        hits = self._hits(doc)
        self.assertIn("V-IS-06/warn", hits.get("is.income_tax") or [])
        self.assertIn("V-IS-03/pass", hits.get("is.total_profit"))
        loss = self._hits(self._doc({"is.total_profit": "-100.00",
                                     "is.income_tax": "0.00",
                                     "is.net_profit": "-100.00"}))
        self.assertNotIn("V-IS-06/warn",
                         [item for items in loss.values() for item in items])
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 595, "height": 842,
                             "quality_flags": [], "raster": None}]}
        decided = self.trust.decide_field(doc["fields"]["is.income_tax"], doc, parsed,
                                          self.config, self._validators(doc)["is.income_tax"],
                                          self.trust.Calibrator(self.config))
        self.assertNotIn("CROSS_FIELD_FAIL", decided["review_reasons"])
        self.assertEqual(decided["signals"]["cross_field_score"], 0.3)

    def test_the_tax_ceiling_severity_comes_from_the_config(self):
        strict = copy.deepcopy(self.config)
        strict["validation"]["income_tax_ceiling"]["severity"] = "fail"
        hits = self._hits(self._doc({"is.total_profit": "100.00", "is.income_tax": "150.00",
                                     "is.net_profit": "-50.00"}), strict)
        self.assertIn("V-IS-06/fail", hits.get("is.income_tax") or [])

    def test_disabling_the_tax_ceiling_silences_it(self):
        off = copy.deepcopy(self.config)
        off["validation"]["income_tax_ceiling"]["enabled"] = False
        hits = self._hits(self._doc({"is.total_profit": "100.00",
                                     "is.income_tax": "150.00",
                                     "is.net_profit": "-50.00"}), off)
        self.assertNotIn("V-IS-06/warn",
                         [item for items in hits.values() for item in items])

class StringLengthDomainTests(unittest.TestCase):
    """SCENARIO §4.3/§4.4 字段表的长度域：过短的串不许自动接受。

    契约在硬校验列里写着「长度 2～80」「非空时长度 ≥ 4」，代码此前一条都不查：一个字的
    「企业名称」非空、有坐标、格式全过，于是自动接受。这里钉住上下限由字段自己的
    min_len/max_len 声明（代码没有第二张长度表）、边界值放行、无长度声明的字段不受影响、
    以及改配置就翻面。
    """

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())

    def _hits(self, values, doc_type="BUSINESS_LICENSE", config=None):
        fields = {}
        for index, (path, value) in enumerate(values.items()):
            fields[path] = {"path": path, "value": value, "value_type": "string",
                            "grounding": {"page": 1, "bbox": [320, 10 + index * 10,
                                                              450, 20 + index * 10]}}
        doc = {"document_id": "S1", "doc_type": doc_type, "fields": fields}
        validators = {}
        self.trust.validate_document_fields(doc, {"pages": []}, config or self.config,
                                            validators)
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in validators.items() if items}

    def test_a_one_character_name_fails(self):
        self.assertEqual(self._hits({"bl.company_name": "司"}),
                         {"bl.company_name": ["V-ID-09/fail"]})

    def test_the_boundaries_pass(self):
        self.assertEqual(self._hits({"bl.company_name": "阿里"}), {})
        self.assertEqual(self._hits({"bl.company_name": "阿" * 80}), {})

    def test_a_name_over_the_ceiling_fails(self):
        self.assertEqual(self._hits({"bl.company_name": "阿" * 81}),
                         {"bl.company_name": ["V-ID-09/fail"]})

    def test_fields_without_declared_bounds_stay_silent(self):
        self.assertEqual(self._hits({"is.company_name": "司"}, "INCOME_STATEMENT"), {})

    def test_the_bounds_are_read_from_the_config(self):
        tight = copy.deepcopy(self.config)
        for spec in tight["doc_types"]["BUSINESS_LICENSE"]["fields"]:
            if spec["path"] == "bl.company_name":
                spec["max_len"] = 2
        self.assertEqual(self._hits({"bl.company_name": "阿里巴"}, config=tight),
                         {"bl.company_name": ["V-ID-09/fail"]})
        loose = copy.deepcopy(self.config)
        for spec in loose["doc_types"]["BUSINESS_LICENSE"]["fields"]:
            if spec["path"] == "bl.company_name":
                spec.pop("min_len", None)
                spec.pop("max_len", None)
        self.assertEqual(self._hits({"bl.company_name": "司"}, config=loose), {})

    def test_surrounding_blanks_do_not_inflate_the_length(self):
        self.assertEqual(self._hits({"bl.company_name": " 甲 "}),
                         {"bl.company_name": ["V-ID-09/fail"]})


class EnumUnknownTests(unittest.TestCase):
    """SCENARIO §4.6 的「未知则复核」：unknown 是配置认的词，但不是可用结论。

    契约把呆账 / 资产处置 / 对外担保三面标志的硬校验写成「未知则复核」；抽取侧把
    「未知 / 不详 / -」归一成 unknown 之后，V-ID-07 只问「是不是配置里认的词」，
    于是这个值可以一路自动接受。这里钉住：unknown 判 V-ID-10 并且决策侧记的是
    ENUM_UNKNOWN（不是 FORMAT_FAIL）、是/否不连带、名单读配置、其它枚举不受影响。
    """

    def setUp(self):
        from docpipe import trust
        self.trust = trust
        self.config = copy.deepcopy(load_config())

    def _hits(self, values, doc_type="CREDIT_REPORT_ENTERPRISE", config=None):
        fields = {}
        for index, (path, value) in enumerate(values.items()):
            fields[path] = {"path": path, "value": value, "value_type": "string",
                            "grounding": {"page": 1, "bbox": [120, 10 + index * 12,
                                                              360, 22 + index * 12]}}
        doc = {"document_id": "E1", "doc_type": doc_type, "fields": fields}
        validators = {}
        self.trust.validate_document_fields(doc, {"pages": []}, config or self.config,
                                            validators)
        return {path: sorted(f"{item['id']}/{item['status']}" for item in items)
                for path, items in validators.items() if items}

    def _decided(self, path, value):
        config = copy.deepcopy(self.config)
        config["calibration"]["artifact"] = None
        field = {"path": path, "value": value, "value_type": "string",
                 "required": True, "risk": "enum_flag",
                 "grounding": {"page": 1, "bbox": [120, 10, 360, 22]},
                 "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                             "table_score": 1.0, "cross_field_score": 1.0,
                             "grounding_present": 1},
                 "meta": {}, "decision": "review", "validators": [], "review_reasons": []}
        doc = {"document_id": "E1", "doc_type": "CREDIT_REPORT_ENTERPRISE",
               "fields": {path: field}}
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 400, "height": 400}]}
        validators = {}
        self.trust.validate_document_fields(doc, parsed, config, validators)
        return self.trust.decide_field(field, doc, parsed, config,
                                       validators.get(path) or [],
                                       self.trust.Calibrator(config))

    def test_the_unknown_flag_fails(self):
        self.assertEqual(self._hits({"cr.bad_debt_flag": "unknown"}),
                         {"cr.bad_debt_flag": ["V-ID-10/fail"]})

    def test_yes_and_no_are_not_caught(self):
        self.assertEqual(self._hits({"cr.bad_debt_flag": "no",
                                     "cr.asset_disposal_flag": "yes",
                                     "cr.guarantee_flag": "no"}), {})

    def test_the_list_comes_from_the_config(self):
        relaxed = copy.deepcopy(self.config)
        relaxed["validation"]["enum_review_values"] = {}
        self.assertEqual(self._hits({"cr.bad_debt_flag": "unknown"}, config=relaxed), {})
        narrowed = copy.deepcopy(self.config)
        narrowed["validation"]["enum_review_values"] = {"flag": ["yes"]}
        self.assertEqual(self._hits({"cr.bad_debt_flag": "unknown"}, config=narrowed), {})

    def test_other_enums_are_untouched(self):
        self.assertEqual(self._hits({"bs.currency": "CNY"}, "BANK_STATEMENT"), {})

    def test_the_decision_names_the_real_action(self):
        decided = self._decided("cr.bad_debt_flag", "unknown")
        self.assertEqual(decided["decision"], "review")
        self.assertIn("ENUM_UNKNOWN", decided["review_reasons"])
        self.assertNotIn("FORMAT_FAIL", decided["review_reasons"])

    def test_the_ordinary_flag_is_not_named_unknown(self):
        decided = self._decided("cr.bad_debt_flag", "no")
        self.assertEqual(decided["review_reasons"], [])



class ScenarioContractTests(unittest.TestCase):
    """SCENARIO §13 是契约的一部分：键要在、值要对、得有人读。"""

    def _spec(self):
        from docpipe import configcontract
        text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
        return configcontract, configcontract.section13_spec(text)

    def test_the_shipped_contract_holds(self):
        configcontract, spec = self._spec()
        result = configcontract.audit(spec, load_config(), configcontract.source_text())
        self.assertGreaterEqual(result["leaves"], 25)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["mismatched"], [])
        self.assertEqual(result["unread"], [])
        self.assertEqual(result["stale_deviations"], [])
        self.assertEqual(result["deviations"], ["output.include_parsed_debug"])

    def test_a_key_missing_from_the_yaml_is_reported(self):
        configcontract, spec = self._spec()
        config = copy.deepcopy(load_config())
        del config["parsing"]["table_engine"]
        result = configcontract.audit(spec, config, configcontract.source_text())
        self.assertEqual(result["missing"], ["parsing.table_engine"])

    def test_a_value_drift_is_reported(self):
        configcontract, spec = self._spec()
        config = copy.deepcopy(load_config())
        config["thresholds"]["global_min_confidence"] = 0.5
        result = configcontract.audit(spec, config, configcontract.source_text())
        self.assertEqual(len(result["mismatched"]), 1)
        self.assertTrue(result["mismatched"][0].startswith("thresholds.global_min_confidence"))

    def test_a_key_nobody_reads_is_reported(self):
        configcontract, spec = self._spec()
        config = copy.deepcopy(load_config())
        spec = copy.deepcopy(spec)
        spec["parsing"]["ghost_engine"] = "unregistered"
        config["parsing"]["ghost_engine"] = "unregistered"
        result = configcontract.audit(spec, config, configcontract.source_text())
        self.assertEqual(result["unread"], ["parsing.ghost_engine"])

    def test_dynamic_tier_keys_are_not_reported_unread(self):
        configcontract, spec = self._spec()
        result = configcontract.audit(spec, load_config(), configcontract.source_text())
        self.assertIn("thresholds.risk_tiers.identity", result["dynamic"])
        self.assertNotIn("thresholds.risk_tiers.identity", result["unread"])

    def test_a_stale_deviation_name_is_reported(self):
        configcontract, spec = self._spec()
        spec = copy.deepcopy(spec)
        del spec["output"]["include_parsed_debug"]
        result = configcontract.audit(spec, load_config(), configcontract.source_text())
        self.assertEqual(result["stale_deviations"], ["output.include_parsed_debug"])

    def test_unknown_engine_names_raise(self):
        from docpipe import parse
        config = copy.deepcopy(load_config())
        config["parsing"]["pdf_text_engine"] = "docling"
        with self.assertRaises(ValueError):
            parse.text_engine(config)
        config = copy.deepcopy(load_config())
        config["parsing"]["table_engine"] = "camelot"
        with self.assertRaises(ValueError):
            parse.table_engine(config)
        self.assertEqual(parse.text_engine(load_config()), "pymupdf")
        self.assertEqual(parse.table_engine(load_config()), "pdfplumber")

    def test_document_grouping_cannot_be_turned_off(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["group_by_document"] = False
        with self.assertRaises(ValueError):
            calibrate.require_document_grouping(config)
        calibrate.require_document_grouping(load_config())

    def test_the_calibration_method_must_be_implemented(self):
        config = copy.deepcopy(load_config())
        config["calibration"]["method"] = "svm"
        with self.assertRaises(ValueError):
            calibrate.calibration_method(config)
        self.assertEqual(calibrate.calibration_method(load_config()),
                         "logistic_regression_isotonic")


class LabelContractTests(unittest.TestCase):
    """SCENARIO §9.2：标注少一个键，评估器要么当场炸，要么静默记错。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.document = None
        for doc in corpusaudit.manifest_documents(cls.config):
            label, error = labelaudit.label_of(doc, cls.config)
            sizes = labelaudit.page_sizes(doc.get("document_id"), cls.config)
            if not error and sizes and (label or {}).get("fields"):
                cls.document, cls.label, cls.sizes = doc, label, sizes
                break
        cls.path = next(iter((cls.label or {}).get("fields") or {}), None)

    def setUp(self):
        if self.document is None:
            self.skipTest("没有带解析缓存的标注可用来篡改")

    def codes(self, mutate):
        doctored = copy.deepcopy(self.label)
        mutate(doctored)
        shape = labelaudit.audit_label(self.document, doctored, self.config, self.sizes)
        return [problem["code"] for problem in shape["problems"]]

    def test_the_shipped_labels_satisfy_section9_2(self):
        result = labelaudit.audit(self.config)
        fields = sum(shape["fields"] for shape in result["splits"].values())
        checked = sum(shape["bounds_checked"] for shape in result["splits"].values())
        self.assertEqual(result["problems"], 0)
        self.assertGreaterEqual(fields, 2000)
        self.assertEqual(checked, fields)

    def test_the_manifest_decides_which_splits_are_audited(self):
        documents = corpusaudit.manifest_documents(self.config)
        result = labelaudit.audit(self.config, documents=documents)
        self.assertEqual(set(result["splits"]),
                         {doc["split"] for doc in documents if doc.get("split")})

    def test_a_label_without_bbox_is_reported(self):
        self.assertEqual(self.codes(lambda it: it["fields"][self.path].pop("bbox")),
                         ["field_bbox_missing"])

    def test_a_label_without_page_is_reported(self):
        self.assertEqual(self.codes(lambda it: it["fields"][self.path].pop("page")),
                         ["field_page_missing"])

    def test_a_page_beyond_the_page_count_is_reported(self):
        beyond = int(self.document["page_count"]) + 1
        self.assertEqual(
            self.codes(lambda it: it["fields"][self.path].__setitem__("page", beyond)),
            ["field_page_out_of_range"])

    def test_a_collapsed_bbox_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it["fields"][self.path].__setitem__("bbox", [0, 0, 0, 0])),
            ["field_bbox_degenerate"])

    def test_a_bbox_off_the_page_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it["fields"][self.path].__setitem__(
                "bbox", [10000, 10, 10100, 20])),
            ["field_bbox_outside"])

    def test_a_non_numeric_bbox_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it["fields"][self.path].__setitem__(
                "bbox", ["a", 0, 1, 1])),
            ["field_bbox_shape"])

    def test_a_missing_value_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it["fields"][self.path].__setitem__("value", None)),
            ["field_value_missing"])

    def test_an_unregistered_field_path_is_reported(self):
        def add_ghost(it):
            it["fields"]["bl.ghost_field"] = copy.deepcopy(it["fields"][self.path])
        self.assertEqual(self.codes(add_ghost), ["field_path_unknown"])

    def test_an_unknown_doc_type_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it.__setitem__("doc_type", "GHOST_DOC_TYPE")),
            ["doc_type_unknown"])

    def test_a_document_id_mismatch_is_reported(self):
        self.assertEqual(
            self.codes(lambda it: it.__setitem__("document_id", "D999")),
            ["document_id_mismatch"])

    def test_an_empty_field_table_is_reported(self):
        self.assertEqual(self.codes(lambda it: it.__setitem__("fields", {})),
                         ["fields_missing"])

    def test_a_missing_label_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            document = {"document_id": "D000", "doc_type": "BUSINESS_LICENSE",
                        "page_count": 1,
                        "label_path": str(Path(tmp) / "ghost.json")}
            label, error = labelaudit.label_of(document, self.config)
            shape = labelaudit.audit_label(document, label, self.config, {}, error)
        self.assertEqual([problem["code"] for problem in shape["problems"]],
                         ["label_missing"])


class CsvContractTests(unittest.TestCase):
    """SCENARIO §8.4「列固定为」11 列：写出来 14 列，多出的三列必须登记且排在契约列之后。"""

    @classmethod
    def setUpClass(cls):
        from docpipe.run import _write_csv
        cls.write_csv = staticmethod(_write_csv)
        cls.config = load_config()
        cls.spec = csvaudit.section84_columns(
            (ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
        cls.fields = {
            "bs.closing_balance": {
                "value": "1300.00", "value_type": "decimal", "confidence": 0.9,
                "decision": "accept", "grounding": {"page": 1, "bbox": [10, 30, 80, 40]},
                "extractor_id": "label_value:期末余额"},
            "bs.transactions[0].counterparty_name": {
                "value": "示例信息技术有限公司", "value_type": "string", "confidence": 0.4,
                "decision": "review", "grounding": {"page": 2, "bbox": [10, 60, 180, 70]},
                "extractor_id": "table_cell:对手方",
                "review_reasons": ["LOW_OCR"],
            },
        }

    def build(self, temp, payload=None, config=None):
        out = Path(temp)
        payload = payload or self.payload()
        (out / "C1").mkdir(parents=True)
        fields = payload[0]["fields"] if payload else {}
        (out / "C1" / "C1.json").write_text(
            json.dumps({"document_id": "C1", "fields": fields}, ensure_ascii=False),
            encoding="utf-8")
        self.write_csv(out, payload, config or self.config)
        return out

    def payload(self):
        return [{"document_id": "C1", "doc_type": "BANK_STATEMENT",
                 "fields": copy.deepcopy(self.fields)}]

    def test_formula_text_is_escaped_and_signed_amount_stays_numeric(self):
        payload = self.payload()
        payload[0]["document_id"] = "+示例"
        fields = payload[0]["fields"]
        fields["bs.closing_balance"]["value"] = "-1300.00"
        fields["bs.transactions[0].counterparty_name"]["value"] = " \t＝1+2"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.write_csv(out, payload, self.config)
            with (out / "fields.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["document_id"] for row in rows], ["'+示例", "'+示例"])
        self.assertEqual([row["value"] for row in rows], ["-1300.00", "' \t＝1+2"])

    def rewrite(self, out, mutate, bom=True, config=None):
        columns = csvaudit.expected_columns(config or self.config)
        raw = (out / "fields.csv").read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        rows = list(reader)
        columns, rows = mutate(columns, rows)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        payload = buffer.getvalue().encode("utf-8")
        if bom:
            payload = b"\xef\xbb\xbf" + payload
        return payload

    def codes(self, out, mutate, bom=True, config=None):
        payload = self.rewrite(out, mutate, bom=bom, config=config)
        shape = csvaudit.audit(out, config or self.config, csv_bytes=payload)
        return [problem["code"] for problem in shape["problems"]]

    def test_the_config_prefix_is_word_for_word_section8_4(self):
        self.assertEqual(len(self.spec), 11)
        columns = csvaudit.expected_columns(self.config)
        self.assertEqual(columns[:len(self.spec)], self.spec)
        self.assertEqual(columns[len(self.spec):],
                         ["review_actions", "value_rewrite", "chain_evidence"])

    def test_the_written_header_equals_the_config_and_covers_every_row(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            result = csvaudit.audit(out, self.config)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["columns"], csvaudit.expected_columns(self.config))
        self.assertEqual((result["rows"], result["fields"]), (2, 2))

    def test_a_reordered_header_is_reported(self):
        def swap(columns, rows):
            columns = [columns[1], columns[0]] + columns[2:]
            return columns, rows
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, swap)
        self.assertEqual(codes, ["header_order"])

    def test_a_dropped_column_is_reported(self):
        def drop(columns, rows):
            columns = [column for column in columns if column != "review_reasons"]
            return columns, [{key: row.get(key, "") for key in columns} for row in rows]
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, drop)
        self.assertEqual(codes, ["header_missing"])

    def test_a_stripped_bom_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            shape = csvaudit.audit(out, self.config,
                                   csv_bytes=self.rewrite(out, lambda c, r: (c, r), bom=False))
        self.assertEqual([item["code"] for item in shape["problems"]], ["encoding_mismatch"])

    def test_a_row_that_never_gets_written_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, lambda columns, rows: (columns, rows[1:]))
        self.assertEqual(codes, ["row_missing"])

    def test_a_doubled_row_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, lambda columns, rows: (columns, rows + rows[:1]))
        self.assertEqual(codes, ["row_duplicate"])

    def test_a_row_for_an_unknown_document_is_reported(self):
        def ghost(columns, rows):
            return columns, rows + [dict(rows[0], document_id="Z999")]
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, ghost)
        self.assertEqual(codes, ["row_unknown_document"])

    def test_a_bad_decision_and_a_missing_value_type_are_reported(self):
        def spoil(columns, rows):
            rows[0]["decision"] = "accepted"
            rows[1]["value_type"] = ""
            return columns, rows
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, spoil)
        self.assertEqual(sorted(codes), ["decision_invalid", "value_type_missing"])

    def test_a_bbox_that_is_not_four_numbers_is_reported(self):
        def spoil(columns, rows):
            rows[0]["bbox"] = "[10, 30, 80]"
            return columns, rows
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp)
            codes = self.codes(out, spoil)
        self.assertEqual(codes, ["bbox_shape"])

    def test_a_config_whose_prefix_drifted_from_the_spec_is_reported(self):
        config = copy.deepcopy(self.config)
        columns = list(config["output"]["csv_columns"])
        columns[0], columns[1] = columns[1], columns[0]
        config["output"]["csv_columns"] = columns
        with tempfile.TemporaryDirectory() as temp:
            out = self.build(temp, config=config)
            result = csvaudit.audit(out, config)
        self.assertIn("config_columns_drift", [item["code"] for item in result["problems"]])
        self.assertEqual(len(result["problems"]), 1)


class SummaryContractTests(unittest.TestCase):
    """SCENARIO §8.3 的键树与数、§8.2 的目录形状：少一个键要么 KeyError 要么静默读错。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.spec = summaryaudit.section83_spec(
            (ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
        cls.paths = summaryaudit.spec_paths(cls.spec)

    def build(self, temp):
        out = Path(temp)
        (out / "C1").mkdir(parents=True)
        target = out / "C1" / "C1.json"
        target.write_text(json.dumps({"document_id": "C1", "fields": {"bs.name": {}}}),
                          encoding="utf-8")
        summary = {
            "run": {"started_at": "2026-10-29T00:00:00", "finished_at": "2026-10-29T00:00:01",
                    "pipeline_version": "0.1.0", "input_dir": "in", "document_count": 1},
            "overall": {"field_count": 1, "accept_count": 1, "review_count": 0,
                        "required_field_count": 1, "required_accept_count": 1,
                        "auto_accept_rate": 1.0, "field_accept_rate": 1.0,
                        "review_rate": 0.0, "avg_confidence": 0.9,
                        "grounding_missing_count": 0, "hard_fail_count": 0},
            "by_doc_type": [{"doc_type": "BANK_STATEMENT", "field_count": 1, "accept_count": 1,
                             "required_field_count": 1, "required_accept_count": 1,
                             "auto_accept_rate": 1.0, "field_accept_rate": 1.0}],
            "by_field_path": [{"field_path": "bs.name", "field_count": 1, "accept_count": 1,
                               "required_field_count": 1, "required_accept_count": 1,
                               "auto_accept_rate": 1.0, "field_accept_rate": 1.0}],
            "documents": [{"document_id": "C1", "doc_type": "BANK_STATEMENT",
                           "field_count": 1, "accept_count": 1,
                           "required_field_count": 1, "required_accept_count": 1,
                           "auto_accept_rate": 1.0, "field_accept_rate": 1.0,
                           "path": str(target)}],
        }
        (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "fields.csv").write_text("document_id,field_path\nC1,bs.name\n", encoding="utf-8-sig")
        return out, summary

    def codes(self, out, summary, csv_count=None):
        shape = summaryaudit.audit(out, self.config, summary=summary, csv_count=csv_count)
        return [item["code"] for item in shape["problems"]]

    def test_the_section8_3_spec_is_read_from_the_document(self):
        self.assertGreaterEqual(len(self.paths), 16)
        self.assertIn(("overall", "review_count"), self.paths)
        self.assertIn(("run", "document_count"), self.paths)
        self.assertIn(("documents",), self.paths)

    def test_a_well_formed_artifact_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            out, _ = self.build(temp)
            result = summaryaudit.audit(out, self.config)
        self.assertEqual(result["problems"], [])
        self.assertEqual((result["documents"], result["rows"]), (1, 1))

    def test_a_dropped_key_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["overall"].pop("review_count")
            codes = self.codes(out, summary)
        self.assertIn("key_missing", codes)

    def test_a_bumped_accept_count_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["overall"]["accept_count"] = 2
            codes = self.codes(out, summary)
        self.assertIn("count_mismatch", codes)
        self.assertIn("rate_mismatch", codes)

    def test_the_legacy_all_field_rate_is_rejected(self):
        """auto_accept_rate 若还按全部字段算（老口径），必须被点名，不能两个口径同一个名字。"""
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            overall = summary["overall"]
            overall.update(field_count=3, accept_count=2, review_count=1,
                           required_field_count=2, required_accept_count=1,
                           auto_accept_rate=0.6667, field_accept_rate=0.6667,
                           review_rate=0.3333)
            for row in summary["by_doc_type"] + summary["by_field_path"] + summary["documents"]:
                row.update(field_count=3, accept_count=2,
                           required_field_count=2, required_accept_count=1,
                           auto_accept_rate=0.6667, field_accept_rate=0.6667)
            codes = self.codes(out, summary)
        self.assertIn("rate_mismatch", codes)
        self.assertNotIn("count_mismatch", codes)

    def test_a_missing_required_denominator_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["overall"].pop("required_field_count")
            codes = self.codes(out, summary)
        self.assertIn("count_mismatch", codes)

    def test_a_group_row_without_the_required_denominator_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["by_doc_type"][0].pop("required_field_count")
            codes = self.codes(out, summary)
        self.assertIn("count_mismatch", codes)

    def test_a_group_that_does_not_add_up_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["by_doc_type"][0]["field_count"] = 5
            codes = self.codes(out, summary)
        self.assertIn("count_mismatch", codes)

    def test_a_document_count_that_disagrees_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["run"]["document_count"] = 3
            codes = self.codes(out, summary)
        self.assertIn("count_mismatch", codes)

    def test_a_directory_document_missing_from_the_summary_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            (out / "C2").mkdir()
            (out / "C2" / "C2.json").write_text('{"fields": {}}', encoding="utf-8")
            codes = self.codes(out, summary)
        self.assertIn("document_unlisted", codes)

    def test_a_summary_document_without_a_directory_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            summary["documents"].append({"document_id": "C9", "field_count": 0,
                                         "path": str(out / "C9" / "C9.json")})
            codes = self.codes(out, summary)
        self.assertIn("document_missing", codes)
        self.assertIn("document_path_missing", codes)

    def test_a_csv_row_count_that_disagrees_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, summary = self.build(temp)
            codes = self.codes(out, summary, csv_count=2)
        self.assertEqual(codes, ["csv_rows_mismatch"])

    def test_an_unreadable_summary_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            out, _ = self.build(temp)
            (out / "summary.json").write_text("{not json", encoding="utf-8")
            codes = [item["code"]
                     for item in summaryaudit.audit(out, self.config)["problems"]]
        self.assertEqual(codes, ["summary_unreadable"])


@requires("out/dev")
class DocumentContractTests(unittest.TestCase):
    """SCENARIO §8.1 的单文档 JSON：键树、金额写法、置信度精度、原因码、空值语义。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
        cls.spec = docaudit.section81_spec(text)
        cls.reasons = docaudit.reason_codes(text)

    def build(self) -> dict:
        field = {
            "value": "示例科技有限公司",
            "raw_text": "户名：示例科技有限公司",
            "value_type": "string",
            "confidence": 0.987,
            "decision": "accept",
            "grounding": {"page": 1, "bbox": [120.5, 88.2, 260.1, 102.7],
                          "bbox_norm": [0.157, 0.081, 0.339, 0.094],
                          "block_id": "p1-b3", "char_span": [3, 13]},
            "extractor_id": "rule_label_v1",
            "validators": [{"id": "V-X-01", "status": "pass", "detail": None}],
            "review_reasons": [],
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "extractor_agreement": None, "grounding_present": 1,
                        "raw_grounded": 1.0},
        }
        return {
            "schema_version": "1.0",
            "document_id": "C1",
            "doc_type": "BANK_STATEMENT",
            "doc_type_confidence": 0.99,
            "source": {"file_name": "C1.pdf", "sha256": "0" * 64, "page_count": 1},
            "extraction_timestamp": "2026-10-29T00:00:00+08:00",
            "pipeline_version": "0.1.0",
            "document_quality": {"parse_score": 0.9, "ocr_mean_confidence": 0.9,
                                 "warnings": []},
            "fields": {"bs.account_name": field},
            "validation_summary": {"hard_fail_count": 0, "warn_count": 0, "rules": []},
            "summary": {"field_count": 1, "accept_count": 1, "review_count": 0},
        }

    def codes(self, doc) -> list:
        result = docaudit.audit(ROOT / "out", self.config, document=doc)
        return [item["code"] for item in result["problems"]]

    def test_the_section8_1_spec_is_read_from_the_document(self):
        self.assertGreaterEqual(len(docaudit.spec_leaves(self.spec)), 15)
        self.assertIn("summary", self.spec)
        self.assertIn("review_reasons", str(docaudit._field_spec(self.spec)))
        self.assertIn("GROUNDING_MISSING", self.reasons)
        self.assertNotIn("SIGN_INFERRED", self.reasons)

    def test_a_well_formed_document_passes(self):
        self.assertEqual(self.codes(self.build()), [])

    def test_an_unregistered_top_level_key_is_reported(self):
        doc = self.build()
        doc["mystery"] = 1
        self.assertIn("key_extra", self.codes(doc))

    def test_a_registered_extension_key_passes(self):
        doc = self.build()
        doc["timing"] = {"total_seconds": 1.0}
        self.assertNotIn("key_extra", self.codes(doc))

    def test_decimal_written_as_number_is_reported(self):
        doc = self.build()
        doc["fields"]["bs.account_name"].update({"value": 12500.0, "value_type": "decimal"})
        self.assertIn("decimal_not_string", self.codes(doc))

    def test_decimal_written_as_string_passes(self):
        doc = self.build()
        doc["fields"]["bs.account_name"].update({"value": "-12500.00", "value_type": "decimal"})
        self.assertNotIn("decimal_not_string", self.codes(doc))

    def test_confidence_with_four_decimals_is_reported(self):
        doc = self.build()
        doc["fields"]["bs.account_name"]["confidence"] = 0.12345
        self.assertIn("confidence_precision", self.codes(doc))
        doc["fields"]["bs.account_name"]["confidence"] = 0.98
        self.assertNotIn("confidence_precision", self.codes(doc))

    def test_an_unknown_decision_is_reported(self):
        doc = self.build()
        doc["fields"]["bs.account_name"]["decision"] = "maybe"
        self.assertIn("decision_invalid", self.codes(doc))

    def test_a_timestamp_without_offset_is_reported(self):
        doc = self.build()
        doc["extraction_timestamp"] = "2026-10-29T00:00:00"
        self.assertIn("timestamp_offset", self.codes(doc))

    def test_a_summary_that_does_not_add_up_is_reported(self):
        doc = self.build()
        doc["summary"]["field_count"] = 2
        self.assertIn("summary_mismatch", self.codes(doc))

    def test_a_required_null_needs_the_required_empty_reason(self):
        doc = self.build()
        field = doc["fields"]["bs.account_name"]
        field.update({"value": None, "required": True})
        self.assertIn("null_reason_missing", self.codes(doc))
        field["review_reasons"] = ["REQUIRED_EMPTY"]
        self.assertNotIn("null_reason_missing", self.codes(doc))

    def test_an_unknown_reason_is_reported_but_registered_ones_pass(self):
        doc = self.build()
        doc["fields"]["bs.account_name"]["review_reasons"] = ["NOT_A_REASON"]
        self.assertIn("reason_unknown", self.codes(doc))
        doc["fields"]["bs.account_name"]["review_reasons"] = ["SIGN_INFERRED"]
        self.assertNotIn("reason_unknown", self.codes(doc))

    def test_a_stale_reason_extension_is_reported(self):
        config = copy.deepcopy(self.config)
        config["validation"]["review_reason_extensions"] = ["GROUNDING_MISSING"]
        result = docaudit.audit(ROOT / "out", config, scenario_text=(
            ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
        codes = [item["code"] for item in result["problems"]]
        self.assertIn("reason_extension_stale", codes)

    def test_bom_and_cli_exit_code(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "C1").mkdir()
            target = out / "C1" / "C1.json"
            target.write_bytes(b"\xef\xbb\xbf" + json.dumps(self.build()).encode("utf-8"))
            result = docaudit.audit(out, self.config)
            codes = [item["code"] for item in result["problems"]]
            self.assertIn("bom_present", codes)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(docaudit.main(["--out", str(out)]), 1)
            target.write_bytes(json.dumps(self.build()).encode("utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(docaudit.main(["--out", str(out)]), 0)

@requires("out/dev")
class SignalContractTests(unittest.TestCase):
    """SCENARIO §6.1 的七个置信度信号：键齐、值域、扩展登记。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
        cls.specs = docaudit.signal_spec(cls.text)
        cls.extensions = set(docaudit.signal_extensions(cls.config))

    def field(self, **overrides) -> dict:
        signals = {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                   "table_score": 1.0, "cross_field_score": 1.0,
                   "extractor_agreement": None, "grounding_present": 1,
                   "raw_grounded": 1.0}
        signals.update(overrides)
        return {"value": "示例科技有限公司", "signals": signals}

    def codes(self, field) -> list:
        return [item["code"] for item in docaudit.signal_problems(
            "bs.account_name", field, self.specs, self.extensions)]

    def test_the_section6_1_table_is_read_from_the_document(self):
        self.assertEqual(len(self.specs), 7)
        self.assertTrue(self.specs["ocr_conf"]["range"])
        self.assertEqual(self.specs["rule_exact"]["values"], [0.0, 0.5, 1.0])
        self.assertFalse(self.specs["format_valid"]["range"])
        self.assertFalse(self.specs["ocr_conf"]["null"])
        self.assertTrue(self.specs["extractor_agreement"]["null"])
        self.assertIn("raw_grounded", self.extensions)

    def test_a_well_formed_field_passes(self):
        self.assertEqual(self.codes(self.field()), [])

    def test_a_dropped_signal_is_reported(self):
        field = self.field()
        del field["signals"]["table_score"]
        self.assertIn("signal_missing", self.codes(field))

    def test_a_value_outside_the_domain_is_reported(self):
        for name, value in (("format_valid", 0.7), ("ocr_conf", 1.5),
                            ("rule_exact", 0.2), ("cross_field_score", 0.7),
                            ("grounding_present", 2)):
            with self.subTest(name=name, value=value):
                self.assertIn("signal_out_of_domain", self.codes(self.field(**{name: value})))
        for name, value in (("ocr_conf", 0.0), ("rule_exact", 0.5),
                            ("cross_field_score", 0.3), ("grounding_present", 0)):
            with self.subTest(name=name, value=value):
                self.assertNotIn("signal_out_of_domain", self.codes(self.field(**{name: value})))

    def test_null_is_allowed_only_where_the_table_says_so(self):
        self.assertNotIn("signal_out_of_domain",
                         self.codes(self.field(extractor_agreement=None)))
        self.assertIn("signal_out_of_domain",
                      self.codes(self.field(format_valid=None)))

    def test_a_json_true_is_not_the_number_one(self):
        self.assertIn("signal_out_of_domain",
                      self.codes(self.field(grounding_present=True)))

    def test_an_unregistered_signal_is_reported(self):
        self.assertIn("signal_extra", self.codes(self.field(mystery_signal=1)))
        self.assertNotIn("signal_extra", self.codes(self.field()))

    def test_a_field_without_signals_is_reported(self):
        self.assertIn("signals_shape", self.codes({"value": "示例科技有限公司"}))

    def test_a_stale_extension_is_reported(self):
        config = copy.deepcopy(self.config)
        config["validation"]["signal_extensions"] = ["ocr_conf"]
        stale = docaudit.stale_signal_extensions(config, self.specs)
        self.assertEqual([item["code"] for item in stale], ["signal_extension_stale"])

    def test_the_live_documents_carry_every_signal(self):
        files = docaudit.document_files(ROOT / "out" / "dev")
        if not files:
            self.skipTest("out/dev has no products")
        doc = json.loads(files[0].read_text(encoding="utf-8"))
        result = docaudit.audit(ROOT / "out", self.config, document=doc)
        self.assertEqual(result["signals"], len(self.specs))
        self.assertEqual([item for item in result["problems"]
                          if item["code"].startswith("signal")], [])


class NullSemanticsTests(unittest.TestCase):
    """SCENARIO §8.1 的空值写法：抽取层只留一种「空」（None），四个出口各按自己的语法表达。"""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    @staticmethod
    def _spec(required: bool = True) -> dict:
        return {"path": "bs.account_name", "label": "户名", "type": "string",
                "required": required, "risk": "party_name"}

    def _field(self, value, required: bool = True) -> dict:
        from docpipe.extract import _field_payload
        field = _field_payload(self._spec(required), value, "ok", "", 0.98,
                               "rule_label_v1", None, {}, 1.0, 1.0)
        field["decision"] = "accept"
        return field

    def test_blank_strings_collapse_to_null(self):
        for value in ("", "   ", "\t\n"):
            with self.subTest(value=repr(value)):
                self.assertIsNone(self._field(value)["value"])

    def test_real_values_keep_their_whitespace(self):
        self.assertEqual(self._field(" 示例科技有限公司 ")["value"], " 示例科技有限公司 ")

    def test_required_null_is_reviewed_with_required_empty(self):
        field = self._field("")
        decided = decide_field(field, {"document_id": "T1", "doc_type": "BANK_STATEMENT"},
                               {"doc_type": "BANK_STATEMENT", "pages": []}, self.config,
                               [], Calibrator(self.config))
        self.assertIsNone(decided["value"])
        self.assertEqual(decided["decision"], "review")
        self.assertIn("REQUIRED_EMPTY", decided["review_reasons"])

    def test_optional_null_is_not_flagged_required_empty(self):
        field = self._field("", required=False)
        decided = decide_field(field, {"document_id": "T1", "doc_type": "BANK_STATEMENT"},
                               {"doc_type": "BANK_STATEMENT", "pages": []}, self.config,
                               [], Calibrator(self.config))
        self.assertIsNone(decided["value"])
        self.assertNotIn("REQUIRED_EMPTY", decided["review_reasons"])

    def test_json_outlet_writes_null_not_empty_string(self):
        field = self._field("")
        dumped = json.dumps(field, ensure_ascii=False)
        self.assertIn('"value": null', dumped)
        self.assertNotIn('"value": ""', dumped)

    def test_csv_outlet_writes_an_empty_cell(self):
        from docpipe.run import _write_csv
        payload = [{
            "document_id": "T1",
            "doc_type": "BANK_STATEMENT",
            "fields": {
                "bs.account_name": {
                    "value": None, "value_type": "string", "confidence": 0.4,
                    "decision": "review", "grounding": {}, "extractor_id": "rule_label_v1",
                    "review_reasons": ["REQUIRED_EMPTY"],
                },
            },
        }]
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(Path(tmp), payload, self.config)
            raw = (Path(tmp) / "fields.csv").read_bytes().decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw)))
        self.assertEqual(rows[0]["value"], "")
        self.assertEqual(rows[0]["review_reasons"], "REQUIRED_EMPTY")

    def test_docaudit_flags_empty_string_values(self):
        spec = docaudit.section81_spec((ROOT / "SCENARIO.md").read_text(encoding="utf-8"))
        base = DocumentContractTests().build()
        for value in ("", "   "):
            with self.subTest(value=repr(value)):
                doc = copy.deepcopy(base)
                path = next(iter(doc["fields"]))
                doc["fields"][path]["value"] = value
                codes = [item["code"] for item in
                         docaudit.field_problems(doc, spec, set(),
                                                 docaudit.reason_codes(
                                                     (ROOT / "SCENARIO.md").read_text(encoding="utf-8")))]
                self.assertIn("empty_string_value", codes)

@requires("out/dev", "labels/dev")
class EvaluationReportTests(unittest.TestCase):
    """SCENARIO §10.2：评估报告要输出七类内容，其中两项此前根本没算。

    报告只按文档类型与风险档分组：既没有「按字段路径」，也没有「平均置信度」。
    换一个抽取器或换一套阈值之后，看不出是哪一类字段在拖后腿；而均值不带覆盖率，
    三个格子撑起来的平均会冒充一千五百个格子。
    """

    def setUp(self):
        self.config = load_config()

    def _record(self, confidence, decision="accept", correct=True):
        return {"path": "bl.company_name", "correct": correct, "decision": decision,
                "risk": "identifying", "grounded": True, "confidence": confidence,
                "review_status": None, "reasons": [], "signals": {}, "value_type": "string"}

    def test_avg_confidence_only_counts_fields_that_have_one(self):
        stats = _decision_stats([self._record(0.9), self._record(0.7),
                                 self._record(None, decision="review")])
        self.assertEqual(stats["field_count"], 3)
        self.assertEqual(stats["avg_confidence"], 0.8)
        self.assertEqual(stats["confidence_coverage"], 0.6667)

    def test_avg_confidence_is_null_when_no_field_has_one(self):
        stats = _decision_stats([self._record(None), self._record(None, decision="review")])
        self.assertIsNone(stats["avg_confidence"])
        self.assertEqual(stats["confidence_coverage"], 0.0)

    def _write_pair(self, temp):
        pred = Path(temp) / "out" / "D900"
        labels = Path(temp) / "dev"
        pred.mkdir(parents=True)
        labels.mkdir(parents=True)
        fields = {}
        for index, amount in ((0, "100.00"), (3, "250.00")):
            fields[f"bs.transactions[{index}].amount"] = {
                "value": amount, "decision": "accept", "confidence": 0.95,
                "review_reasons": [],
            }
        (pred / "D900.json").write_text(json.dumps({
            "fields": fields, "source": {"page_count": 1},
            "timing": {"total_seconds": 1.5},
        }), encoding="utf-8")
        (labels / "D900.json").write_text(json.dumps({
            "doc_type": "BANK_STATEMENT",
            "fields": {path: {"value": item["value"]} for path, item in fields.items()},
        }), encoding="utf-8")
        return pred.parent, labels

    def test_by_field_path_collapses_transaction_indexes(self):
        from docpipe import evaluate
        with tempfile.TemporaryDirectory() as temp:
            pred, labels = self._write_pair(temp)
            result = evaluate.evaluate_predictions(pred, labels, self.config)
        rows = result["summary"]["by_field_path"]
        self.assertEqual([row["name"] for row in rows], ["bs.transactions[i].amount"])
        self.assertEqual(rows[0]["field_count"], 2)
        self.assertEqual(rows[0]["accept_count"], 2)
        self.assertEqual(sum(row["field_count"] for row in rows),
                         result["summary"]["field_count"])
        self.assertEqual(result["summary"]["avg_confidence"], 0.95)
        self.assertEqual(result["summary"]["confidence_coverage"], 1.0)

    def test_the_report_check_bites_when_an_item_is_dropped(self):
        from docpipe import evaluate
        from docpipe.acceptance import Report, _check_evaluation_report
        result = evaluate.evaluate_predictions(ROOT / "out" / "dev", ROOT / "labels" / "dev",
                                               self.config)

        def verdict(mutate=None):
            damaged = copy.deepcopy(result)
            if mutate:
                mutate(damaged["summary"])
            report = Report()
            _check_evaluation_report(report, damaged)
            self.assertEqual(len(report.checks), 1)
            return report.checks[0]["ok"]

        self.assertTrue(verdict())
        self.assertFalse(verdict(lambda summary: summary.pop("avg_confidence")))
        self.assertFalse(verdict(lambda summary: summary.pop("by_field_path")))
        self.assertFalse(verdict(lambda summary: summary["confusion"].pop("over_review")))
        self.assertFalse(verdict(lambda summary: summary["latency"].pop("avg_page_seconds")))

class Section65MetricsTests(unittest.TestCase):
    """SCENARIO §6.5 指标表的最后两行：溯源缺失率与文档全自动率。

    这两行此前一个都没算。报告里只有「已接受字段的溯源缺失数」——一份坐标全丢、
    整份进复核的文档在那个口径下显示 0% 缺失；「一份文档还要不要人从头看一遍」
    更是从未记录过，按字段数平摊会把「整份不可用」平均成一个小数点。
    """

    def setUp(self):
        self.config = load_config()

    @staticmethod
    def _field(value, decision="accept", page=1, bbox=(0.1, 0.1, 0.3, 0.14)):
        field = {"value": value, "value_type": "string", "confidence": 0.95,
                 "decision": decision, "review_reasons": []}
        if page is not None:
            field["grounding"] = {"page": page,
                                  "bbox": list(bbox) if bbox else None}
        return field

    def _write(self, temp, documents):
        """documents: {doc_id: {path: (gt_value, pred_field)}}"""
        out_dir = Path(temp) / "out"
        labels_dir = Path(temp) / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        for doc_id, fields in documents.items():
            (out_dir / doc_id).mkdir(parents=True, exist_ok=True)
            (out_dir / doc_id / f"{doc_id}.json").write_text(json.dumps({
                "fields": {path: field for path, (_, field) in fields.items()},
                "source": {"page_count": 1},
            }), encoding="utf-8")
            (labels_dir / f"{doc_id}.json").write_text(json.dumps({
                "doc_type": "BANK_STATEMENT",
                "fields": {path: {"value": gt, "page": 1,
                                  "bbox": [0.1, 0.1, 0.3, 0.14]}
                           for path, (gt, _) in fields.items()},
            }), encoding="utf-8")
        return out_dir, labels_dir

    def test_grounding_missing_counts_reviewed_fields_as_well(self):
        from docpipe.evaluate import evaluate_predictions
        with tempfile.TemporaryDirectory() as temp:
            out_dir, labels_dir = self._write(temp, {"D900": {
                "bs.account_name": ("示例科技有限公司", self._field("示例科技有限公司")),
                "bs.currency": ("CNY", self._field("CNY", decision="review", page=None)),
            }})
            summary = evaluate_predictions(out_dir, labels_dir, self.config)["summary"]
        # 旧口径只数已接受字段：那一格坐标全丢，却仍然是 0。
        self.assertEqual(summary["accepted_grounding_missing"], 0)
        self.assertEqual(summary["grounding_missing_count"], 1)
        self.assertEqual(summary["grounding_missing_rate"], 0.5)

    def test_a_bbox_that_cannot_be_used_counts_as_missing(self):
        from docpipe.evaluate import evaluate_predictions
        cases = {"零面积": (1, (0.2, 0.1, 0.2, 0.14)),
                 "缺页码": (None, (0.1, 0.1, 0.3, 0.14)),
                 "只有两个数": (1, (0.1, 0.1)),
                 "空框": (1, None)}
        for name, (page, bbox) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temp:
                out_dir, labels_dir = self._write(temp, {"D901": {
                    "bs.account_name": ("示例科技有限公司",
                                        self._field("示例科技有限公司", page=page, bbox=bbox)),
                }})
                result = evaluate_predictions(out_dir, labels_dir, self.config)
                summary = result["summary"]
                self.assertEqual(summary["grounding_missing_count"], 1)
                self.assertEqual(summary["grounding_missing_rate"], 1.0)
                # 全自动率只问「必填字段是不是都 accept」，坐标能不能画是另一张表；
                # §6.5 这两行各有各的分母，不合并成一个门槛。
                self.assertTrue(result["documents"][0]["document_fully_auto"])

    def test_optional_review_does_not_block_full_automation(self):
        from docpipe.evaluate import evaluate_predictions
        with tempfile.TemporaryDirectory() as temp:
            out_dir, labels_dir = self._write(temp, {"D902": {
                "bs.account_name": ("示例科技有限公司", self._field("示例科技有限公司")),
                "bs.currency": ("CNY", self._field("CNY")),
                "bs.branch_name": ("示例支行", self._field("示例支行", decision="review",
                                                          page=None)),
            }})
            result = evaluate_predictions(out_dir, labels_dir, self.config)
        document = result["documents"][0]
        self.assertEqual(document["required_field_count"], 2)
        self.assertTrue(document["document_fully_auto"])
        self.assertEqual(result["summary"]["document_fully_auto_count"], 1)
        self.assertEqual(result["summary"]["document_auto_rate"], 1.0)
        # 可选格进复核不挡整份自动化，但它仍然是一次人工核对，溯源缺失照数。
        self.assertEqual(result["summary"]["grounding_missing_count"], 1)

    def test_one_required_review_takes_the_whole_document_out(self):
        from docpipe.evaluate import evaluate_predictions
        with tempfile.TemporaryDirectory() as temp:
            out_dir, labels_dir = self._write(temp, {
                "D903": {
                    "bs.account_name": ("示例科技有限公司",
                                        self._field("示例科技有限公司", decision="review")),
                    "bs.currency": ("CNY", self._field("CNY")),
                },
                "D904": {
                    "bs.account_name": ("另一个公司", self._field("另一个公司")),
                    "bs.currency": ("CNY", self._field("CNY")),
                },
            })
            result = evaluate_predictions(out_dir, labels_dir, self.config)
        by_id = {doc["document_id"]: doc["document_fully_auto"] for doc in result["documents"]}
        self.assertEqual(by_id, {"D903": False, "D904": True})
        self.assertEqual(result["summary"]["document_fully_auto_count"], 1)
        self.assertEqual(result["summary"]["document_auto_rate"], 0.5)

    def test_the_check_bites_when_either_metric_drifts(self):
        from docpipe.acceptance import Report, _check_section65_metrics
        from docpipe.evaluate import evaluate_predictions
        with tempfile.TemporaryDirectory() as temp:
            out_dir, labels_dir = self._write(temp, {
                "D905": {
                    "bs.account_name": ("示例科技有限公司", self._field("示例科技有限公司")),
                    "bs.currency": ("CNY", self._field("CNY", decision="review", page=None)),
                },
                "D906": {
                    "bs.account_name": ("另一个公司", self._field("另一个公司")),
                    "bs.currency": ("CNY", self._field("CNY")),
                },
            })
            result = evaluate_predictions(out_dir, labels_dir, self.config)

            def verdict(mutate=None):
                damaged = copy.deepcopy(result)
                if mutate:
                    mutate(damaged["summary"])
                report = Report()
                _check_section65_metrics(report, damaged, out_dir, labels_dir, self.config)
                self.assertEqual(len(report.checks), 3)
                return [check["ok"] for check in report.checks]

            self.assertEqual(verdict(), [True, True, True])
            self.assertFalse(verdict(lambda s: s.pop("grounding_missing_count"))[0])
            self.assertFalse(verdict(lambda s: s.__setitem__("grounding_missing_rate", 0.0))[0])
            self.assertFalse(verdict(lambda s: s.pop("document_fully_auto_count"))[1])
            self.assertFalse(verdict(lambda s: s.__setitem__("document_auto_rate", 1.0))[1])

    def test_the_reverse_leg_is_not_hardcoded(self):
        from docpipe.acceptance import Report, _check_section65_metrics
        from docpipe.evaluate import evaluate_predictions
        with tempfile.TemporaryDirectory() as temp:
            out_dir, labels_dir = self._write(temp, {"D907": {
                "bs.account_name": ("示例科技有限公司", self._field("示例科技有限公司")),
                "bs.currency": ("CNY", self._field("CNY")),
            }})
            path = out_dir / "D907" / "D907.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["fields"]["bs.currency"]["decision"] = "review"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = evaluate_predictions(out_dir, labels_dir, self.config)
            self.assertEqual(result["summary"]["document_fully_auto_count"], 0)
            report = Report()
            _check_section65_metrics(report, result, out_dir, labels_dir, self.config)
            # 一份全自动文档都没有时，反向腿无从下手，必须报红而不是静默通过。
            self.assertEqual([check["ok"] for check in report.checks], [True, True, False])


class RequiredAcceptRateTests(unittest.TestCase):
    """SCENARIO §6.5 第一行：自动接受率的分母是必填字段。

    这条指标以前拿全部字段当分母：dev 1495 格里 561 格可选（且全部自动接受），
    分母越虚胖门槛越低。测试钉住三件事——可选格不进分母、必填进复核会拉动它、
    整组没有必填格时如实说「不适用」而不是报一个 0 或一个 1。
    """

    def setUp(self):
        self.config = load_config()
        self.required = ["bl.company_name", "bl.legal_representative"]
        self.optional = "bl.entity_type"

    def _write(self, temp, decisions):
        out = Path(temp) / "out"
        labels = Path(temp) / "labels"
        doc_dir = out / "D950"
        doc_dir.mkdir(parents=True)
        labels.mkdir(parents=True)
        fields = {}
        label_fields = {}
        for path, decision in decisions.items():
            fields[path] = {"path": path, "value": "示例", "value_type": "string",
                            "confidence": 0.97, "decision": decision,
                            "grounding": {"page": 1, "bbox": [10, 10, 90, 24]},
                            "extractor_id": "rule_label_v1", "review_reasons": []}
            label_fields[path] = {"value": "示例", "page": 1, "bbox": [10, 10, 90, 24]}
        (doc_dir / "D950.json").write_text(json.dumps({
            "document_id": "D950", "doc_type": "BUSINESS_LICENSE", "fields": fields,
            "source": {"page_count": 1}, "timing": {"total_seconds": 0.5},
        }, ensure_ascii=False), encoding="utf-8")
        (labels / "D950.json").write_text(json.dumps({
            "document_id": "D950", "doc_type": "BUSINESS_LICENSE", "fields": label_fields,
        }, ensure_ascii=False), encoding="utf-8")
        return out, labels

    def _evaluate(self, temp, decisions):
        from docpipe import evaluate
        out, labels = self._write(temp, decisions)
        return evaluate.evaluate_predictions(out, labels, self.config)

    def test_optional_accepts_stay_out_of_the_denominator(self):
        decisions = {self.required[0]: "accept", self.required[1]: "accept",
                     self.optional: "accept"}
        with tempfile.TemporaryDirectory() as temp:
            summary = self._evaluate(temp, decisions)["summary"]
        self.assertEqual(summary["required_field_count"], 2)
        self.assertEqual(summary["required_accept_count"], 2)
        self.assertEqual(summary["auto_accept_rate"], 1.0)
        self.assertEqual(summary["field_count"], 3)
        self.assertEqual(summary["field_accept_rate"], 1.0)

    def test_one_required_review_moves_only_the_required_rate(self):
        decisions = {self.required[0]: "accept", self.required[1]: "review",
                     self.optional: "accept"}
        with tempfile.TemporaryDirectory() as temp:
            summary = self._evaluate(temp, decisions)["summary"]
        self.assertEqual(summary["auto_accept_rate"], 0.5)
        self.assertEqual(summary["field_accept_rate"], 0.6667)
        self.assertNotEqual(summary["auto_accept_rate"], summary["field_accept_rate"])
        self.assertEqual(summary["claim"]["metrics"][0]["denominator_name"],
                         "required_field_count")
        self.assertEqual(summary["claim"]["metrics"][0]["denominator"], 2)

    def test_a_group_without_required_fields_says_not_applicable(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = self._evaluate(temp, {self.optional: "accept"})["summary"]
        row = summary["by_doc_type"][0]
        self.assertEqual(row["required_field_count"], 0)
        self.assertEqual(row["field_accept_rate"], 1.0)
        claim = {item["name"]: item for item in row["claim"]["metrics"]}
        self.assertEqual(claim["auto_accept_rate"]["status"], "not_applicable")
        self.assertEqual(claim["auto_accept_rate"]["population"], 0)

    def test_the_check_recounts_and_bites_on_a_wrong_denominator(self):
        from docpipe.acceptance import Report, _check_section65_accept_rate
        decisions = {self.required[0]: "accept", self.required[1]: "review",
                     self.optional: "accept"}
        with tempfile.TemporaryDirectory() as temp:
            out, labels = self._write(temp, decisions)
            from docpipe import evaluate
            evaluation = evaluate.evaluate_predictions(out, labels, self.config)

            def verdict(mutate=None):
                damaged = copy.deepcopy(evaluation)
                if mutate:
                    mutate(damaged["summary"])
                report = Report()
                _check_section65_accept_rate(report, damaged, out, labels, self.config)
                self.assertEqual(len(report.checks), 2)
                return {item["name"].split(".")[-1]: item["ok"] for item in report.checks}

            clean = verdict()
            self.assertTrue(clean["section65_accept_rate_denominator"], clean)
            self.assertTrue(clean["section65_accept_rate_bites"], clean)
            legacy = verdict(lambda summary: summary.update(auto_accept_rate=0.6667))
            self.assertFalse(legacy["section65_accept_rate_denominator"])
            shrunk = verdict(lambda summary: summary.update(required_field_count=1))
            self.assertFalse(shrunk["section65_accept_rate_denominator"])
            wrong_count = verdict(lambda summary: summary.update(required_accept_count=2))
            self.assertFalse(wrong_count["section65_accept_rate_denominator"])

class CliSurfaceTests(unittest.TestCase):
    """SCENARIO §11：文档里的五条命令必须还能按写的跑，选项带值与否也要对得上。"""

    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "SCENARIO.md").read_text(encoding="utf-8")
        cls.commands = clispec.documented(cls.text)

    def test_section11_lists_the_five_stage_commands(self):
        self.assertEqual([item["module"] for item in self.commands],
                         ["docpipe.synth", "docpipe.parse", "docpipe.run",
                          "docpipe.evaluate", "docpipe.serve"])

    def test_flags_are_read_with_their_arity(self):
        section = ("## 11. 阶段交付物\n\n```bash\n"
                   "python -m docpipe.run --input samples/dev --force\n"
                   "python -m docpipe.serve --out out/holdout --port 8765\n"
                   "```\n")
        commands = clispec.documented(section)
        self.assertEqual([{flag["name"]: flag["value"] for flag in item["flags"]}
                          for item in commands],
                         [{"--input": True, "--force": False},
                          {"--out": True, "--port": True}])

    def test_each_documented_module_exposes_a_parser_with_its_flags(self):
        for item in self.commands:
            parser = clispec.parser_for(item["module"])
            bound = clispec.bound_flags(parser)
            for flag in item["flags"]:
                self.assertIn(flag["name"], bound, item["module"])
                self.assertEqual(bound[flag["name"]], flag["value"], item["module"])

    def test_live_contract_has_no_problems(self):
        result = clispec.audit(self.text)
        self.assertEqual(result["commands"], 5)
        self.assertEqual(result["flags"], 10)
        self.assertEqual(result["problems"], [])

    def test_documented_flag_that_does_not_exist_is_named(self):
        found = clispec.problems([{"module": "docpipe.synth", "command": "doctored",
                                   "flags": [{"name": "--no-such-flag", "value": False}]}])
        self.assertEqual([item["code"] for item in found], ["flag_missing"])
        self.assertIn("--no-such-flag", found[0]["detail"])

    def test_arity_drift_is_named(self):
        found = clispec.problems([{"module": "docpipe.run", "command": "doctored",
                                   "flags": [{"name": "--force", "value": True}]}])
        self.assertEqual([item["code"] for item in found], ["flag_arity"])

    def test_a_documented_module_that_does_not_exist_is_named(self):
        found = clispec.problems([{"module": "docpipe.nosuch", "flags": [],
                                   "command": "doctored"}])
        self.assertEqual([item["code"] for item in found], ["module_missing"])
        self.assertIn("docpipe.nosuch", found[0]["detail"])

    def test_an_empty_command_block_is_named(self):
        self.assertEqual([item["code"] for item in clispec.problems([])],
                         ["section11_unread"])

    def test_capture_stops_at_argparse_and_restores_it(self):
        parser = clispec.parser_for("docpipe.serve")
        self.assertEqual(clispec.bound_flags(parser)["--port"], True)
        probe = clispec.argparse.ArgumentParser()
        probe.add_argument("--x")
        self.assertEqual(probe.parse_args(["--x", "1"]).x, "1")



class _FakeSecondTable:
    def __init__(self, bbox, rows, cols):
        self.bbox = bbox
        self.row_count = rows
        self.col_count = cols


class _FakeSecondPage:
    def __init__(self, tables=None, error=False):
        self.tables = tables or []
        self.error = error

    def find_tables(self):
        if self.error:
            raise RuntimeError("第二通路炸了")
        return type("Found", (), {"tables": self.tables})()


class _FakeSecondDocument:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


@requires("work/parsed/D004.json", "samples/manifest.json")
class TableCrossPathTests(unittest.TestCase):
    """SCENARIO §12.2：第二条表格通路（PyMuPDF find_tables）的对账口径。"""

    BBOX = [35.0, 210.0, 557.0, 628.0]

    def _table(self, rows=2, headers=("a",), columns=("a",)):
        return {"table_id": "p1-t1", "name": "transactions", "page": 1,
                "bbox": list(self.BBOX), "headers": list(headers),
                "columns": list(columns),
                "rows": [{"cells": {name: {} for name in (columns or ("a",))}}
                         for _ in range(rows)]}

    def _second(self, rows=3, cols=1, bbox=None):
        return {"bbox": list(bbox or self.BBOX), "rows": rows, "cols": cols}

    def test_the_header_row_is_part_of_the_row_count(self):
        table = self._table(rows=18, headers=tuple("abcdefgh"), columns=tuple("abcdefgh"))
        self.assertEqual(tablecross.expected_rows(table), 19)

    def test_columns_fall_back_to_the_widest_row(self):
        table = self._table(rows=2)
        table["columns"] = []
        table["rows"] = [{"cells": {"a": {}}}, {"cells": {"a": {}, "b": {}, "c": {}}}]
        self.assertEqual(tablecross.first_columns(table), 3)

    def test_matching_shape_agrees(self):
        verdict = tablecross.compare_table(self._table(rows=2), self._second())
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["verdict"], "agree")

    def test_a_dropped_row_is_named(self):
        verdict = tablecross.compare_table(self._table(rows=1), self._second(rows=3))
        self.assertEqual(verdict["problems"], [tablecross.PROBLEM_ROW_COUNT])

    def test_a_split_column_is_named(self):
        verdict = tablecross.compare_table(self._table(rows=2), self._second(rows=3, cols=2))
        self.assertEqual(verdict["problems"], [tablecross.PROBLEM_COL_COUNT])

    def test_a_moved_table_box_is_named(self):
        moved = self._second(bbox=[55.0, 230.0, 577.0, 648.0])
        verdict = tablecross.compare_table(self._table(rows=2), moved, 0.98)
        self.assertEqual(verdict["problems"], [tablecross.PROBLEM_BBOX])

    def test_the_bbox_gate_is_the_argument_not_a_constant(self):
        hair = self._second(bbox=[35.0, 210.0, 558.0, 628.0])
        self.assertEqual(tablecross.compare_table(self._table(rows=2), hair, 0.98)["problems"], [])
        self.assertEqual(tablecross.compare_table(self._table(rows=2), hair, 0.9999)["problems"],
                         [tablecross.PROBLEM_BBOX])

    def test_a_missing_second_table_is_named(self):
        verdict = tablecross.compare_table(self._table(rows=2), None)
        self.assertEqual(verdict["problems"], [tablecross.PROBLEM_NOT_FOUND])

    def test_scanned_pages_are_skipped_with_a_declared_reason(self):
        parsed = {"document_id": "X", "engine": "rapidocr",
                  "pages": [{"page": 1, "engine": "rapidocr",
                             "tables": [self._table(rows=2)]}]}
        entry = tablecross.measure_document(Path("x.pdf"), parsed,
                                            opener=lambda _: _FakeSecondDocument([_FakeSecondPage()]))
        self.assertEqual(entry["tables"], [])
        self.assertEqual(list(entry["skipped"]),
                         [tablecross.SKIP_PAGE_NOT_TEXT_LAYER])
        self.assertIn(tablecross.SKIP_PAGE_NOT_TEXT_LAYER, tablecross.DECLARED_SKIPS)

    def test_text_pages_are_measured_with_the_injected_second_path(self):
        parsed = {"document_id": "X", "engine": "pymupdf_text",
                  "pages": [{"page": 1, "engine": "pymupdf_text",
                             "tables": [self._table(rows=2)]}]}
        document = _FakeSecondDocument([
            _FakeSecondPage([_FakeSecondTable(self.BBOX, 3, 1)])])
        entry = tablecross.measure_document(Path("x.pdf"), parsed, opener=lambda _: document)
        self.assertEqual([item["verdict"] for item in entry["tables"]], ["agree"])
        self.assertTrue(document.closed)

    def test_a_broken_second_path_is_recorded_not_swallowed(self):
        parsed = {"document_id": "X", "engine": "pymupdf_text",
                  "pages": [{"page": 1, "engine": "pymupdf_text",
                             "tables": [self._table(rows=2)]}]}
        entry = tablecross.measure_document(
            Path("x.pdf"), parsed, opener=lambda _: _FakeSecondDocument([_FakeSecondPage(error=True)]))
        self.assertEqual(entry["tables"][0]["problems"], [tablecross.PROBLEM_SECOND_PATH])

    def test_the_shipped_threshold_comes_from_config(self):
        config = load_config()
        self.assertEqual(tablecross.threshold_from_config(config), 0.98)
        config["parsing"]["table_crosspath"]["bbox_min_iou"] = 0.5
        self.assertEqual(tablecross.threshold_from_config(config), 0.5)

    def test_the_split_measurement_covers_every_text_layer_document(self):
        cache = ROOT / "work" / "parsed" / "D004.json"
        if not cache.exists():
            self.skipTest("work/parsed 缓存缺失")
        payload = tablecross.measure_split(ROOT / "samples" / "dev", ROOT / "work" / "parsed",
                                           ROOT / "samples" / "manifest.json", "dev")
        self.assertEqual(payload["coverage"]["documents_scanned"],
                         payload["coverage"]["text_layer_docs"])
        self.assertEqual(payload["undeclared_skips"], [])
        self.assertEqual(payload["undeclared_problems"], [])
        self.assertGreaterEqual(payload["tables"], 1)


@requires("samples/manifest.json")
class DocTypeContractTests(unittest.TestCase):
    """新增一类文档的接入契约（docpipe.doctypeaudit）与两处静默兜底。"""

    @staticmethod
    def _manifest():
        return json.loads((ROOT / "samples" / "manifest.json").read_text(encoding="utf-8"))

    @staticmethod
    def _entry(payload, doc_type):
        return [item for item in payload["types"] if item["doc_type"] == doc_type][0]

    def test_shipped_contract_is_clean(self):
        config = load_config()
        payload = doctypeaudit.audit(config, self._manifest())
        self.assertEqual(payload["problems_total"], 0, payload["problems"])
        self.assertEqual(payload["type_count"], len(config["doc_types"]))
        # 发票与缴款书只有 schema/抽取/夹具，没有合成版面：跳过要如实写出来。
        self.assertEqual(payload["skipped_types"], ["CUSTOMS_VAT_PAYMENT", "VAT_INVOICE"])
        for entry in payload["types"]:
            if not entry["synth_required"]:
                continue
            self.assertTrue(entry["synth_builder"], entry)
            self.assertGreaterEqual(entry["samples"], 1, entry)

    def test_a_thin_declaration_is_named_code_by_code(self):
        config = copy.deepcopy(load_config())
        config["doc_types"]["SYNTH_THIN"] = {
            "label": "草稿票据", "keywords": {"草稿": 1}, "tables": {},
            "fields": [{"path": "st.only", "label": "唯一", "type": "string",
                        "required": True, "risk": "not_a_tier",
                        "extractor": "tabel_cell", "rules": []}]}
        config["evaluation"]["doc_type_contract"]["require_synth"].append("SYNTH_THIN")
        config["cross_document"]["rules"]["V-X-01"]["paths"].append("st.ghost")
        payload = doctypeaudit.audit(config, self._manifest())
        codes = {problem["code"] for entry in payload["types"]
                 for problem in entry["problems"]}
        codes |= {problem["code"] for problem in payload["details"]}
        self.assertEqual(codes, {doctypeaudit.UNDETECTABLE, doctypeaudit.THIN_REQUIRED,
                                 doctypeaudit.UNKNOWN_RISK, doctypeaudit.EXTRACTOR_UNKNOWN,
                                 doctypeaudit.NO_BUILDER, doctypeaudit.NO_SAMPLES,
                                 doctypeaudit.DEAD_CROSS_PATH})

    def test_every_declared_extractor_is_registered(self):
        config = copy.deepcopy(load_config())
        config["doc_types"]["BUSINESS_LICENSE"]["fields"][0]["extractor"] = "txt"
        payload = doctypeaudit.audit(config, self._manifest())
        codes = [problem["code"] for problem
                 in self._entry(payload, "BUSINESS_LICENSE")["problems"]]
        self.assertIn(doctypeaudit.EXTRACTOR_UNKNOWN, codes)

    def test_table_cell_must_have_a_bracket_path(self):
        config = copy.deepcopy(load_config())
        config["doc_types"]["BANK_STATEMENT"]["fields"].append(
            {"path": "bs.designation", "label": "摘要", "type": "string",
             "required": False, "risk": "free_text", "extractor": "table_cell",
             "table": "transactions", "column": "摘要", "array": "bs.transactions",
             "rules": []})
        payload = doctypeaudit.audit(config, self._manifest())
        details = [problem["detail"] for problem
                   in self._entry(payload, "BANK_STATEMENT")["problems"]]
        self.assertIn("声明 table_cell 但路径没有 [i]", details)

    def test_array_must_match_the_path_prefix(self):
        config = copy.deepcopy(load_config())
        spec = [item for item in config["doc_types"]["BANK_STATEMENT"]["fields"]
                if item["path"] == "bs.transactions[i].summary"][0]
        spec["array"] = "bs.other"
        payload = doctypeaudit.audit(config, self._manifest())
        codes = [problem["code"] for problem
                 in self._entry(payload, "BANK_STATEMENT")["problems"]]
        self.assertIn(doctypeaudit.ARRAY_MISMATCH, codes)

    def test_threshold_follows_config(self):
        config = copy.deepcopy(load_config())
        config["evaluation"]["doc_type_contract"]["min_required_fields"] = 99
        strict = doctypeaudit.audit(config, self._manifest())
        self.assertEqual(strict["problems"]["thin_required"], strict["type_count"])
        config["evaluation"]["doc_type_contract"]["min_required_fields"] = 1
        loose = doctypeaudit.audit(config, self._manifest())
        self.assertNotIn("thin_required", loose["problems"])

    def test_missing_manifest_is_named_not_swallowed(self):
        payload = doctypeaudit.audit(load_config(), None)
        self.assertEqual(payload["problems"].get(doctypeaudit.MANIFEST_MISSING), 1)

    def test_manifest_types_must_be_declared(self):
        payload = doctypeaudit.audit(load_config(), {"documents": [{"doc_type": "MYSTERY"}]})
        self.assertEqual(payload["problems"].get(doctypeaudit.UNKNOWN_MANIFEST_TYPE), 1)

    def test_synth_builders_match_the_declared_synthetic_types(self):
        config = load_config()
        declared = doctypeaudit.options(config)["require_synth"]
        self.assertEqual(sorted(synth.SYNTH_BUILDERS), sorted(declared))
        self.assertEqual(set(declared) - set(config["doc_types"]), set())

    def test_unknown_doc_type_raises_instead_of_becoming_a_credit_report(self):
        synth.DOC_TYPES["X1"] = "MYSTERY_DOC"
        synth.COMPANY_INDEX["X1"] = 0
        try:
            with self.assertRaises(ValueError):
                synth.generate_document("X1", "dev", [{"company_name": "示例"}],
                                        random.Random(1), Path(tempfile.gettempdir()))
        finally:
            synth.DOC_TYPES.pop("X1", None)
            synth.COMPANY_INDEX.pop("X1", None)

    def test_declared_extractor_beats_the_path_shape(self):
        config = copy.deepcopy(load_config())
        moved = [item for item in config["doc_types"]["BANK_STATEMENT"]["fields"]
                 if item["path"] == "bs.transactions[i].txn_date"][0]
        parsed = {"document_id": "T1", "doc_type": "BANK_STATEMENT",
                  "engine": "pymupdf_text",
                  "pages": [{"page": 1, "width": 595, "height": 842, "lines": [],
                             "tables": [{
                                 "table_id": "p1-t1", "name": "transactions", "page": 1,
                                 "bbox": [20, 100, 560, 300],
                                 "headers": ["交易日期", "摘要"],
                                 "columns": ["交易日期", "摘要"], "anomalies": [],
                                 "rows": [{"row_index": 0, "cells": {
                                     "交易日期": {"text": "2025-07-01",
                                                  "bbox": [20, 120, 80, 132],
                                                  "confidence": 0.99},
                                     "摘要": {"text": "往来款",
                                              "bbox": [90, 120, 160, 132],
                                              "confidence": 0.99}}}]}]}]}
        shipped = extract_document(parsed, config)["fields"]
        self.assertEqual(shipped["bs.transactions[0].txn_date"]["value"], "2025-07-01")

        moved["extractor"] = "label_value"
        moved["aliases"] = []
        fields = extract_document(parsed, config)["fields"]
        self.assertNotIn("bs.transactions[0].txn_date", fields)
        self.assertEqual(fields["bs.transactions[i].txn_date"]["status"], "missing")
        self.assertTrue(fields["bs.transactions[i].txn_date"]["extractor_id"]
                        .startswith("label_value"))


@requires("out/dev")
class ParseContractTests(unittest.TestCase):
    """SCENARIO §12.1 统一解析输出的契约（docpipe.parseaudit）。"""

    @staticmethod
    def _config():
        config = load_config()
        section = dict((config.get("evaluation") or {}).get("parse_contract") or {})
        section.update({"min_documents": 1, "min_pages": 1, "min_lines": 1})
        config["evaluation"]["parse_contract"] = section
        return config

    @staticmethod
    def _spec():
        return parseaudit.section121_spec((ROOT / "SCENARIO.md").read_text(encoding="utf-8"))

    @staticmethod
    def _record():
        import pymupdf  # noqa: F401  （与解析层同一套坐标约定，无需真的渲染）
        from docpipe import parse as _parse
        return {
            "document_id": "T001", "page_count": 1, "engine": "pymupdf_text",
            "parser_version": _parse.PARSER_VERSION,
            "pages": [{
                "page": 1, "width": 595.0, "height": 842.0, "engine": "pymupdf_text",
                "lines": [{"text": "户名：示例", "bbox": [10.0, 10.0, 100.0, 20.0],
                           "confidence": 0.99}],
                "tables": [{
                    "table_id": "p1-t1", "bbox": [5.0, 100.0, 500.0, 300.0],
                    "columns": [{"header": "金额", "center": 50.0, "radius": 20.0}],
                    "rows": [{"row_index": 0, "page": 1, "cells": {
                        "金额": {"text": "1.00", "bbox": [5.0, 100.0, 40.0, 110.0],
                                 "confidence": 0.9}}}],
                }],
            }],
        }

    def _codes(self, payload):
        return sorted({item["code"] for item in payload["details"]})

    def test_spec_comes_from_the_document(self):
        spec = self._spec()
        paths = parseaudit.contract_paths(spec)
        self.assertEqual(paths[0], "document_id")
        self.assertIn("pages[].blocks", paths)
        self.assertIn("engine.name", paths)
        self.assertGreaterEqual(len(paths), 25)
        self.assertEqual(len(paths), len(set(paths)))

    def test_shipped_deviations_are_declared_and_justified(self):
        known = set(parseaudit.all_paths(self._spec()))
        for path, entry in parseaudit.DEVIATIONS.items():
            self.assertIn(path, known, path)
            self.assertGreaterEqual(len(entry.get("reason") or ""), 20, path)

    def test_a_clean_record_has_no_problems(self):
        payload = parseaudit.audit([self._record()], self._spec(), self._config())
        self.assertEqual(payload["details"], [])
        self.assertEqual(payload["contract"]["present"], 9, payload["contract"])

    def test_deviation_is_stale_once_the_document_shape_returns(self):
        record = self._record()
        record["pages"][0]["blocks"] = [{"text": "x"}]
        payload = parseaudit.audit([record], self._spec(), self._config())
        self.assertEqual(self._codes(payload), [parseaudit.DEVIATION_STALE])

    def test_deviation_is_broken_when_the_replacement_disappears(self):
        record = self._record()
        record["pages"][0]["lines"] = []
        payload = parseaudit.audit([record], self._spec(), self._config())
        codes = self._codes(payload)
        self.assertIn(parseaudit.DEVIATION_BROKEN, codes)
        self.assertNotIn(parseaudit.DEVIATION_STALE, codes)

    def test_missing_contract_key_when_nobody_declared_it(self):
        record = self._record()
        record["pages"][0]["tables"][0].pop("table_id")
        payload = parseaudit.audit([record], self._spec(), self._config())
        self.assertEqual(self._codes(payload),
                         sorted([parseaudit.BAD_TABLE, parseaudit.MISSING_CONTRACT_KEY]))

    def test_an_unlisted_deviation_path_is_a_typo(self):
        polluted = dict(parseaudit.DEVIATIONS)
        polluted["pages[].区块"] = {"implemented": "", "reason": "打错字的一层"}
        with mock.patch.object(parseaudit, "DEVIATIONS", polluted):
            payload = parseaudit.audit([self._record()], self._spec(), self._config())
        self.assertEqual(self._codes(payload), [parseaudit.DEVIATION_UNLISTED])

    def test_invariants_bite_separately(self):
        counted = self._record()
        counted["page_count"] = 3
        self.assertEqual(self._codes(parseaudit.audit([counted], self._spec(), self._config())),
                         [parseaudit.BAD_PAGE_COUNT])

        moved = self._record()
        moved["pages"][0]["lines"][0]["bbox"] = [-50.0, -50.0, 10.0, 10.0]
        self.assertEqual(self._codes(parseaudit.audit([moved], self._spec(), self._config())),
                         [parseaudit.LINE_BBOX_OUT])

        renumbered = self._record()
        renumbered["pages"][0]["tables"][0]["rows"][0]["row_index"] = 7
        self.assertEqual(self._codes(parseaudit.audit([renumbered], self._spec(),
                                                      self._config())),
                         [parseaudit.BAD_ROW_INDEX])

        foreign = self._record()
        foreign["pages"][0]["tables"][0]["rows"][0]["cells"]["摘要"] = {
            "text": "x", "bbox": [10.0, 100.0, 40.0, 110.0], "confidence": 0.9}
        self.assertEqual(self._codes(parseaudit.audit([foreign], self._spec(), self._config())),
                         sorted([parseaudit.CELL_NOT_IN_COLUMNS]))

    def test_coverage_floor_refuses_an_empty_victory(self):
        config = self._config()
        config["evaluation"]["parse_contract"].update(
            {"min_documents": 5, "min_pages": 20, "min_lines": 100})
        payload = parseaudit.audit([self._record()], self._spec(), config)
        self.assertEqual(payload["problems"], {parseaudit.THIN_CORPUS: 3})

    def test_stale_parser_version_is_reported_not_blamed(self):
        record = self._record()
        record["parser_version"] = "0.0.1"
        payload = parseaudit.audit([record], self._spec(), self._config())
        self.assertEqual(payload["details"], [])
        self.assertEqual(payload["stale_versions"], ["0.0.1"])
        self.assertNotEqual(payload["current_version"], "0.0.1")

    def test_audit_writes_nothing_under_out(self):
        out_dir = ROOT / "out"
        before = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(out_dir.rglob("*")) if path.is_file()}
        parseaudit.audit([self._record()], self._spec(), self._config())
        after = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(out_dir.rglob("*")) if path.is_file()}
        self.assertEqual(before, after)


@requires("out/dev")
class ModelContractTests(unittest.TestCase):
    def test_uncalibrated_audit_rejects_a_changed_score_or_basis(self):
        from docpipe import modelaudit
        from docpipe.trust import heuristic_confidence

        config = load_config()
        field = {"path": "bl.company_name", "signals": {"ocr_conf": 0.8},
                 "confidence_basis": "heuristic", "calibration_samples": 0}
        field["confidence"] = round(min(0.99, heuristic_confidence(field, config)), 3)
        document = {"document_id": "demo", "doc_type": "BUSINESS_LICENSE",
                    "fields": {field["path"]: field}}
        opts = {**modelaudit.options(config), "min_delivered_fields": 1}
        clean = modelaudit.audit([document], None, config, opts, allow_uncalibrated=True)
        self.assertEqual(clean["problems"], {})
        field["confidence"] = 0.99
        field["confidence_basis"] = "calibrated"
        field["calibration_samples"] = 5
        altered = modelaudit.audit([document], None, config, opts, allow_uncalibrated=True)
        self.assertEqual(set(altered["problems"]), {
            modelaudit.CONFIDENCE_UNEXPLAINED, modelaudit.BASIS_MISMATCH, modelaudit.SAMPLES_MISMATCH,
        })

    """SCENARIO §6.2 融合与校准的产物侧对账（docpipe.modelaudit）。"""

    def _fixture(self, temp, **overrides):
        """一份与当前配置同形同序的最小产物：系数全 0、截距 4.0，isotonic 为恒等。"""
        config = copy.deepcopy(load_config())
        config["calibration"]["artifact_contract"]["min_delivered_fields"] = 1
        versions = modelaudit.current_versions()
        names = feature_names(config)
        artifact = {
            "calibrate_version": versions["calibrate_version"],
            "trust_version": versions["trust_version"],
            "method": config["calibration"]["method"],
            "features": names,
            "coef": [0.0] * len(names),
            "intercept": 4.0,
            "isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "sample_grouping": "field_auto",
            "field_samples": {"bs.account_name": 500},
            "group_samples": {"BANK_STATEMENT": 500},
            "cv_folds": [{"train_docs": ["T1"], "test_docs": ["T1"]}] * 5,
        }
        artifact.update(overrides)
        path = Path(temp) / "calibration.json"
        path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
        config["calibration"]["artifact"] = str(path)
        return config, artifact

    def _document(self, config, path="bs.account_name", doc_type="BANK_STATEMENT"):
        field = {
            "path": path, "label": "户名", "value": "示例公司", "value_type": "string",
            "required": True, "risk": "party_name",
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "grounding_present": 1},
        }
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 200, "height": 200}]}
        document = {"document_id": "T1", "doc_type": doc_type}
        decided = decide_field(field, document, parsed, config, [], Calibrator(config))
        for key in ("confidence", "confidence_basis", "calibration_samples"):
            field[key] = decided[key]
        document["fields"] = {path: field}
        return document

    @staticmethod
    def _codes(payload):
        return sorted({item["code"] for item in payload["details"]})

    def test_audit_reproduces_the_decided_confidence(self):
        """判决写进产物的数，产品侧能独立重算出来——这是「这个数凭什么」的全部内容。"""
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp)
            document = self._document(config)
            field = document["fields"]["bs.account_name"]
            self.assertEqual(field["confidence_basis"], "calibrated")
            payload = modelaudit.audit([document], artifact, config)
            self.assertEqual(payload["problems_total"], 0, payload["details"])
            self.assertEqual(payload["totals"]["fields"], 1)
            self.assertEqual(payload["notes"], {})

    def test_tampered_confidence_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp)
            document = self._document(config)
            field = document["fields"]["bs.account_name"]
            field["confidence"] = 0.0 if field["confidence"] > 0.5 else 0.99
            payload = modelaudit.audit([document], artifact, config)
            self.assertEqual(payload["problems"], {modelaudit.CONFIDENCE_UNEXPLAINED: 1})

    def test_missing_confidence_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp)
            document = self._document(config)
            document["fields"]["bs.account_name"].pop("confidence")
            payload = modelaudit.audit([document], artifact, config)
            # 没有 confidence 的格子不进计数，于是这份单文档语料也够不上下限。
            self.assertEqual(payload["problems"].get(modelaudit.CONFIDENCE_MISSING), 1)
            self.assertEqual(payload["details"][0]["code"], modelaudit.CONFIDENCE_MISSING)
            self.assertEqual(payload["problems"].get(modelaudit.THIN_CORPUS), 1)

    def test_truncated_coef_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ = self._fixture(temp)
            artifact = {"features": feature_names(config), "coef": [0.0] * 3,
                        "isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
            payload = modelaudit.artifact_problems(artifact, config)
            self.assertEqual(self._codes(payload), [modelaudit.COEF_SHAPE])

    def test_reordered_features_are_named(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ = self._fixture(temp)
            names = feature_names(config)
            names[0], names[1] = names[1], names[0]
            artifact = {"features": names, "coef": [0.0] * len(names),
                        "isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
            payload = modelaudit.artifact_problems(artifact, config)
            self.assertEqual(self._codes(payload), [modelaudit.FEATURE_ORDER])

    def test_narrow_feature_vector_is_named_with_its_coef_width(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ = self._fixture(temp)
            names = feature_names(config)
            artifact = {"features": names[:-1], "coef": [0.0] * len(names),
                        "isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
            payload = modelaudit.artifact_problems(artifact, config)
            self.assertEqual(set(self._codes(payload)),
                             {modelaudit.FEATURE_ORDER, modelaudit.COEF_SHAPE})

    def test_calibrator_refuses_mismatched_artifacts(self):
        """`zip()` 会静默截断：尺寸对不上必须当场拒绝，而不是少算最后几维。"""
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp)
            names = feature_names(config)
            cases = [
                dict(artifact, coef=artifact["coef"][:-1]),
                dict(artifact, features=list(names[1:]) + [names[0]]),
            ]
            for index, broken in enumerate(cases):
                target = Path(temp) / f"broken_{index}.json"
                target.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
                probe = copy.deepcopy(config)
                probe["calibration"]["artifact"] = str(target)
                with self.assertRaises(ValueError):
                    Calibrator(probe)

    def test_artifact_without_features_is_named_not_swallowed(self):
        """老产物没有 `features`：Calibrator 无从比对必须放行，审计必须点名。"""
        with tempfile.TemporaryDirectory() as temp:
            config = copy.deepcopy(load_config())
            target = Path(temp) / "legacy.json"
            target.write_text(json.dumps({"coef": [0.0], "intercept": 4.0,
                                           "isotonic": {"x": [0.0, 1.0],
                                                        "y": [0.0, 1.0]}}),
                              encoding="utf-8")
            config["calibration"]["artifact"] = str(target)
            self.assertIsNotNone(Calibrator(config).artifact)
            payload = modelaudit.artifact_problems(
                {"coef": [0.0], "intercept": 4.0,
                 "isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}, config)
            self.assertEqual(self._codes(payload), [modelaudit.FEATURES_MISSING])

    def test_non_monotone_isotonic_is_named(self):
        with tempfile.TemporaryDirectory() as temp:
            config, _ = self._fixture(temp)
            artifact = {"features": feature_names(config),
                        "coef": [0.0] * len(feature_names(config)),
                        "isotonic": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.9, 0.1]}}
            payload = modelaudit.artifact_problems(artifact, config)
            self.assertEqual(self._codes(payload), [modelaudit.ISOTONIC_NOT_MONOTONE])

    def test_version_and_method_skew_are_reported_not_blamed(self):
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp, calibrate_version="0.0.1",
                                             method="logistic_isotonic")
            payload = modelaudit.audit([self._document(config)], artifact, config)
            self.assertEqual(payload["problems_total"], 0, payload["details"])
            self.assertEqual(payload["notes"].get(modelaudit.VERSION_SKEW), 1)
            self.assertEqual(payload["notes"].get(modelaudit.METHOD_SKEW), 1)

    def test_fold_count_and_oof_coverage_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(
                temp, cv_folds=[{"train_docs": ["T1", "T2"], "test_docs": ["T1"]}])
            payload = modelaudit.audit([self._document(config)], artifact, config)
            self.assertEqual(payload["problems_total"], 0, payload["details"])
            self.assertEqual(payload["notes"].get(modelaudit.FOLD_COUNT_BELOW_TARGET), 1)
            self.assertEqual(payload["notes"].get(modelaudit.OOF_COVERAGE_BELOW_TARGET), 1)
            self.assertEqual(payload["coverage"]["covered"], 1)
            self.assertEqual(payload["coverage"]["documents"], 2)

    def test_small_sample_field_uses_group_count_and_the_gap_is_reported(self):
        """§6.2 第 5 条的边界：`field_auto` 换用组样本数，上限分支就走不到。"""
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp, field_samples={"bs.account_name": 5})
            document = self._document(config)
            field = document["fields"]["bs.account_name"]
            self.assertEqual(field["confidence_basis"], "calibrated")
            self.assertEqual(field["calibration_samples"], 500)
            payload = modelaudit.audit([document], artifact, config)
            self.assertEqual(payload["problems_total"], 0, payload["details"])
            self.assertEqual(payload["notes"].get(modelaudit.SAMPLE_COUNTING_GAP), 1)
            self.assertEqual(payload["totals"]["low_sample_on_group"], 1)
            by_field = dict(artifact, sample_grouping="field")
            switched = modelaudit.audit([document], by_field, config)
            self.assertIn(modelaudit.BASIS_MISMATCH, self._codes(switched))
            self.assertEqual(payload["totals"]["fields"], switched["totals"]["fields"])

    def test_audit_writes_nothing_under_out(self):
        out_dir = ROOT / "out"
        if not out_dir.exists():
            self.skipTest("no out/ directory in this checkout")
        before = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(out_dir.rglob("*")) if path.is_file()}
        with tempfile.TemporaryDirectory() as temp:
            config, artifact = self._fixture(temp)
            modelaudit.audit([self._document(config)], artifact, config)
        after = {str(path.relative_to(out_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(out_dir.rglob("*")) if path.is_file()}
        self.assertEqual(before, after)


class DocumentRefusalTests(unittest.TestCase):
    """§6.4 的 UNKNOWN_DOC_TYPE 过去只挂在字段上，而类型认不出时连一个字段都没有，
    于是「强制复核」退化成零字段零原因的空文档：汇总里和「全都自动接受」长得一样，
    单文件网页版甚至会把整份藏掉。拒答必须落在文档上。"""

    def _parsed(self):
        return {"page_count": 1, "quality": {"parse_score": 0.99},
                "pages": [{"page": 1, "width": 200, "height": 200}]}

    def _assess(self, doc, config=None):
        config = copy.deepcopy(config or load_config())
        config["calibration"]["artifact"] = None
        return config, assess([doc], {doc["document_id"]: self._parsed()}, config)[0]

    def test_unknown_type_gets_a_document_level_refusal(self):
        config, doc = self._assess({"document_id": "U1", "doc_type": "UNKNOWN",
                                    "doc_type_confidence": 0.41, "fields": {},
                                    "source": "U1.pdf"})
        refusal = doc["document_refusal"]
        self.assertEqual(refusal["code"], "UNKNOWN_DOC_TYPE")
        self.assertEqual(refusal["field_count"], 0)
        self.assertEqual(refusal["doc_type_confidence"], 0.41)
        self.assertEqual(refusal["action"],
                         config["validation"]["review_actions"]["UNKNOWN_DOC_TYPE"])

    def test_recognised_type_with_no_fields_is_not_called_unknown(self):
        """空文档不等于未识别：认得出的类型零字段是另一件事，别给它挂错码。"""
        _, doc = self._assess({"document_id": "K1", "doc_type": "BUSINESS_LICENSE",
                               "fields": {}, "source": "K1.pdf"})
        self.assertNotIn("document_refusal", doc)

    def test_action_wording_comes_from_config(self):
        config = copy.deepcopy(load_config())
        config["validation"]["review_actions"]["UNKNOWN_DOC_TYPE"] = "先用别的工具确认这是什么"
        _, doc = self._assess({"document_id": "U2", "doc_type": "UNKNOWN", "fields": {},
                               "source": "U2.pdf"}, config)
        self.assertEqual(doc["document_refusal"]["action"], "先用别的工具确认这是什么")

    def test_refusal_reaches_the_payload_and_the_final_summary(self):
        from docpipe import run as run_mod

        config, doc = self._assess({"document_id": "U3", "doc_type": "UNKNOWN",
                                    "doc_type_confidence": 0.2, "fields": {},
                                    "source": "U3.pdf"})
        payload = run_mod._document_payload(doc, self._parsed(), Path("U3.pdf"), None, config)
        self.assertEqual(payload["document_refusal"]["code"], "UNKNOWN_DOC_TYPE")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "U3").mkdir()
            (root / "U3" / "U3.json").write_text(json.dumps(payload, ensure_ascii=False),
                                                encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({
                "run": {"pipeline_version": "fixture", "started_at": "2026-01-01T00:00:00+08:00"},
                "documents": [{"document_id": "U3", "doc_type": "UNKNOWN",
                               "document_refusal": payload["document_refusal"],
                               "source": "U3.pdf", "page_count": 1, "path": "U3/U3.json",
                               "field_count": 0, "accept_count": 0, "review_count": 0}],
            }, ensure_ascii=False), encoding="utf-8")
            _, final_summary, _ = load_final_documents(root, config)
            html = Path(build_report(root, config=config)["path"]).read_text(encoding="utf-8")
        self.assertEqual(final_summary["documents"][0]["document_refusal"]["code"],
                         "UNKNOWN_DOC_TYPE")
        self.assertEqual(final_summary["overall"]["refused_document_count"], 1)
        # 报表是按 doc.refusal 先放行的，否则零字段的文档会被 visible 过滤整份藏掉。
        self.assertIn("doc.refusal ||", (ROOT / "docpipe" / "report.py").read_text(encoding="utf-8"))
        self.assertIn('"refusal"', html)
        self.assertIn("文档类型未识别", html)

    def test_review_ui_reads_the_document_refusal(self):
        app = (ROOT / "docpipe" / "web" / "app.js").read_text(encoding="utf-8")
        style = (ROOT / "docpipe" / "web" / "style.css").read_text(encoding="utf-8")
        self.assertIn("state.doc.document_refusal", app)
        self.assertIn(".doc-refusal", style)


class AbstainReachabilityTests(unittest.TestCase):
    """§6.4 的 13 个原因码里，AMBIGUOUS_UNIT / AMBIGUOUS_DATE 在单测里一次都没被点着过。
    它们不是不可达，只是没人证明过——「能发出来」这件事本身要让程序回答。"""

    def _decide(self, value, meta, config=None):
        config = copy.deepcopy(config or load_config())
        config["calibration"]["artifact"] = None
        field = {
            "path": "is.revenue", "label": "营业收入", "value": value,
            "value_type": "decimal", "required": False, "risk": "monetary",
            "grounding": {"page": 1, "bbox": [10, 10, 80, 20]},
            "signals": {"ocr_conf": 0.99, "rule_exact": 1.0, "format_valid": 1.0,
                        "table_score": 1.0, "cross_field_score": 1.0,
                        "grounding_present": 1},
            "meta": meta, "decision": "review", "validators": [], "review_reasons": [],
        }
        parsed = {"page_count": 1, "quality": {"parse_score": 0.99},
                  "pages": [{"page": 1, "width": 200, "height": 200}]}
        doc = {"document_id": "R1", "doc_type": "INCOME_STATEMENT", "fields": {}}
        return decide_field(field, doc, parsed, config, [], Calibrator(config))

    def test_missing_unit_line_reaches_ambiguous_unit(self):
        config = load_config()
        value, status, meta = _normalize("1,234.56", {"type": "decimal",
                                                      "unit_from_document": True},
                                         config, {"unit": None, "unit_factor": None})
        self.assertEqual(status, "ok")
        self.assertTrue(meta.get("unit_unknown"))
        result = self._decide(value, meta)
        self.assertEqual(result["decision"], "review")
        self.assertIn("AMBIGUOUS_UNIT", result["review_reasons"])

    def test_incomplete_date_reaches_ambiguous_date(self):
        config = load_config()
        value, status, meta = _normalize("2026年3月", {"type": "date"}, config, {})
        self.assertEqual(status, "parse_failed")
        self.assertEqual(meta.get("parse_error"), "date")
        result = self._decide(value, meta)
        self.assertIn("AMBIGUOUS_DATE", result["review_reasons"])


class ThresholdContractTests(unittest.TestCase):
    """SCENARIO §6.3 风险档阈值契约（docpipe.riskcontract）与 thresholds 的两头对账。"""

    def test_section_63_is_readable(self):
        spec = riskcontract.section63_spec()
        self.assertEqual(len(spec["tiers"]), 8, spec)
        self.assertAlmostEqual(spec["floor"], 0.90)
        self.assertEqual(spec["tiers"]["monetary"]["threshold"], 0.97)

    def test_shipped_config_matches_the_contract(self):
        payload = riskcontract.audit(load_config(), riskcontract.section63_spec())
        self.assertEqual(payload["problems_total"], 0, payload["details"])
        self.assertGreaterEqual(payload["members"]["resolved"], 20)
        self.assertGreaterEqual(payload["checked_fields"], 40)
        self.assertEqual(payload["floor"], payload["config_floor"])

    def test_registration_authority_sits_in_the_contract_tier(self):
        # 本轮修掉的真偏航：登记机关曾按 optional_low_risk 0.93 放行，契约要求 0.95。
        spec = field_spec("BUSINESS_LICENSE", "bl.registration_authority")
        self.assertIsNotNone(spec)
        self.assertEqual(spec["risk"], "free_text")

    def test_a_lowered_tier_threshold_is_named(self):
        config = copy.deepcopy(load_config())
        config["thresholds"]["risk_tiers"]["date"] = 0.90
        payload = riskcontract.audit(config)
        self.assertEqual(payload["problems"], {"threshold_mismatch": 1})
        self.assertEqual(payload["details"][0]["where"], "thresholds.risk_tiers.date")

    def test_a_field_below_its_contract_tier_is_named(self):
        config = copy.deepcopy(load_config())
        target = [item for item in config["doc_types"]["BUSINESS_LICENSE"]["fields"]
                  if item["path"] == "bl.company_name"][0]
        target["risk"] = "free_text"
        payload = riskcontract.audit(config)
        self.assertEqual(payload["problems"], {"member_weaker": 1})
        self.assertEqual(payload["details"][0]["where"],
                         "BUSINESS_LICENSE.bl.company_name")

    def test_a_field_in_an_undeclared_tier_is_named(self):
        config = copy.deepcopy(load_config())
        target = [item for item in config["doc_types"]["BUSINESS_LICENSE"]["fields"]
                  if item["path"] == "bl.company_name"][0]
        target["risk"] = "mystery_tier"
        payload = riskcontract.audit(config)
        self.assertEqual(payload["problems"], {"unknown_risk_member": 1})

    def test_tier_sets_are_compared_both_ways(self):
        missing = copy.deepcopy(riskcontract.section63_spec())
        missing["tiers"]["bonus_risk"] = {"threshold": 0.90, "members": "币种"}
        unlisted = copy.deepcopy(load_config())
        unlisted["thresholds"]["risk_tiers"]["bonus_risk"] = 0.90
        self.assertEqual(riskcontract.audit(load_config(), missing)["problems"],
                         {"tier_missing": 1})
        self.assertEqual(riskcontract.audit(unlisted)["problems"],
                         {"tier_unlisted": 1})

    def test_floor_and_date_rules_bite(self):
        config = copy.deepcopy(load_config())
        config["thresholds"]["global_min_confidence"] = 0.85
        self.assertEqual(riskcontract.audit(config)["problems"], {"floor_mismatch": 1})

        config = copy.deepcopy(load_config())
        moved = [item for item in config["doc_types"]["BUSINESS_LICENSE"]["fields"]
                 if item["path"] == "bl.establishment_date"][0]
        moved["risk"] = "free_text"
        payload = riskcontract.audit(config)
        self.assertEqual(payload["problems"], {"date_tier_not_universal": 1})
        self.assertEqual(payload["details"][0]["where"],
                         "BUSINESS_LICENSE.bl.establishment_date")

    def test_names_resolve_exactly_before_containment(self):
        # 「户名」若不按精确名先解析，包含匹配会把它牵到「对方户名」（free_text）。
        index = riskcontract.field_index(load_config())
        hits = riskcontract.resolve("户名", index)
        self.assertEqual([item["path"] for item in hits], ["bs.account_name"])

    def test_stale_exception_is_named(self):
        original = riskcontract.UNRESOLVED_OK
        stale_map = {tier: dict(items) for tier, items in original.items()}
        stale_map.setdefault("identity", {})["统一社会信用代码"] = "临时登记"
        riskcontract.UNRESOLVED_OK = stale_map
        try:
            payload = riskcontract.audit(load_config())
        finally:
            riskcontract.UNRESOLVED_OK = original
        self.assertEqual(payload["problems"], {"member_stale_exception": 1})

    def test_undeclared_problem_code_raises(self):
        config = copy.deepcopy(load_config())
        config["thresholds"]["risk_tiers"]["monetary"] = 0.90
        original = riskcontract.DECLARED_PROBLEMS
        riskcontract.DECLARED_PROBLEMS = frozenset()
        try:
            with self.assertRaises(ValueError):
                riskcontract.audit(config)
        finally:
            riskcontract.DECLARED_PROBLEMS = original


class GateContractTests(unittest.TestCase):
    """SCENARIO §1.3 成功定义 / §10.3 阶段门槛：五个门槛数在配置里的那一份要逐值相等。"""

    def _audit(self, probe=None, success=None, stages=None):
        return gatecontract.audit(copy.deepcopy(probe or load_config()),
                                  success, stages)

    def _codes(self, probe=None, success=None, stages=None):
        return sorted({item["code"] for item in
                       self._audit(probe, success, stages)["details"]})

    def test_success_numbers_are_readable(self):
        numbers = gatecontract.success_spec()["numbers"]
        self.assertEqual(numbers, {"max_accepted_error_rate": 0.02,
                                   "max_monetary_accepted_error_rate": 0.01,
                                   "max_accepted_grounding_missing": 0,
                                   "min_grounding_hit_rate": 0.98,
                                   "min_auto_accept_rate": 0.60})

    def test_stage_table_is_readable(self):
        stages = gatecontract.stage_spec()["stages"]
        self.assertEqual(sorted(stages), [0, 1, 2, 3, 4])
        self.assertEqual(stages[3]["numbers"],
                         {"max_accepted_error_rate": 0.02,
                          "max_monetary_accepted_error_rate": 0.01,
                          "max_accepted_grounding_missing": 0})
        self.assertEqual(stages[4]["numbers"],
                         {"max_accepted_error_rate": 0.02,
                          "min_auto_accept_rate": 0.60})

    def test_shipped_config_matches_the_documents(self):
        payload = self._audit()
        self.assertEqual(payload["problems"], {})
        self.assertEqual(payload["metrics"], 5)
        self.assertEqual(payload["numbers_read"], 5)
        self.assertEqual(payload["gates_compared"], 5)
        self.assertEqual(payload["stages"], 5)

    def test_a_relaxed_gate_is_named(self):
        config = copy.deepcopy(load_config())
        config["acceptance"]["min_auto_accept_rate"] = 0.50
        payload = self._audit(config)
        self.assertEqual(payload["problems"], {"gate_mismatch": 1})
        self.assertEqual(payload["details"][0]["where"], "acceptance.min_auto_accept_rate")

    def test_a_missing_gate_key_is_named(self):
        config = copy.deepcopy(load_config())
        config["acceptance"].pop("min_grounding_hit_rate")
        payload = self._audit(config)
        self.assertEqual(payload["problems"], {"gate_missing": 1})
        self.assertEqual(payload["details"][0]["where"], "acceptance.min_grounding_hit_rate")

    def test_the_two_sections_must_agree(self):
        success = gatecontract.success_spec()
        success["numbers"]["min_auto_accept_rate"] = 0.61
        payload = self._audit(success=success)
        self.assertEqual(payload["problems"], {"doc_conflict": 1, "gate_mismatch": 1})
        self.assertEqual(payload["details"][0]["where"], "§10.3 阶段 4 vs §1.3")

    def test_a_missing_stage_row_is_named(self):
        stages = gatecontract.stage_spec()
        stages["stages"].pop(3)
        config = copy.deepcopy(load_config())
        config["evaluation"]["gate_contract"]["min_stages"] = 4
        payload = self._audit(config, stages=stages)
        self.assertEqual(payload["problems"], {"stage_missing": 1})
        self.assertEqual(payload["details"][0]["where"], "SCENARIO.md §10.3 阶段 3")

    def test_a_thin_stage_section_is_named(self):
        config = copy.deepcopy(load_config())
        config["evaluation"]["gate_contract"]["min_stages"] = 6
        self.assertEqual(self._codes(config), ["stage_section_thin"])

    def test_a_gate_nobody_reads_is_named(self):
        original = gatecontract.READERS
        gatecontract.READERS = ()
        try:
            payload = self._audit()
        finally:
            gatecontract.READERS = original
        self.assertEqual(payload["problems"], {"gate_unread": 5})

    def test_gate_values_are_exact_not_rounded(self):
        config = copy.deepcopy(load_config())
        config["acceptance"]["min_auto_accept_rate"] = 0.6001
        self.assertEqual(self._codes(config), ["gate_mismatch"])

    def test_undeclared_problem_code_raises(self):
        config = copy.deepcopy(load_config())
        config["acceptance"]["min_auto_accept_rate"] = 0.50
        original = gatecontract.DECLARED_PROBLEMS
        gatecontract.DECLARED_PROBLEMS = frozenset()
        try:
            with self.assertRaises(ValueError):
                gatecontract.audit(config)
        finally:
            gatecontract.DECLARED_PROBLEMS = original


class ServeBrowserTests(unittest.TestCase):
    def test_browser_opens_actual_port_after_collision(self):
        from docpipe import serve
        server = mock.Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with mock.patch.object(serve, "make_server", side_effect=[OSError("busy"), server]), \
                mock.patch("webbrowser.open") as browser, contextlib.redirect_stdout(io.StringIO()):
            serve.main(["--port", "8765", "--open-browser"])
        browser.assert_called_once_with("http://127.0.0.1:8766")
        server.server_close.assert_called_once()

    def test_browser_is_opt_in(self):
        from docpipe import serve
        server = mock.Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with mock.patch.object(serve, "make_server", return_value=server), \
                mock.patch("webbrowser.open") as browser, contextlib.redirect_stdout(io.StringIO()):
            serve.main(["--port", "8765"])
        browser.assert_not_called()
        server.server_close.assert_called_once()

    def test_no_browser_when_all_ports_are_busy(self):
        from docpipe import serve
        with mock.patch.object(serve, "make_server", side_effect=OSError("busy")), \
                mock.patch("webbrowser.open") as browser:
            with self.assertRaises(SystemExit):
                serve.main(["--port", "8765", "--open-browser"])
        browser.assert_not_called()
