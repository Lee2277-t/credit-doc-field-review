"""Local review UI server for an out/ directory."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import pymupdf as fitz

from .config import ROOT, load_config, resolve_path
from .export import export_zip
from .importer import (allowed_suffixes, process_inbox, remove_document,
                       store_upload, user_out_dir, zip_allowed)
from .parse import normalized_source, open_document
from .report import build_report, doc_type_labels
from .reviews import action_vocab, payload_vocab, read_review_ledger
from .run import PIPELINE_VERSION
from .textutil import now_iso

WEB_DIR = ROOT / "docpipe" / "web"
ZIP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.zip$")


def _needs_newline(path: Path) -> bool:
    """台账的最后一个字节不是换行时，续写前必须先补一个换行。"""
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) not in (b"\n", b"\r")


class ReviewHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR lets two processes bind the same port; disable it so
    # port scanning actually detects a busy port.
    allow_reuse_address = False
    allow_reuse_port = False


def _empty_summary(out_dir: Path) -> Dict:
    return {
        "run": {
            "started_at": None,
            "finished_at": None,
            "pipeline_version": PIPELINE_VERSION,
            "input_dir": str(out_dir),
            "document_count": 0,
        },
        "overall": {
            "field_count": 0, "accept_count": 0, "review_count": 0,
            "required_field_count": 0, "required_accept_count": 0,
            "auto_accept_rate": 0.0, "field_accept_rate": 0.0, "review_rate": 0.0,
            "avg_confidence": 0.0,
            "grounding_missing_count": 0, "hard_fail_count": 0,
            "reviewed_count": 0, "reviewed_accept_count": 0,
            "reviewed_corrected_count": 0, "pending_review_count": 0,
            "correction_rate": 0.0, "document_seconds_total": 0.0,
            "avg_document_seconds": 0.0, "avg_page_seconds": 0.0,
        },
        "by_doc_type": [],
        "by_field_path": [],
        "documents": [],
    }


class ReviewStore:
    def __init__(self, out_dir: Path, config: Dict):
        self.config = config
        self._lock = threading.RLock()
        self.workspaces = []
        for item in config.get("ui", {}).get("workspaces") or []:
            self.workspaces.append({
                "id": item["id"],
                "label": item.get("label", item["id"]),
                "out": resolve_path(item["out"], config).resolve(),
            })
        self.set_out_dir(out_dir)

    def set_out_dir(self, out_dir: Path) -> None:
        with self._lock:
            self.out_dir = Path(out_dir).resolve()
            self._doc_cache: Dict[str, Dict] = {}
            self._report: Optional[Dict] = None
            summary_path = self.out_dir / "summary.json"
            if summary_path.exists():
                self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                self.summary = _empty_summary(self.out_dir)
            self.documents = {item["document_id"]: item
                              for item in self.summary.get("documents", [])}

    def workspace_payload(self) -> Dict:
        with self._lock:
            current = next((item for item in self.workspaces
                            if item["out"] == self.out_dir), None)
            items = []
            for workspace in self.workspaces:
                if workspace["out"] == self.out_dir:
                    count = len(self.documents)
                else:
                    count = 0
                    summary_path = workspace["out"] / "summary.json"
                    if summary_path.exists():
                        try:
                            count = len(json.loads(summary_path.read_text(
                                encoding="utf-8")).get("documents", []))
                        except (OSError, json.JSONDecodeError):
                            count = 0
                items.append({
                    "id": workspace["id"],
                    "label": workspace["label"],
                    "documents": count,
                    "current": workspace["out"] == self.out_dir,
                })
            if current is None:
                current = {"id": "custom", "label": "当前目录"}
                items.insert(0, {"id": "custom", "label": "当前目录",
                                 "documents": len(self.documents), "current": True})
            return {
                "current": current["id"],
                "items": items,
                "max_upload_mb": int(self.config.get("ui", {}).get("max_upload_mb", 60)),
                "max_import_files": int(self.config.get("ui", {}).get("max_import_files", 200)),
                "allowed_suffixes": allowed_suffixes(self.config),
                "zip_allowed": zip_allowed(self.config),
                # 复核动作与状态的词汇表随载荷下发：界面不再自己写死 `accept`/`correct`。
                "review_vocab": payload_vocab(self.config),
                # 文档类型中文名同理：配置是唯一主人，界面里的那份只是载荷没到时的兜底。
                "doc_type_labels": doc_type_labels(self.config),
            }

    def switch_workspace(self, workspace_id: str) -> Dict:
        workspace = next((item for item in self.workspaces
                          if item["id"] == workspace_id), None)
        if workspace is None:
            raise ValueError("unknown workspace")
        self.set_out_dir(workspace["out"])
        return self.workspace_payload()

    def doc_path(self, doc_id: str) -> Optional[Path]:
        item = self.documents.get(doc_id)
        if not item:
            return None
        return self.out_dir / item["path"]

    def load_document(self, doc_id: str) -> Optional[Dict]:
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        path = self.doc_path(doc_id)
        if not path or not path.exists():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        self._doc_cache[doc_id] = document
        return document

    def ledger(self, doc_id: str) -> Dict:
        """台账的读法只留一份：能用的行（按文件顺序）、每个路径的最后一条、读不出来的行。

        这里以前自己再解析一遍，且把「合法 JSON 但不是对象」的行原样交给界面——一行 `null`
        会让 `record.field_path` 抛异常、整页打不开。
        """
        return read_review_ledger(self.out_dir, doc_id)

    def reviews(self, doc_id: str) -> list:
        return self.ledger(doc_id)["all"]

    def ledger_unread(self, doc_id: str) -> int:
        return len(self.ledger(doc_id)["unread"])

    def append_review(self, doc_id: str, payload: Dict) -> Dict:
        document = self.load_document(doc_id)
        if document is None:
            raise ValueError("unknown document")
        field_path = payload.get("field_path")
        field = document["fields"].get(field_path)
        if field is None:
            raise ValueError("unknown field_path")
        action = payload.get("action")
        vocab = action_vocab(self.config)
        if str(action) not in vocab["actions"]:
            raise ValueError("action must be one of " + " / ".join(sorted(vocab["actions"])))
        corrected = payload.get("corrected_value")
        if str(action) in vocab["value_changing"]:
            if corrected is None or str(corrected).strip() == "":
                raise ValueError("corrected_value is required")
            corrected = str(corrected).strip()
        else:
            corrected = None
        reviewer = str(payload.get("reviewer") or "local")[:60]
        record = {
            "document_id": doc_id,
            "field_path": field_path,
            "original_value": field.get("value"),
            "corrected_value": corrected,
            "action": action,
            "reviewer": reviewer,
            "timestamp": now_iso(),
            "auto_decision": field.get("decision"),
            **({"source_sha256": document["source"]["sha256"]}
               if (document.get("source") or {}).get("sha256") else {}),
        }
        review_dir = self.out_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = review_dir / f"{doc_id}.review.jsonl"
        with ledger_path.open("a", encoding="utf-8") as handle:
            # 上一行没有换行结尾（手改过台账、或上一次写被截断）时直接续写，会把两条记录粘成
            # 一行——那两条人工修正随后被读取侧当坏行一起丢掉，而界面当时已说过「已保存」。
            if ledger_path.exists() and ledger_path.stat().st_size and _needs_newline(ledger_path):
                handle.write("\n")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # 「已保存」不能只是 write() 没报错：写完要用导出层同一套读法确认真的读得回来。
        stored = self.ledger(doc_id)
        saved = stored["records"].get(field_path)
        if saved is None or saved.get("timestamp") != record["timestamp"]:
            raise RuntimeError(f"台账写入后读不回来：{doc_id} 的 {field_path}"
                               f"（台账有 {len(stored['unread'])} 行解析不出来）")
        self._report = None
        return {"record": record, "ledger_unread": len(stored["unread"])}

    def render_page(self, doc_id: str, page_no: int, dpi: int) -> Optional[Path]:
        document = self.load_document(doc_id)
        if document is None:
            return None
        source = Path(document.get("source", {}).get("path", ""))
        page_count = int(document.get("source", {}).get("page_count") or 0)
        if not source.exists() or page_no < 1 or page_no > page_count:
            return None
        work = resolve_path(self.config["paths"]["work"], self.config) / "render"
        work.mkdir(parents=True, exist_ok=True)
        # The renderer must show the same geometry the fields were parsed from, so
        # it opens the normalized copy and re-renders whenever that copy is newer.
        ready = normalized_source(source, self.config)
        stamp = max(source.stat().st_mtime, ready.stat().st_mtime)
        target = work / f"{doc_id}_p{page_no}_{dpi}.png"
        if not target.exists() or target.stat().st_mtime < stamp:
            pdf = open_document(source, self.config)
            page = pdf[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            pix.save(target)
            pdf.close()
        return target

    def receive_upload(self, filename: str, stream, length: int) -> Dict:
        with self._lock:
            return store_upload(stream, filename, length, self.config)

    def run_import(self) -> Dict:
        with self._lock:
            result = process_inbox(self.config)
            self.set_out_dir(Path(result["out_dir"]))
            return result

    def remove(self, doc_id: str) -> Dict:
        with self._lock:
            if self.out_dir != user_out_dir(self.config).resolve():
                raise ValueError("示例与验收数据区由样本目录生成，不能在这里删除；"
                                 "请切到「我的资料」再删除自己导入的资料")
            result = remove_document(doc_id, self.config)
            self.set_out_dir(Path(result["out_dir"]))
            return result

    def export_results(self) -> Dict:
        with self._lock:
            if not (self.out_dir / "summary.json").exists():
                raise ValueError("当前数据区还没有可导出的结果，请先导入资料并完成复核")
            result = export_zip(self.out_dir, config=self.config)
            self._report = result["report"]
            result["download"] = f"/api/export/download/{quote(result['name'])}"
            return result

    def report(self, force: bool = False) -> Dict:
        with self._lock:
            if not (self.out_dir / "summary.json").exists():
                raise ValueError("当前数据区还没有结果，请先导入资料")
            if force or self._report is None:
                self._report = build_report(self.out_dir, config=self.config)
            return self._report


class Handler(BaseHTTPRequestHandler):
    server_version = "DocReview/0.2"
    store: ReviewStore = None  # set by make_server

    def log_message(self, fmt, *args):
        pass

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: Optional[str] = None,
              download_name: Optional[str] = None):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or
                         mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{download_name}"')
        self.send_header("Cache-Control",
                         "no-store" if path.suffix in (".png", ".html") else "max-age=60")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self, limit: int = 1_000_000) -> Dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > limit:
            raise ValueError("invalid body size")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return bool(re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$", origin))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            target = (WEB_DIR / name).resolve()
            if WEB_DIR.resolve() not in target.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._file(target)
            return
        if path == "/api/summary":
            self._json(self.store.summary)
            return
        if path == "/api/workspaces":
            self._json(self.store.workspace_payload())
            return
        if path in {"/report", "/api/report"}:
            try:
                report = self.store.report(force="refresh" in parse_qs(parsed.query))
            except (ValueError, SystemExit) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path == "/report":
                self._file(Path(report["path"]), "text/html; charset=utf-8")
            else:
                self._json({"ok": True, **report})
            return
        match = re.match(r"^/api/export/download/([^/]+)$", path)
        if match:
            name = unquote(match.group(1))
            if not ZIP_NAME_RE.match(name):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            target = resolve_path(self.store.config["paths"]["work"],
                                  self.store.config) / "exports" / name
            self._file(target, "application/zip", name)
            return
        match = re.match(r"^/api/document/([^/]+)$", path)
        if match:
            doc_id = unquote(match.group(1))
            document = self.store.load_document(doc_id)
            if document is None:
                self._json({"error": "unknown document"}, HTTPStatus.NOT_FOUND)
                return
            ledger = self.store.ledger(doc_id)
            self._json({"document": document, "reviews": ledger["all"],
                        "ledger_unread": len(ledger["unread"]),
                        "ledger_stale": len(ledger["stale"])})
            return
        match = re.match(r"^/api/page/([^/]+)/(\d+)\.png$", path)
        if match:
            doc_id, page_no = unquote(match.group(1)), int(match.group(2))
            query = parse_qs(parsed.query)
            dpi = int(query.get("dpi", [self.store.config["ui"].get("render_dpi", 150)])[0])
            dpi = max(72, min(300, dpi))
            target = self.store.render_page(doc_id, page_no, dpi)
            if target is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._file(target, "image/png")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if not self._origin_allowed():
            self._json({"error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
            return
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/workspace":
                payload = self._read_json()
                workspace = self.store.switch_workspace(str(payload.get("id") or ""))
                self._json({"ok": True, "workspace": workspace,
                            "summary": self.store.summary})
                return
            if path == "/api/import":
                query = parse_qs(parsed.query)
                filename = (query.get("filename") or [""])[0]
                length = int(self.headers.get("Content-Length") or 0)
                result = self.store.receive_upload(filename, self.rfile, length)
                self._json({"ok": True, **result})
                return
            if path == "/api/import/run":
                result = self.store.run_import()
                self._json({
                    "ok": True,
                    "out_dir": result["out_dir"],
                    "document_ids": result["document_ids"],
                    "failures": result["failures"],
                    "summary": self.store.summary,
                    "workspace": self.store.workspace_payload(),
                })
                return
            if path == "/api/export":
                self._json({"ok": True, **self.store.export_results()})
                return
            match = re.match(r"^/api/document/([^/]+)/remove$", path)
            if match:
                result = self.store.remove(unquote(match.group(1)))
                self._json({"ok": True, "removed": result["document_id"],
                            "trash": result["trash"], "moved": result["moved"],
                            "pruned": result.get("pruned") or [],
                            "remaining": result["remaining"],
                            "summary": self.store.summary,
                            "workspace": self.store.workspace_payload()})
                return
            match = re.match(r"^/api/review/([^/]+)$", path)
            if match:
                doc_id = unquote(match.group(1))
                payload = self._read_json()
                result = self.store.append_review(doc_id, payload)
                self._json({"ok": True, **result})
                return
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except SystemExit as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except PermissionError as exc:
            self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
            return
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def make_server(out_dir: Path, host: str, port: int, config: Dict) -> ThreadingHTTPServer:
    store = ReviewStore(out_dir, config)
    handler = type("BoundHandler", (Handler,), {"store": store})
    return ReviewHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local field review UI")
    parser.add_argument("--out", default=None,
                        help="输出目录；不填则进入配置里第一个数据区（默认「我的资料」）")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--open-browser", action="store_true",
                        help="启动后在浏览器打开实际监听地址")
    args = parser.parse_args(argv)
    config = load_config()
    ui = config.get("ui", {})
    workspaces = ui.get("workspaces") or []
    out_dir = Path(args.out) if args.out else (
        resolve_path(workspaces[0]["out"], config) if workspaces else ROOT / "out")
    host = args.host or ui.get("host", "127.0.0.1")
    port = args.port or int(ui.get("port", 8765))
    max_port = int(ui.get("port_scan_max", 8785))
    server = None
    for candidate in range(port, max_port + 1):
        try:
            server = make_server(out_dir, host, candidate, config)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port in {port}-{max_port}")
    url = f"http://{host}:{port}"
    print(f"review UI: {url}")
    print("press Ctrl+C to stop")
    try:
        if args.open_browser:
            import webbrowser
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
