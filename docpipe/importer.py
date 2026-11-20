"""Import user documents (PDF or images) into a private workspace.

Uploads are stored in the inbox directory, deduplicated by SHA-256, then the
regular pipeline runs over the inbox into the user workspace. Parsed pages are
cached, so importing one more file does not re-parse earlier ones.

The accepted type comes from the file signature, never from the claimed
filename, so a renamed file cannot slip a different format past the check.
A .zip is unpacked on the way in: only members whose own signature is a
supported document survive, and the archive itself never lands in the inbox.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import time
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

from .config import load_config, resolve_path
from .parse import iter_inputs, sha256_file
from .run import run

SIGNATURES: List[Tuple[str, bytes]] = [
    ("pdf", b"%PDF-"),
    ("png", b"\x89PNG\r\n\x1a\n"),
    ("jpg", b"\xff\xd8\xff"),
    ("gif", b"GIF87a"),
    ("gif", b"GIF89a"),
    ("bmp", b"BM"),
    ("tif", b"II*\x00"),
    ("tif", b"MM\x00*"),
]
HEAD_BYTES = 16
ZIP_HEAD = b"PK\x03\x04"
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
TRASH_STAMP = re.compile(r"^\d{8}-\d{6}$")
MB = 1024 * 1024


def allowed_suffixes(config: Optional[Dict] = None) -> List[str]:
    config = config or load_config()
    items = config.get("ingest", {}).get("allowed_suffixes", ["pdf"])
    return [str(item).lower().lstrip(".") for item in items]


def zip_allowed(config: Optional[Dict] = None) -> bool:
    """Whether a .zip container may be unpacked on upload (not a document type)."""
    config = config or load_config()
    return bool(config.get("ingest", {}).get("zip_allowed", True))


def detect_suffix(head: bytes) -> Optional[str]:
    """Return the canonical suffix implied by a file's leading bytes."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(ZIP_HEAD):
        return "zip"
    for suffix, magic in SIGNATURES:
        if head.startswith(magic):
            return suffix
    return None


def inbox_dir(config: Optional[Dict] = None) -> Path:
    config = config or load_config()
    return resolve_path(config["paths"].get("inbox", "inbox"), config)


def user_out_dir(config: Optional[Dict] = None) -> Path:
    config = config or load_config()
    return resolve_path(config["paths"].get("user_out", "out/user"), config)


def safe_upload_name(filename: str, suffix: str) -> str:
    raw = str(filename or "").replace("\\", "/")
    name = _UNSAFE.sub("_", raw.rsplit("/", 1)[-1]).strip().strip(".")
    stem = Path(name).stem.strip()[:80] or "document"
    return f"{stem}.{suffix}"


def _unique_target(inbox: Path, filename: str, suffix: str, sha: str) -> Tuple[Path, bool]:
    """Next free inbox path, or the existing one when an identical file is already there."""
    base = inbox / safe_upload_name(filename, suffix)
    target = base
    counter = 1
    while target.exists():
        if sha256_file(target) == sha:
            return target, True
        counter += 1
        if counter > 999:
            raise ValueError("同名文件过多")
        target = inbox / safe_upload_name(f"{base.stem}-{counter}", suffix)
    return target, False


def store_upload(stream: BinaryIO, filename: str, length: int,
                 config: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    max_bytes = int(config.get("ui", {}).get("max_upload_mb", 60)) * MB
    if length <= 0:
        raise ValueError("文件为空")
    if length > max_bytes:
        raise ValueError(f"单个文件不能超过 {max_bytes // MB} MB")

    inbox = inbox_dir(config)
    inbox.mkdir(parents=True, exist_ok=True)
    temp = inbox / f".upload-{secrets.token_hex(6)}.part"
    digest = hashlib.sha256()
    head = b""
    written = 0
    try:
        with temp.open("wb") as handle:
            while written < length:
                chunk = stream.read(min(MB, length - written))
                if not chunk:
                    break
                if len(head) < HEAD_BYTES:
                    head += chunk
                    head = head[:HEAD_BYTES]
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
        if written != length:
            raise ValueError("上传中断，请重试")
        suffix = detect_suffix(head)
        if suffix is None:
            raise ValueError("无法识别文件类型，请上传 PDF 或常见格式图片")
        if suffix == "zip":
            if not zip_allowed(config):
                raise ValueError("当前配置不允许导入压缩包（config 的 ingest.zip_allowed）")
            return store_zip(temp, filename, config)
        if suffix not in allowed_suffixes(config):
            raise ValueError(f"暂不支持 .{suffix} 文件（可在 config 的 ingest.allowed_suffixes 里放开）")

        sha = digest.hexdigest()
        target, duplicate = _unique_target(inbox, filename, suffix, sha)
        if duplicate:
            return {"name": target.name, "suffix": suffix, "duplicate": True,
                    "sha256": sha, "size": written}
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    return {"name": target.name, "suffix": suffix, "duplicate": False,
            "sha256": sha, "size": written}


def store_zip(archive_path: Path, filename: str, config: Optional[Dict] = None) -> Dict:
    """Unpack an uploaded archive into the inbox, member by member.

    Only each member's basename survives, so ``../`` paths and folder structure
    inside the archive cannot write anywhere but the inbox. Counts, declared
    sizes and the compression ratio are checked before a member is inflated, and
    a member only counts as a document when its own header says so; the archive
    itself never stays in the inbox.
    """
    config = config or load_config()
    ingest = config.get("ingest", {})
    max_entries = int(ingest.get("zip_max_entries", 200))
    max_total = int(ingest.get("zip_max_total_mb", 500)) * MB
    max_ratio = float(ingest.get("zip_max_ratio", 40))
    per_file = int(config.get("ui", {}).get("max_upload_mb", 60)) * MB
    supported = set(allowed_suffixes(config))
    inbox = inbox_dir(config)
    inbox.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [info for info in archive.infolist()
                       if not info.is_dir() and str(info.filename).strip("/")]
            if not members:
                raise ValueError("压缩包是空的")
            if len(members) > max_entries:
                raise ValueError(f"压缩包内文件太多（{len(members)} 个），一次最多导入 {max_entries} 个")
            if sum(int(info.file_size or 0) for info in members) > max_total:
                raise ValueError(f"压缩包解压后超过 {max_total // MB} MB 上限，请分批导入")
            stored: List[str] = []
            stored_names: List[str] = []
            duplicates = 0
            skipped = 0
            for info in members:
                name = Path(str(info.filename).replace("\\", "/")).name
                size = int(info.file_size or 0)
                if not name or name.startswith(".") or name.startswith("._"):
                    skipped += 1
                    continue
                if size <= 0 or size > per_file:
                    skipped += 1
                    continue
                compressed = int(info.compress_size or 0)
                if size > MB and compressed and size / compressed > max_ratio:
                    raise ValueError("压缩包疑似异常（解压后体积远大于压缩包），已中止")
                with archive.open(info) as handle:
                    data = handle.read(per_file + 1)
                if len(data) > per_file:
                    skipped += 1
                    continue
                suffix = detect_suffix(data[:HEAD_BYTES])
                if suffix is None or suffix not in supported:
                    skipped += 1
                    continue
                target, duplicate = _unique_target(inbox, name, suffix,
                                                   hashlib.sha256(data).hexdigest())
                stored_names.append(target.name)
                if duplicate:
                    duplicates += 1
                else:
                    target.write_bytes(data)
                    stored.append(target.name)
            if not stored and not duplicates:
                raise ValueError("压缩包里没有可导入的资料（只认 PDF 与常见格式图片）")
            return {"name": safe_upload_name(filename, "zip"), "suffix": "zip",
                    "duplicate": not stored, "extracted": stored, "stored_names": stored_names,
                    "extracted_count": len(stored), "duplicates": duplicates,
                    "skipped": skipped, "size": archive_path.stat().st_size}
    except (zipfile.BadZipFile, NotImplementedError) as error:
        raise ValueError(f"压缩包打不开或格式不支持（{type(error).__name__}）") from error
    finally:
        if archive_path.exists():
            archive_path.unlink()


def process_inbox(config: Optional[Dict] = None, force: bool = False) -> Dict:
    config = config or load_config()
    inbox = inbox_dir(config)
    out_dir = user_out_dir(config)
    if not iter_inputs(inbox, config):
        raise ValueError("收件箱里还没有可导入的资料（PDF 或图片）")
    summary = run(inbox, out_dir, config=config, force=force)
    return {
        "summary": summary,
        "out_dir": str(out_dir.resolve()),
        "document_ids": [item["document_id"] for item in summary.get("documents", [])],
        "failures": summary.get("failures", []),
    }


def _move_into(path: Path, dest_dir: Path) -> Optional[Path]:
    """Move one file or directory into ``dest_dir``, never overwriting."""
    if not path.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / path.name
    counter = 1
    while target.exists():
        target = dest_dir / f"{path.stem}-{counter}{path.suffix}"
        counter += 1
    path.replace(target)
    return target


def _source_of(out_dir: Path, doc_id: str) -> Tuple[Optional[Path], Optional[str]]:
    """(original file, sha256) for an imported document, read back from its payload."""
    payload = out_dir / doc_id / f"{doc_id}.json"
    if not payload.exists():
        return None, None
    try:
        data = json.loads(payload.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    source = data.get("source") or {}
    path = Path(source.get("path") or "")
    return (path if path.is_absolute() else None), source.get("sha256")


def _stamp_seconds(name: str) -> float:
    try:
        return time.mktime(time.strptime(name, "%Y%m%d-%H%M%S"))
    except ValueError:
        return 0.0


def prune_trash(config: Optional[Dict] = None) -> List[str]:
    """Apply the retention policy to ``work/trash`` and return the purged run names.

    A run survives only while it is both among the newest ``trash_keep_runs``
    and younger than ``trash_keep_days``. Only direct subdirectories whose name
    is a delete stamp are ever touched, so a mistyped path or a hand-placed
    folder inside the trash root is left alone.
    """
    config = config or load_config()
    ui = config.get("ui", {})
    if not ui.get("trash_prune", True):
        return []
    keep_runs = int(ui.get("trash_keep_runs", 20))
    keep_days = int(ui.get("trash_keep_days", 30))
    root = (resolve_path(config["paths"]["work"], config) / "trash").resolve()
    if not root.is_dir():
        return []
    runs = sorted((path for path in root.iterdir()
                   if path.is_dir() and TRASH_STAMP.match(path.name)),
                  key=lambda path: path.name, reverse=True)
    cutoff = time.time() - keep_days * 86400
    purged: List[str] = []
    for index, path in enumerate(runs):
        if index < keep_runs and _stamp_seconds(path.name) >= cutoff:
            continue
        if path.parent != root:
            continue
        try:
            shutil.rmtree(path)
            purged.append(path.name)
        except OSError:
            continue
    return purged


def remove_document(document_id: str, config: Optional[Dict] = None) -> Dict:
    """Retire one imported document and rebuild the workspace from what is left.

    Nothing is deleted outright: the inbox original, the parsed and render caches,
    the output folder and the review history all move to ``work/trash/<stamp>/``, so
    a mistaken click can be undone by moving the files back. Runs beyond the
    retention policy are purged on the way out.
    """
    config = config or load_config()
    doc_id = str(document_id or "").strip()
    if not doc_id or doc_id in {".", ".."} or _UNSAFE.search(doc_id):
        raise ValueError("资料标识不合法")
    out_dir = user_out_dir(config)
    work = resolve_path(config["paths"]["work"], config)
    inbox = inbox_dir(config)
    source, sha = _source_of(out_dir, doc_id)
    if source is None:
        matches = [path for path in inbox.glob(f"{doc_id}.*") if path.is_file()]
        source = matches[0] if len(matches) == 1 else None

    trash = work / "trash" / time.strftime("%Y%m%d-%H%M%S")
    moved: List[str] = []

    def retire(path: Optional[Path]) -> None:
        if path is None:
            return
        target = _move_into(Path(path), trash)
        if target is not None:
            moved.append(str(target))

    # Only files inside the inbox belong to the import; a document whose original
    # lives elsewhere (a sample folder) keeps its source where it is.
    if source is not None and source.exists():
        try:
            inside_inbox = inbox.resolve() in source.resolve().parents
        except OSError:
            inside_inbox = False
        retire(source if inside_inbox else None)

    retire(out_dir / doc_id)
    retire(out_dir / "review" / f"{doc_id}.review.jsonl")
    retire(work / "parsed" / f"{doc_id}.json")
    images = work / "render" / "images"
    if sha:
        for cached in sorted(images.glob(f"{sha}-*")):
            retire(cached)
    for rendered in sorted((work / "render").glob(f"{doc_id}_p*_*.png")):
        retire(rendered)

    if iter_inputs(inbox, config):
        result = process_inbox(config)
    else:
        # An emptied inbox has nothing left to show; drop the stale workspace
        # index and the generated report so the UI falls back to its "no data yet"
        # state instead of offering yesterday's HTML.
        for leftover in (out_dir / "summary.json", out_dir / "fields.csv",
                        out_dir / "review", out_dir / "final"):
            retire(leftover)
        result = {"summary": {"documents": []}, "out_dir": str(out_dir.resolve()),
                  "document_ids": []}
    return {"document_id": doc_id, "trash": str(trash.resolve()), "moved": moved,
            "pruned": prune_trash(config), "remaining": len(result["document_ids"]), **result}
