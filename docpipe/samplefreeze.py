"""样本清单冻结：SCENARIO 第 109 行「样本清单冻结后不得原地修改；需要修改就新建版本号」。

样本文件是标注与指标的共同底座。原地改一份 PDF、忘了改标注，指标会假装变好/变坏，
而流水线一个字都不报——`sha256` 清单本来就在 `samples/*/manifest.json` 里记着，
只是从来没有人拿它和磁盘对过。

三道判据：

- `hash_mismatches`：清单里写的 sha256 与文件当前内容不一致 → 原地改过；
- `missing_files` / `unlisted`：清单与磁盘对不上（删了、塞了新文件）；
- `content_changed_without_version_bump`：内容摘要变了，但清单 `version` 没动
  （拿基线 `config/sample_freeze.json` 比）——「需要修改就新建版本号」这句的机械形态。

只做诊断与守门，不参与任何流水线判决。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import ROOT, load_config, resolve_path

BASELINE = ROOT / "config" / "sample_freeze.json"

NOTE = ("样本冻结快照。SCENARIO 第 109 行：清单冻结后不得原地修改，需要修改就新建版本号。"
        "整批重生成后先升 `samples/*/manifest.json` 的 version，再跑 "
        "`python -m docpipe.samplefreeze --write` 登记新摘要。")


def _split_dir(config: Dict, key: str, split: str) -> Path:
    return resolve_path(config["paths"][key], config) / split


def manifest_path(split: str, config: Optional[Dict] = None) -> Path:
    """分片清单优先；dev/holdout 共用顶层 `samples/manifest.json`。"""
    config = config or load_config()
    samples = resolve_path(config["paths"]["samples"], config)
    local = samples / split / "manifest.json"
    return local if split == "hard" or local.exists() else samples / "manifest.json"


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_digests(split: str, config: Optional[Dict] = None) -> Dict[str, str]:
    """分片两侧（样本 + 标注）的文件摘要；清单自身除外。"""
    config = config or load_config()
    files: Dict[str, str] = {}
    for key in ("samples", "labels"):
        root = _split_dir(config, key, split)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                try:
                    name = path.resolve().relative_to(ROOT).as_posix()
                except ValueError:
                    name = path.as_posix()
                files[name] = _digest(path)
    return files


def content_digest(files: Dict[str, str]) -> str:
    payload = "\n".join(f"{name} {digest}" for name, digest in sorted(files.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_split(split: str, config: Optional[Dict] = None, manifest: Optional[Dict] = None) -> Dict:
    config = config or load_config()
    path = manifest_path(split, config)
    result = {"split": split, "manifest": str(path), "version": None,
              "missing_manifest": not path.exists(), "missing_files": [],
              "hash_mismatches": [], "unlisted": [], "documents": 0}
    if manifest is None and result["missing_manifest"]:
        return result
    manifest = manifest if manifest is not None else load_manifest(path)
    result["version"] = manifest.get("version")
    listed = set()
    for item in manifest.get("documents") or []:
        if item.get("split") and item["split"] != split:
            continue
        result["documents"] += 1
        for key in ("path", "label_path"):
            rel = item.get(key)
            if not rel:
                continue
            target = Path(rel)
            if not target.is_absolute():
                target = ROOT / rel
            try:
                name = target.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                name = target.as_posix()
            listed.add(name)
            if not target.exists():
                result["missing_files"].append([item.get("document_id"), key])
            elif key == "path" and item.get("sha256") and _digest(target) != item["sha256"]:
                result["hash_mismatches"].append([item.get("document_id"), rel])
    files = file_digests(split, config)
    result["unlisted"] = sorted(set(files) - listed)
    result["digest"] = content_digest(files)
    return result


def load_baseline(path: Path = BASELINE) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("splits") or {}


def write_baseline(splits: Dict[str, Dict], path: Path = BASELINE) -> None:
    payload = {"note": NOTE, "splits": splits}
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def audit(config: Optional[Dict] = None, baseline_path: Path = BASELINE) -> Dict:
    config = config or load_config()
    baseline = load_baseline(baseline_path)
    splits: Dict[str, Dict] = {}
    problems: List[str] = []
    if baseline is None:
        problems.append(f"缺基线 {baseline_path.name}：跑 `python -m docpipe.samplefreeze --write` 登记")
    for split in ("dev", "holdout", "hard"):
        info = audit_split(split, config)
        if info["missing_manifest"]:
            if file_digests(split, config):
                problems.append(f"{split}: 有样本或标注但没有清单")
            continue
        frozen = (baseline or {}).get(split) or {}
        if frozen:
            historical_digest = (frozen.get("previous") or {}).get(str(info.get("version")))
            if info.get("version", 0) < frozen.get("version", 0) and historical_digest != info["digest"]:
                problems.append(f"{split}: 清单 version 后退了（{frozen.get('version')} → {info.get('version')}）")
            elif info.get("digest") != frozen.get("digest") and info.get("version") == frozen.get("version"):
                problems.append(f"{split}: 内容变了但 version 没升（仍是 {info.get('version')}）")
        else:
            problems.append(f"{split}: 基线里没有这一片")
        splits[split] = {"version": info.get("version"), "digest": info.get("digest"),
                         "documents": info["documents"]}
        if info["hash_mismatches"]:
            problems.append(f"{split}: 原地改过 {info['hash_mismatches']}")
        if info["missing_files"]:
            problems.append(f"{split}: 清单里的文件不见了 {info['missing_files']}")
        if info["unlisted"]:
            problems.append(f"{split}: 磁盘上有清单外的文件 {info['unlisted'][:5]}")
    return {"splits": splits, "problems": problems}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sample manifest freeze audit")
    parser.add_argument("--write", action="store_true", help="把当前状态登记为基线")
    parser.add_argument("--baseline", default=str(BASELINE))
    args = parser.parse_args(argv)
    config = load_config()
    if args.write:
        baseline = Path(args.baseline)
        previous_splits = load_baseline(baseline) or {}
        splits = {}
        for split in ("dev", "holdout", "hard"):
            info = audit_split(split, config)
            if info["missing_manifest"]:
                continue
            frozen = previous_splits.get(split) or {}
            previous = dict(frozen.get("previous") or {})
            if frozen and frozen["version"] < info["version"]:
                previous[str(frozen["version"])] = frozen["digest"]
            splits[split] = {"version": info["version"], "digest": info["digest"],
                             "documents": info["documents"]}
            if previous:
                splits[split]["previous"] = previous
        write_baseline(splits, baseline)
        print(f"登记 {len(splits)} 个分片 → {args.baseline}")
        return 0
    result = audit(config, Path(args.baseline))
    for split, info in result["splits"].items():
        print(f"{split}: version={info['version']} documents={info['documents']} "
              f"digest={str(info['digest'])[:12]}")
    for problem in result["problems"]:
        print("问题:", problem)
    return 0 if not result["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
