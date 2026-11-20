"""静态审计：交付链路上有没有任何一处会走出本机。

SCENARIO 的铁律「全链路本地/离线」以前只写在文档里，靠人守。加一个 CDN 字体、把 OCR
模型改成在线下载、在单文件网页版里塞一张远程图片——功能全都正常、测试一个字都不报，
数据却出了域。这里把这条铁律做成三类可复查的判据，全部读 `security.offline` 的配置：

1. **远程地址字面量**：`docpipe/*.py` 里带协议的字符串（含 docstring）只要命中
   `blocked_schemes` 且主机不在 `loopback_hosts`，就是违规——回环之外的主机名一律算，
   内网 IP 也算，因为「不出域」问的是数据离开本机，不是离开公网。
2. **网络连接点**：`urlopen` / `urlretrieve` / `create_connection` / `getaddrinfo` /
   `gethostbyname` 与 `import socket` / `requests` / `httpx` / `urllib3` 只允许出现在
   `network_allowed_modules` 里（现在只有验收自己，它只连本机服务），且该模块必须
   同时通过第 1 类判据。流水线本体（解析、抽取、信任决策、导出）出现任何一个就是违规。
3. **交付页面的资源引用**：`docpipe/web/*` 与生成 HTML 的源码里，`src=` / `href=` /
   `srcset=` / `url()` / `fetch()` 之类只许相对路径、片段与 `inline_schemes`（`data:`
   内联、`file:`）；出现被禁协议或协议相对写法 `//host` 就报。

只做诊断与守门，不参与任何流水线判决。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

URL_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*)://([^\s'\"<>|\\{}`,;)]+)")
NETWORK_FUNCS = {"urlopen", "urlretrieve", "create_connection", "getaddrinfo", "gethostbyname"}
NETWORK_MODULES = {"socket", "requests", "httpx", "urllib3", "aiohttp", "ftplib"}

DEFAULTS = {
    "loopback_hosts": ["127.0.0.1", "localhost", "::1"],
    "blocked_schemes": ["http", "https", "ftp", "ftps", "sftp", "ws", "wss"],
    "inline_schemes": ["data", "file"],
    "network_allowed_modules": ["acceptance.py"],
    "asset_reference_attrs": ["src", "href", "srcset", "poster", "background"],
    "asset_url_functions": ["url", "fetch", "importScripts", "EventSource"],
    # 命名空间声明里的 URL 是**标识符**不是**定位符**：`xmlns="http://www.w3.org/2000/svg"`
    # 永远不会被解析成一次请求。只认这些属性名，所以 `<image xlink:href="http://…">`
    # 这种真会去取远程资源的写法仍然会报。
    "inert_attrs": ["xmlns"],
}


def options(config: Dict) -> Dict:
    """`security.offline` 整段；缺键时退回与出厂配置逐字相同的默认值。"""
    section = (config.get("security") or {}).get("offline") or {}
    merged = dict(DEFAULTS)
    merged.update({key: value for key, value in section.items() if value is not None})
    return merged


def _host_of(rest: str) -> str:
    host = re.split(r"[/?:#]", rest, 1)[0]
    return host.strip("[]").split("%", 1)[0]


def _inert_prefix(text: str, start: int, inert: List[str]) -> bool:
    before = text[:start]
    for attr in inert:
        if re.search(rf"{re.escape(str(attr))}(?::[A-Za-z0-9_-]+)?\s*=\s*[\"']?$", before, re.I):
            return True
    return False


def remote_urls(text: str, opts: Dict) -> List[Tuple[str, str]]:
    """返回 [( scheme://host , 原文片段 )]，只列真正会走出本机的地址。"""
    blocked = [str(item).lower() for item in opts["blocked_schemes"]]
    loopback = [str(item) for item in opts["loopback_hosts"]]
    hits: List[Tuple[str, str]] = []
    for match in URL_RE.finditer(text):
        scheme, rest = match.group(1), match.group(2)
        lowered = scheme.lower()
        host = _host_of(rest)
        if lowered not in blocked or host in loopback:
            continue
        if _inert_prefix(text, match.start(1), [str(a) for a in opts["inert_attrs"]]):
            continue
        hits.append((f"{lowered}://{host}", f"{scheme}://{rest}"))
    return hits


def _string_literals(tree) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((getattr(node, "lineno", 0), node.value))
    return found


def remote_literals(source: str, opts: Dict) -> List[Dict]:
    tree = ast.parse(source)
    hits = []
    for lineno, value in _string_literals(tree):
        for label, raw in remote_urls(value, opts):
            hits.append({"line": lineno, "url": label, "text": raw[:80]})
    return hits


def network_calls(source: str) -> List[Dict]:
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or ""
            if isinstance(node, ast.Import):
                name = node.names[0].name
            root = name.split(".", 1)[0]
            if root in NETWORK_MODULES:
                hits.append({"line": node.lineno, "call": f"import {name}"})
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        label = None
        if isinstance(func, ast.Attribute) and func.attr in NETWORK_FUNCS:
            label = f"{ast.unparse(func.value)}.{func.attr}"
        elif isinstance(func, ast.Name) and func.id in NETWORK_FUNCS:
            label = func.id
        if label:
            hits.append({"line": node.lineno, "call": label})
    return hits


def _attr_pattern(attrs: List[str]) -> re.Pattern:
    joined = "|".join(re.escape(str(item)) for item in attrs)
    return re.compile(rf"\b(?:{joined})\s*=\s*[\"']([^\"']+)[\"']", re.I)


def _func_pattern(funcs: List[str]) -> re.Pattern:
    joined = "|".join(re.escape(str(item)) for item in funcs)
    return re.compile(rf"\b(?:{joined})\s*\(\s*[\"'`]([^\"'`\n]+)", re.I)


def remote_asset_refs(text: str, opts: Dict) -> List[Dict]:
    """界面资源引用里只许相对路径 / 片段 / 内联；被禁协议与 `//host` 都报。"""
    blocked = [str(item).lower() for item in opts["blocked_schemes"]]
    inline = [str(item).lower() for item in opts["inline_schemes"]]
    loopback = [str(item) for item in opts["loopback_hosts"]]
    hits: List[Dict] = []
    values = []
    for pattern in (_attr_pattern(opts["asset_reference_attrs"]),
                    _func_pattern(opts["asset_url_functions"]),
                    re.compile(r"url\(\s*[\"']?([^\"')]+)[\"']?\s*\)", re.I)):
        for match in pattern.finditer(text):
            values.append((match.start(), match.group(1)))
    for offset, value in values:
        stripped = value.strip()
        if stripped.startswith("data:") or stripped.startswith("{") or "${" in stripped:
            continue
        if stripped.startswith("//"):
            hits.append({"offset": offset, "ref": stripped[:80],
                         "url": f"//{_host_of(stripped[2:])}"})
            continue
        for scheme, rest in URL_RE.findall(stripped):
            lowered = scheme.lower()
            host = _host_of(rest)
            if lowered in inline:
                continue
            if lowered in blocked and host not in loopback:
                hits.append({"offset": offset, "ref": stripped[:80], "url": f"{lowered}://{host}"})
    return hits


def audit(root: Path, config: Dict) -> Dict:
    opts = options(config)
    allowed_modules = {str(item) for item in opts["network_allowed_modules"]}
    remote_literals_found, call_sites_found, asset_refs_found = [], [], []
    scanned = 0
    for file in sorted((root / "docpipe").glob("*.py")):
        if file.name == Path(__file__).name:
            continue
        source = file.read_text(encoding="utf-8")
        scanned += 1
        for item in remote_literals(source, opts):
            remote_literals_found.append({**item, "file": file.name})
        calls = network_calls(source)
        if calls and file.name not in allowed_modules:
            for item in calls:
                call_sites_found.append({**item, "file": file.name})
        if file.name == "report.py":
            for item in remote_asset_refs(source, opts):
                asset_refs_found.append({**item, "file": file.name})
    for asset in sorted((root / "docpipe" / "web").glob("*")):
        if not asset.is_file():
            continue
        scanned += 1
        for item in remote_asset_refs(asset.read_text(encoding="utf-8"), opts):
            asset_refs_found.append({**item, "file": f"web/{asset.name}"})
    # 一个文件同时出现在"有网络连接"和"连的不是回环"里，等于本地服务被指到远端。
    for module in allowed_modules & {item["file"] for item in call_sites_found}:
        if any(item["file"] == module for item in remote_literals_found):
            remote_literals_found.append({"file": module, "line": 0,
                                          "url": "loopback-only",
                                          "text": "允许联网的模块里出现了非回环地址字面量"})
    return {"scanned": scanned, "remote_literals": remote_literals_found,
            "network_call_sites": call_sites_found, "remote_asset_refs": asset_refs_found}


def main(argv=None) -> int:
    from .config import load_config

    result = audit(ROOT, load_config())
    print(f"扫描 {result['scanned']} 个文件（docpipe/*.py + docpipe/web/*）")
    for key, label in (("remote_literals", "非回环地址字面量"),
                       ("network_call_sites", "白名单之外的网络连接点"),
                       ("remote_asset_refs", "交付页面的远程资源引用")):
        items = result[key]
        print(f"{label}: {len(items)}")
        for item in items:
            where = f"{item['file']}:{item.get('line', item.get('offset', 0))}"
            print(f"  {where} {item.get('url', '')} <- {item.get('text', item.get('ref', ''))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
