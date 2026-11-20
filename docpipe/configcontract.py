"""SCENARIO §13 的配置契约对账：契约里写出的每个配置键都必须存在、值一致、有人读。

方向与 `docpipe.configaudit` 相反：configaudit 问「出厂 YAML 里的键有没有人读」，
死键审计问「YAML 里这些键有没有执行点」，这里问**契约那一头**——§13 的示例配置是
契约的一部分，用户在它上面判断「这个工具能做到什么」。三件事各自能单独失败：

1. `missing`：契约列了、出厂 YAML 里没有——用户照着契约写配置会静默不生效；
2. `mismatched`：两边都有但值不同——契约与出厂是两个产品；
3. `unread`：YAML 里有、`docpipe/` 里没人按名字读——最像装饰键的那种。

已批准的偏差点写在 `DEVIATIONS` 里：它们**每次都被打印出来**，不是静默跳过。
按名字动态查表的段（`thresholds.risk_tiers.*`）与只标识契约版本的顶层键
（`version` / `scenario`）不算 unread，理由写在常量旁边。

不要把这个模块的源码算进「有人读」的证据：它的常量里写着键名，自己给自己作证
就是那句「检查器自己也会犯它要防的错」。`acceptance.py` 同理（它断言键名）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .config import ROOT, load_config

# 契约列了、产品刻意不实现。写在这里等于承认差异，不是关掉检查：
# 审计每次打印，且偏差点必须能在契约里找到（打错键名不会被当成豁免）。
DEVIATIONS = {
    "output.include_parsed_debug": (
        "调试开关已删除：实现它等于往交付 ZIP 里塞解析记录，会改交付字节，"
        "也把逐格坐标这类内部证据当成给人看的东西（README「output.include_parsed_debug」）"
    ),
}

# 按名字动态查表的段：源码里永远不会出现这些叶子键的字面量。
DYNAMIC_PREFIXES = ("thresholds.risk_tiers.",)

# 只标识契约版本、不进任何判决的顶层键。
IDENTITY_KEYS = ("version", "scenario")

# 「有人读」的证据来源：流水线模块。验收与测试会写键名，算进来等于自己作证。
SOURCE_EXCLUDE = {"acceptance.py", "configcontract.py", "configaudit.py"}


def section13_spec(text: str) -> Dict:
    """从 SCENARIO.md 里取出 §13 那一段 yaml 代码块并解析。"""
    marker = "\n## 13. "
    start = text.find(marker)
    if start < 0:
        raise ValueError("SCENARIO.md 里找不到 §13")
    body = text[start:]
    fence = body.find("```yaml")
    if fence < 0:
        raise ValueError("SCENARIO.md §13 里找不到 yaml 代码块")
    end = body.find("```", fence + len("```yaml"))
    if end < 0:
        raise ValueError("SCENARIO.md §13 的 yaml 代码块没有结束标记")
    spec = yaml.safe_load(body[fence + len("```yaml"):end])
    if not isinstance(spec, dict):
        raise ValueError("SCENARIO.md §13 的 yaml 代码块不是映射")
    return spec


def leaves(node, prefix: str = "") -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(leaves(value, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out.append((prefix, node))
    return out


def lookup(config: Dict, path: str) -> Tuple[bool, object]:
    node: object = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _same(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def source_text(root: Path = ROOT) -> str:
    chunks: List[str] = []
    for path in sorted((root / "docpipe").rglob("*.py")):
        if path.name in SOURCE_EXCLUDE:
            continue
        chunks.append(path.read_bytes().decode("utf-8", "ignore"))
    return "\n".join(chunks)


def audit(spec: Dict, config: Dict, source: str) -> Dict:
    """三问 + 已批准偏差点。返回的每个列表都能单独为空/非空。"""
    missing: List[str] = []
    mismatched: List[str] = []
    unread: List[str] = []
    deviations: List[str] = []
    dynamic: List[str] = []
    counted = 0
    paths = [path for path, _ in leaves(spec)]
    stale_deviations = [path for path in DEVIATIONS if path not in paths]
    for path, expected in leaves(spec):
        if path in DEVIATIONS:
            deviations.append(path)
            continue
        found, actual = lookup(config, path)
        if not found:
            missing.append(path)
            continue
        counted += 1
        if not _same(expected, actual):
            mismatched.append(f"{path}: 契约={expected!r} 出厂={actual!r}")
        if path in IDENTITY_KEYS or path.startswith(DYNAMIC_PREFIXES):
            if path not in IDENTITY_KEYS:
                dynamic.append(path)
            continue
        if path.split(".")[-1] not in source:
            unread.append(path)
    return {
        "leaves": counted + len(missing) + len(deviations),
        "missing": missing,
        "mismatched": mismatched,
        "unread": unread,
        "deviations": deviations,
        "stale_deviations": stale_deviations,
        "dynamic": dynamic,
    }


def render(result: Dict) -> str:
    lines = [
        f"契约键 {result['leaves']} 条（已批准偏差 {len(result['deviations'])} 条、"
        f"动态查表 {len(result['dynamic'])} 条）",
        f"出厂配置缺失: {len(result['missing'])}",
        f"值不一致: {len(result['mismatched'])}",
        f"没人读: {len(result['unread'])}",
        f"偏差清单自己过期: {len(result['stale_deviations'])}",
    ]
    for name in ("missing", "mismatched", "unread", "deviations", "stale_deviations"):
        for item in result[name]:
            lines.append(f"  {name}: {item}")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    scenario = Path(argv[0]) if argv else ROOT / "SCENARIO.md"
    config = load_config()
    spec = section13_spec(Path(scenario).read_text(encoding="utf-8"))
    result = audit(spec, config, source_text())
    print(render(result))
    bad = bool(result["missing"] or result["mismatched"] or result["unread"]
               or result["stale_deviations"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
