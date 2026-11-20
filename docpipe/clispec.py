"""SCENARIO §11 的命令面：文档写下的命令必须真的还能跑。

§11 那段 bash 是交付接口本身——用户照着它跑第一步。模块改名、`--input` 换成
`--in`、带值的选项退化成开关，流水线、验收、单测全都不会红：文档还写着旧命令，
第一次使用就失败。这里的机制是把契约那头的命令读出来，导入对应模块，拦下它自己
的 argparse 解析（`main([])` 在 `parse_args` 处就停），拿真 parser 比对两件事：

- 文档写的每个长选项在 parser 上存在；不存在是 `flag_missing`；
- 文档里有值的选项实机也必须吃值（`--port 8765` 不能退化成开关）；
  反方向同样算，是 `flag_arity`。

只读契约、只读源码，不执行任何命令的正文（解析在第一行，拦得住）。
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import ROOT

SECTION = re.compile(r"^## 11\. ", re.M)
FENCE = "```bash"
COMMAND = re.compile(r"^\s*python -m (docpipe\.[A-Za-z0-9_]+)(.*)$")
FLAG = re.compile(r"^--[A-Za-z][\w-]*$")


class _Captured(Exception):
    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__("argparse captured")
        self.parser = parser


def section11(text: str) -> str:
    match = SECTION.search(text)
    if not match:
        raise ValueError("SCENARIO.md 里找不到 §11")
    body = text[match.start():]
    fence = body.find(FENCE)
    if fence < 0:
        raise ValueError("SCENARIO.md §11 里找不到 bash 代码块")
    end = body.find("```", fence + len(FENCE))
    if end < 0:
        raise ValueError("SCENARIO.md §11 的 bash 代码块没有结束标记")
    return body[fence + len(FENCE):end]


def documented(text: str) -> List[Dict]:
    """§11 里每条 `python -m docpipe.x …` 的模块与长选项；带值与否按下一个词判断。"""
    commands = []
    for line in section11(text).splitlines():
        match = COMMAND.match(line)
        if not match:
            continue
        tokens = match.group(2).split()
        flags = []
        for index, token in enumerate(tokens):
            if not FLAG.match(token):
                continue
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            flags.append({"name": token,
                          "value": bool(following) and not following.startswith("--")})
        commands.append({"module": match.group(1), "flags": flags,
                         "command": line.strip()})
    return commands


def parser_for(module_name: str, load: Callable[[str], object] = importlib.import_module
               ) -> argparse.ArgumentParser:
    """拿到模块 `main()` 自己建的那个 parser，不让它继续往下跑。

    五个入口的第一句都是 `parser.parse_args(argv)`，所以把类方法换成抛异常就停在
    解析处。若将来某个 main 在解析前先干活，这里会晚一步——那正是要把问题暴露出来
    的地方，异常信息里写清模块名。
    """
    module = load(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise LookupError(f"{module_name} 没有可调用的 main()")
    real_args = argparse.ArgumentParser.parse_args
    real_known = argparse.ArgumentParser.parse_known_args

    def trap(parser, *args, **kwargs):
        raise _Captured(parser)

    argparse.ArgumentParser.parse_args = trap
    argparse.ArgumentParser.parse_known_args = trap
    try:
        main([])
    except _Captured as caught:
        return caught.parser
    finally:
        argparse.ArgumentParser.parse_args = real_args
        argparse.ArgumentParser.parse_known_args = real_known
    raise LookupError(f"{module_name}.main([]) 没有走到 argparse 解析")


def bound_flags(parser: argparse.ArgumentParser) -> Dict[str, bool]:
    """长选项 → 是否吃值。`store_true`/`help`/`version` 这类 nargs=0 的不吃值。"""
    bound = {}
    for action in parser._actions:
        takes = action.nargs != 0
        for option in action.option_strings:
            bound[option] = takes
    return bound


def problems(commands: List[Dict],
             load: Callable[[str], object] = importlib.import_module) -> List[Dict]:
    if not commands:
        return [{"code": "section11_unread", "detail": "SCENARIO §11 里没有命令"}]
    found = []
    for item in commands:
        module = item["module"]
        try:
            parser = parser_for(module, load)
        except Exception as error:  # 导入失败、没有 main、拦不到解析都算接口断了
            found.append({"code": "module_missing", "detail": f"{module}: {error}"})
            continue
        bound = bound_flags(parser)
        for flag in item["flags"]:
            name = flag["name"]
            if name not in bound:
                found.append({"code": "flag_missing", "detail": f"{module} {name}"})
            elif bound[name] != flag["value"]:
                found.append({"code": "flag_arity",
                              "detail": f"{module} {name} 文档"
                                        f"{'带值' if flag['value'] else '不带值'}、实机"
                                        f"{'带值' if bound[name] else '不带值'}"})
    return found


def audit(text: str, load: Callable[[str], object] = importlib.import_module) -> Dict:
    commands = documented(text)
    return {"commands": len(commands),
            "modules": [item["module"] for item in commands],
            "flags": sum(len(item["flags"]) for item in commands),
            "problems": problems(commands, load)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit SCENARIO §11 command surface")
    parser.add_argument("--scenario", default=str(ROOT / "SCENARIO.md"))
    args = parser.parse_args(argv)
    text = Path(args.scenario).read_text(encoding="utf-8")
    result = audit(text)
    print(f"命令 {result['commands']} 条 / 长选项 {result['flags']} 个 / "
          f"问题 {len(result['problems'])} 条")
    for item in result["problems"][:20]:
        print(f"  ! {item['code']}: {item['detail']}")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
