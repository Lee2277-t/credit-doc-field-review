"""静态审计：代码里的 config 兜底默认值是否等于出厂配置里的值。

「阈值全外置」只保证配置里的数被读到，管不住另一半：`config.get("key", 0)` 里的
`0` 也是一个数。YAML 缺这个键时它决定行为，与出厂值不同就意味着「配置缺失」和
「配置出厂」是两种产品，而单测通常只跑后者。这里用 AST 把 `docpipe/*.py` 里所有
形如 `<含 config 的表达式>.get("key", 字面量)` 的兜底值抓出来。

比对必须按**配置路径**做，不能只按键名：出厂配置里有 23 个叶子键名在不同段下取值
不同（`render_dpi` parsing 300 / ui 150、`min_confidence` 0.5 / 0.55、`dpi` 200 / 400
……），只按键名比对会让「段定位错」的兜底值碰上另一段的同值而蒙混过关。路径解析
覆盖 `config["a"]`、`config.get("a", {})`、`(config.get("a") or {})`、
`self.config.get("a", {})`、`self.store.config["a"]`、`load_config()`、`copy.deepcopy(config)`，
同文件内 `name = <可解析表达式>` 的别名（如 `report_config = config.get("report", {})`），
以及 `def f(config): return <可解析路径>` 这种「调用它等于取那一段」的函数
（如 `_page_quality_options(config).get("min_effective_dpi", 150)`）、所有调用点都
传同一段配置的私有辅助函数形参（如 `_page_options(report_config, override)`）。
解析不出路径的接收者（键名是变量的 `config["doc_types"].get(doc_type, {})`）退回
按键名比对，并单独列进 `unresolved`，不假装它被精确检查过。

只做诊断与守门，不参与任何流水线判决。
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _flatten(node, prefix: Tuple[str, ...] = ()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, prefix + (str(key),))
    else:
        yield prefix, node


def config_leaves(config: Dict) -> Dict[str, List[Tuple[str, object]]]:
    leaves: Dict[str, List[Tuple[str, object]]] = {}
    for path, value in _flatten(config):
        if path:
            leaves.setdefault(path[-1], []).append((path, value))
    return leaves


def config_keys(config: Dict) -> set:
    keys = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                keys.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return keys


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None


def _looks_like_config(name: str) -> bool:
    return "config" in name.lower() or name in {"c", "tuned"}


def path_of(node, aliases: Dict[str, Optional[Tuple[str, ...]]],
            functions: Optional[Dict[str, Tuple[str, ...]]] = None) -> Optional[Tuple[str, ...]]:
    """把接收者表达式解析成配置路径；解析不出来返回 None。"""
    if isinstance(node, ast.Name):
        if node.id in aliases:
            return aliases[node.id]
        # 只有这几个名字本身就是整份配置；`report_config` 这类参数名不该默认指到根上
        # （那会把「report 段里明明有的键」误报成配置里没有）。
        return () if node.id in {"config", "c", "tuned", "cfg"} else None
    if isinstance(node, ast.Attribute):
        if node.attr == "config":
            return path_of(node.value, aliases, functions) or ()
        return None
    if isinstance(node, ast.Subscript):
        base = path_of(node.value, aliases, functions)
        if base is None:
            return None
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return base + (index.value,)
        return None
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for value in node.values:
            resolved = path_of(value, aliases, functions)
            if resolved is not None:
                return resolved
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and functions and func.id in functions:
            return functions[func.id]
        if isinstance(func, ast.Attribute) and func.attr == "get" and node.args:
            base = path_of(func.value, aliases, functions)
            key = node.args[0]
            if base is None or not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return None
            return base + (key.value,)
        if isinstance(func, ast.Name) and _looks_like_config(func.id):
            return ()
        if isinstance(func, ast.Attribute) and func.attr == "deepcopy" and node.args:
            return path_of(node.args[0], aliases, functions)
        return None
    if isinstance(node, ast.Compare):  # (config or load_config()) 之外的防御写法
        left = path_of(node.left, aliases, functions)
        return left
    return None


def _path_functions(tree) -> Dict[str, Tuple[str, ...]]:
    """模块级 `def f(config): return <可解析路径>` —— 调用它等于取那一段配置。

    `_page_quality_options(config).get("min_effective_dpi", 150)` 这种接收者的
    「段」由函数体决定，读到函数体就能把它从「按键名比」升级成「按段比」。
    只认单条 `return`（可忽略文档字符串）且第一个形参就是配置的函数，其余一律不猜。
    """
    returns: Dict[str, ast.AST] = {}
    scopes: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.args.args:
            continue
        body = [item for item in node.body
                if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            continue
        returns[node.name] = body[0].value
        scopes[node.name] = {node.args.args[0].arg: ()}
    resolved: Dict[str, Tuple[str, ...]] = {}
    for _ in range(3):
        changed = False
        for name, value in returns.items():
            path = path_of(value, scopes[name], resolved)
            if path is not None and resolved.get(name) != path:
                resolved[name] = path
                changed = True
        if not changed:
            break
    return resolved


def _aliases(tree, functions: Optional[Dict[str, Tuple[str, ...]]] = None):
    """模块级别的 `name = <可解析表达式>`；函数内的别名由 `_scoped_aliases` 单独算。"""
    aliases: Dict[str, Optional[Tuple[str, ...]]] = {}
    assignments = [node for node in tree.body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)]
    for _ in range(3):
        changed = False
        for node in assignments:
            name = node.targets[0].id
            resolved = path_of(node.value, aliases, functions)
            if resolved is not None and aliases.get(name) != resolved:
                aliases[name] = resolved
                changed = True
        if not changed:
            break
    return aliases


def _module_constants(tree) -> Dict[str, object]:
    """模块级 `NAME = 字面量` 的取值；这类常量当兜底时同样可能和出厂配置分家。"""
    constants: Dict[str, object] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        ok, value = _literal(node.value)
        if ok:
            constants[node.targets[0].id] = value
    return constants


def _parents(tree) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _owner_function(node, parents) -> Optional[ast.AST]:
    current = parents.get(node)
    while current is not None and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
        current = parents.get(current)
    return current


def _param_names(func) -> set:
    """函数的全部形参名；形参一律不算别名（同名实参常常是别人递进来的普通字典）。"""
    if func is None:
        return set()
    args = func.args
    names = {item.arg for item in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _scoped_aliases(scope, functions, outer, parents=None) -> Dict[str, Optional[Tuple[str, ...]]]:
    """同一作用域内 `name = <可解析表达式>` 的别名，继承模块级别名。

    别名必须按作用域算：`options` 这种名字在一个文件里会被赋值成好几个不同的配置段，
    用文件级别名表去解析等于张冠李戴。
    """
    aliases = dict(outer)
    body = scope if scope is not None else ast.Module(body=[], type_ignores=[])
    assignments = [node for node in ast.walk(body)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and (parents is None or _owner_function(node, parents) is scope)]
    for _ in range(3):
        changed = False
        for node in assignments:
            name = node.targets[0].id
            resolved = path_of(node.value, aliases, functions)
            if resolved is not None and aliases.get(name) != resolved:
                aliases[name] = resolved
                changed = True
        if not changed:
            break
    return aliases


def _unassigned(func, name: str) -> bool:
    for item in ast.walk(func):
        if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in item.targets):
            return False
    return True


def _param_sections(tree, functions, module_aliases,
                    parents) -> Dict[str, Dict[str, Tuple[str, ...]]]:
    """私有辅助函数的形参：所有调用点都传同一段配置时，它就是那一段。

    `_page_options(report_config, override)` 的段由调用者决定，`report_config` 这个
    名字本身说明不了任何事。只认下划线开头的函数——按约定私有函数不会被别的模块调用，
    文件内的调用点就是全部；所有调用点必须解析成同一个路径，有一处不确定就整体不认，
    宁可留在 `unresolved` 里按键名比。形参在函数体里被重新赋值也不认。
    """
    calls: Dict[str, List[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)
    scopes: Dict[int, Dict[str, Optional[Tuple[str, ...]]]] = {}
    sections: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("_")
                and node.args.args):
            continue
        all_sites = calls.get(node.name, [])
        sites = [site for site in all_sites
                 if _owner_function(site, parents) is not node and site.args]
        param = node.args.args[0].arg
        if not sites or len(sites) != len(all_sites) or not _unassigned(node, param):
            continue
        resolved_paths = set()
        for site in sites:
            caller = _owner_function(site, parents)
            key = id(caller)
            if key not in scopes:
                table = _scoped_aliases(caller, functions, module_aliases, parents)
                for name in _param_names(caller):
                    table.pop(name, None)
                scopes[key] = table
            resolved_paths.add(path_of(site.args[0], scopes[key], functions))
        if len(resolved_paths) == 1 and None not in resolved_paths:
            sections.setdefault(node.name, {})[param] = next(iter(resolved_paths))
    return sections


def code_defaults(source: str) -> List[Tuple[int, str, object, str, Optional[Tuple[str, ...]]]]:
    """(行号, 键, 兜底值, 接收者原文, 配置路径或 None)。"""
    hits: List[Tuple[int, str, object, str, Optional[Tuple[str, ...]]]] = []
    tree = ast.parse(source)
    functions = _path_functions(tree)
    aliases = _aliases(tree, functions)
    constants = _module_constants(tree)
    parents = _parents(tree)
    sections = _param_sections(tree, functions, aliases, parents)
    scoped: Dict[int, Dict[str, Optional[Tuple[str, ...]]]] = {}

    def aliases_for(node):
        scope = _owner_function(node, parents)
        key = id(scope)
        if key not in scoped:
            table = _scoped_aliases(scope, functions, aliases, parents)
            for name in _param_names(scope):
                table.pop(name, None)
            if scope is not None:
                table.update(sections.get(scope.name) or {})
            scoped[key] = table
        return scoped[key]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get"):
            continue
        if len(node.args) < 2:
            continue
        key_node, default_node = node.args[0], node.args[1]
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
            continue
        ok, default = _literal(default_node)
        if not ok and isinstance(default_node, ast.Name) and default_node.id in constants:
            ok, default = True, constants[default_node.id]
        if not ok:
            continue
        owner = ast.unparse(func.value)
        path = path_of(func.value, aliases_for(node), functions)
        if path is None and not _looks_like_config(owner):
            continue
        hits.append((node.lineno, key_node.value, default, owner, path))
    return hits


def _resolve(config: Dict, path: Optional[Tuple[str, ...]]):
    """沿路径取配置节点；任一段缺失返回 (False, None)。"""
    node = config
    for part in path or ():
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def audit(root: Path, config: Dict) -> Dict:
    leaves = config_leaves(config)
    keys = config_keys(config)
    mismatches, unknown_keys, unresolved = [], [], []
    checked = 0
    for file in sorted((root / "docpipe").glob("*.py")):
        source = file.read_text(encoding="utf-8")
        for lineno, key, default, owner, path in code_defaults(source):
            checked += 1
            record = {"file": file.name, "line": lineno, "key": key,
                      "default": default, "owner": owner}
            full_path = None if path is None else path + (key,)
            if full_path is not None:
                present, node = _resolve(config, path)
                if not present or not isinstance(node, dict):
                    unknown_keys.append({**record, "path": list(path)})
                    continue
                if key not in node:
                    unknown_keys.append({**record, "path": list(path)})
                    continue
                if isinstance(default, (dict, list, tuple)) or default is None:
                    continue
                if default != node[key]:
                    mismatches.append({**record, "path": list(full_path),
                                       "shipped": [f"{'.'.join(full_path)}={node[key]!r}"]})
                continue
            # 接收者不是能从 config 直接走到的节点：退回按键名比对，并如实记账。
            unresolved.append({**record, "path": None})
            if isinstance(default, (dict, list, tuple)) or default is None:
                if key not in keys:
                    unknown_keys.append({**record, "path": None})
                continue
            entries = leaves.get(key)
            if not entries:
                unknown_keys.append({**record, "path": None})
                continue
            if not any(default == value for _, value in entries):
                mismatches.append({**record, "path": None,
                                   "shipped": [f"{'.'.join(item)}={value!r}"
                                               for item, value in entries]})
    return {"checked": checked, "mismatches": mismatches,
            "unknown_keys": unknown_keys, "unresolved": unresolved}


def main(argv=None) -> int:
    from .config import load_config

    result = audit(ROOT, load_config())
    print(f"检查 {result['checked']} 处 config 兜底默认值"
          f"（其中 {len(result['unresolved'])} 处的接收者解析不到配置段，按键名比）")
    print(f"与出厂配置冲突: {len(result['mismatches'])}")
    for item in result["mismatches"]:
        where = ".".join(item["path"]) if item["path"] else f"按键名 {item['key']}"
        print(f"  {item['file']}:{item['line']} .get({item['key']!r}, {item['default']!r})"
              f" <- {item['owner']}  期望: {where}  出厂: {'; '.join(item['shipped'])}")
    print(f"出厂配置里没有同名键: {len(result['unknown_keys'])}")
    for item in result["unknown_keys"]:
        print(f"  {item['file']}:{item['line']} .get({item['key']!r}, {item['default']!r})"
              f" <- {item['owner']}")
    print("接收者无法定位到配置段: " + ", ".join(
        f"{item['file']}:{item['line']}({item['owner']})" for item in result["unresolved"]))
    print(json.dumps({k: len(v) if isinstance(v, list) else v
                      for k, v in result.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
