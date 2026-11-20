"""交接文档锚点自检：标签号 == #L 号，而且锚点后面那句人话也要真的在那一行附近。

anchor_audit.py 只查文件存在与行号范围，查不出「代码插了几行、锚点全体往下漂」这类漂移；
这一版把标签当作可证伪的断言来读：`[docpipe/run.py:96](.../run.py#L96)` 除了号要对上，
还要在 ±2 行内找得到标签里点名的标识符或「中文原话」，找不到就报出来人看。

第二遍查的是锚点**后面**那句话：文档里 `[tests/test_core.py:4264] 的 ValueRewriteNoteTests`
这种写法，号对不对由第一遍管，「那个位置是不是这个类」只有读后面那个人名才知道。上一轮
`chain_note` 插进 review_actions.py 顶部，把 `value_rewrite_note` 从 20 行推到 42 行，第一遍
全绿（标签里没有人名），文档就这么留着一条指向别人函数的锚点过了审——这一遍补的就是这个洞。
第三遍查的是锚点**前面**那句话：`符号名（[path:N]）` 这种「名字在锚点前」的写法，第二遍
根本看不到（它只读 match.end() 之后的 90 字符），于是号漂了 150 行也全绿——第二十二轮
就是这样漏掉 `test_one_percent_rate_passes_the_tax_crosscheck` 的。这一遍读锚点前 60 字符里
最近的那个 `符号` 或「原话」，判据与第二遍**共用同一个** verify_claim（窗口 / 叶子段 /
归属 def-class），免得两套口径自己打架。同一个 claim 前后都出现时只审一次。

报出来的东西不当结论用：人要逐条核对是不是真错了，还是文档换了说法。
"""
import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
# 始终检查 README，本地有交接文档时也检查其中的链接。
# README 里的行号锚点以前没人查过，漂了也不知道。
DOCS = [path for path in (ROOT / "交接文档.md", ROOT / "README.md") if path.exists()]

ANCHOR = re.compile(r"\[([^\]]*?)\]\(([^)\s]*?)#L(\d+)\)")
LABEL_NUM = re.compile(r"(\d+)$")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
QUOTE = re.compile(r"[「『]([^」』]{2,20})[」』]")
# 锚点后面紧跟的断言：锚点 + 空白/「的」/「，」 + `symbol` 或「原话」。分隔符里不放 `）`，
# 因为「（[锚点]），下一个符号」那句说的是**下一条**锚点的主人，硬套过来只会造假阳性。
# 断言长度上限 60、尾窗 90：`config_keys.doc_type_threshold_moves_the_gate`（45 字符）与
# `bl.legal_representative` 以前都被 24 字符的尾窗截断，第二遍根本没跑，锚点漂了 42 行
# 也没人报（2026-10-28 人工复核表格行锚点时才发现）。
PROSE = re.compile(r"^[\s，,]*?(?:的)?[\s，,]*?(?:`([^`\n]{3,60})`|[「『]([^」』]{2,20})[」』])")
# 锚点**前**那句断言：行首到锚点之间，最近的一个 `符号` / 「原话」，中间只允许夹一个开括号。
# 括号里不放「的」，因为「`名字` 的 [`锚点`]」与「`名字`（[锚点]）」是同一种归属，只是写法不同。
PRE = re.compile(r"(?:`([^`\n]{3,60})`|[「『]([^」』]{2,20})[」』])[\s（(]*$")
# `reviews.action_vocab(config)` 这种带实参的调用：摘掉实参按符号名查，跟 `函数名()` 同一处理。
CALL = re.compile(r"^([A-Za-z_][\w.]*)\([^()]*\)$")
PATHISH = re.compile(r"\.(py|js|csv|yaml|yml|md|json|html|css|bat)\b|[\\/]")


def tokens(label: str) -> List[str]:
    """标签里能被「那一行有没有」证伪的部分：标识符 + 「」里的原话。"""
    text = label.split(":")[-1] if ":" in label else label
    found = [m.group(1) for m in QUOTE.finditer(label)]
    found += [t for t in IDENT.findall(text) if not t.endswith(".py")]
    return found


def normalize(text: str) -> str:
    """界面文案里 `N` 与 `${fragments}` 是同一个占位符，比之前先抹平。"""
    text = re.sub(r"\$\{[^}]*\}", "N", text)
    return re.sub(r"\s+", "", text)


def claim_of(match) -> str | None:
    """从 PROSE / PRE 的匹配里取那句断言；反引号里必须是符号名，否则整句跳过。"""
    claim = match.group(1) or match.group(2)
    if match.group(1):
        call = CALL.match(claim)
        if call:
            claim = call.group(1)
        if not re.match(r"^[A-Za-z_][\w.\-]*$", claim):
            return None  # `：只有带 ` 这种是反引号包住了整段句子
    if PATHISH.search(claim):
        return None  # `out/fields.csv`、`tests/test_core.py` 这类是产物/文件名片段
    return claim


def verify_claim(claim: str, source: List[str], target: int) -> bool:
    """这句话在锚点所指位置成不成立：窗口命中 / 叶子段回退 / 归属最近的 def-class。"""
    near = normalize(" ".join(source[max(0, target - 5):target + 4]))
    if normalize(claim) in near:
        return True
    # 点号路径在 YAML 里是分两行写的（`evaluation:` + `table_structure:`），整串永远
    # 搜不到。叶子段能落在窗口里就算这句话成立。
    probe = claim.split(".")[-1] if "." in claim else claim
    if len(probe) >= 4 and normalize(probe) in near:
        return True
    # 锚点常常故意指向函数**体里**的判定行（`[trust.py:248] 的 validate_bank_statement`
    # 指的是那句 return），这种人话按窗口比永远对不上。往上找最近的 def/class，名字对得上
    # 就算这句话归这一层——落到**别的**函数里仍然照报，这才是我们要的牙。`^\s*` 是为了覆盖
    # 类方法（`serve.workspace_payload` 的锚点指向 `"doc_type_labels": …` 那一行，方法名在
    # 缩进的 def 上）。
    owner = re.compile(r"^\s*(?:async )?(?:def|class)\s+(\w+)")
    enclosing = ""
    for row in source[:target]:
        hit = owner.match(row)
        if hit:
            enclosing = hit.group(1)
    return bool(enclosing and enclosing in (claim, probe))


def locate(claim: str, rel: str, source: List[str]) -> str:
    """报真位置：界面文案是模板串拼出来的（「骑缝章残片 ${n} 页」），整文件里搜得到
    就说明这句话存在，只是锚点漂了——把行号打出来，下一个人不用再去猜。"""
    probe = claim.split(".")[-1] if "." in claim else claim
    found = [i + 1 for i, row in enumerate(source) if probe in row]
    if not found:
        return "→ 整份文件都没有这句话"
    return f"→ 真位置 {rel}#L{found[0]}" + (f"（共 {len(found)} 处 {found[:4]}）"
                                          if len(found) > 1 else "")


def main() -> None:
    problems: List[str] = []
    checked = prose_checked = pre_checked = nameless = 0
    # 可以传文档路径进来做 A/B（故意改坏的副本），不传就查出厂那两份。
    docs = [Path(p) for p in sys.argv[1:]] or DOCS
    lines = [(doc.name, number, line) for doc in docs
             for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1)]
    cache: dict = {}
    for doc_name, number, line in lines:
        for match in ANCHOR.finditer(line):
            label, raw_path, target = match.group(1), match.group(2), int(match.group(3))
            rel = raw_path.replace("\\", "/").removeprefix(ROOT.as_posix() + "/")
            if not (ROOT / rel).exists():
                problems.append(f"文档:{doc_name}:{number} {rel} 文件不存在")
                continue
            label_tail = LABEL_NUM.search(label.split("/")[-1].split(" ")[-1] or "")
            if label_tail and int(label_tail.group(1)) != target:
                problems.append(f"文档:{doc_name}:{number} 标签写 {label_tail.group(1)} 但锚到 {rel}#L{target}")
            if rel not in cache:
                cache[rel] = (ROOT / rel).read_text(encoding="utf-8").splitlines()
            source = cache[rel]
            if not 1 <= target <= len(source):
                problems.append(f"文档:{doc_name}:{number} {rel} 只有 {len(source)} 行，锚到 L{target}")
                continue
            checked += 1
            window = " ".join(source[max(0, target - 3):target + 2])
            wanted = tokens(label)
            if wanted and not [w for w in wanted if w in window]:
                found = [i + 1 for i, row in enumerate(source)
                         if any(w in row for w in wanted)]
                problems.append(f"文档:{doc_name}:{number} 标签 {rel}#L{target} 附近找不到 {wanted[:3]}"
                                f"｜{source[target - 1].strip()[:60]}"
                                + (f" → 真位置 {rel}#L{found[0]}" if found else " → 整份文件都没有"))
            if not wanted:
                nameless += 1
                # `[docpipe/acceptance.py:2499]`、`[:2634]` 这类标签里没有可证伪的人名，第一遍只能
                # 核对「标签号==锚号」，第二遍又要紧跟一句带反引号的人话才跑——本轮清漂移时发现
                # 恰恰是这一类在漂（验收里插了 100 多行函数，6 条这种锚点指到了别人的代码里，
                # 检查器一声不吭）。空行是最没争议的一种假：那句话不可能落在空行上。
                if not source[target - 1].strip():
                    nearby = [i + max(0, target - 6) + 1 for i, row in enumerate(
                        source[max(0, target - 7):target + 5]) if row.strip()]
                    problems.append(f"文档:{doc_name}:{number} 无名锚点 {rel}#L{target} 落在空行上"
                                    f"（{rel} 附近非空行 {nearby[:6]}）——号漂了，"
                                    f"或者把锚点后面那句人话写成可证伪的形式（`名字` / 「原话」）")
            # 第二遍：锚点后面那句人话
            tail = line[match.end():match.end() + 90]
            prose = PROSE.match(tail)
            claim = claim_of(prose) if prose else None
            if claim:
                prose_checked += 1
                if not verify_claim(claim, source, target):
                    problems.append(f"文档:{doc_name}:{number} {rel}#L{target} 那一行附近没有"
                                    f"「{claim}」 {locate(claim, rel, source)}")
            # 第三遍：锚点**前面**那句人话（`符号名（[锚点]）` 的写法第二遍看不见）。
            # 同一个 claim 前后都写了就只审一次——两遍共用 verify_claim，口径不会分家。
            pre = PRE.search(line[:match.start()])
            before = claim_of(pre) if pre else None
            if before and before != claim:
                pre_checked += 1
                if not verify_claim(before, source, target):
                    problems.append(f"文档:{doc_name}:{number} {rel}#L{target} 那一行附近没有"
                                    f"锚点前那句话「{before}」 {locate(before, rel, source)}")
    print(f"anchors checked={checked} nameless={nameless} prose={prose_checked} "
          f"pre={pre_checked} problems={len(problems)}")
    for item in problems:
        print("  -", item)


if __name__ == "__main__":
    main()
