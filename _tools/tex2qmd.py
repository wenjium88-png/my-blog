#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tex2qmd.py —— 把本仓库的 LaTeX 课程笔记转换成 Quarto (.qmd) 网页笔记。

用法:
    python _tools/tex2qmd.py <输入.tex> <输出.qmd> [--title "标题"] [--subtitle "副标题"]

它做的事情（按顺序）：
  1. 调用 pandoc 把 LaTeX 转成 Markdown（pandoc 会自己展开导言区里的
     \\newcommand 自定义宏，比如本项目的 \\pdvc）。
  2. 拆掉 pandoc 双层包裹的公式壳:
         $$\\begin{equation} ... \\label{eq:x} \\end{equation}$$   ->   $$ ... $$ {#eq-x}
  3. 把定理类环境 ::: definition 变成带 id 的 Quarto 围栏:
         ::: definition / []{#def:x label="def:x"} ... :::   ->   ::: {.definition #def-x}
  4. \\eqref{eq:x} -> @eq-x（Quarto 交叉引用，配合 number-equations: true）
  5. 修复 MathJax 不认识的宏: \\dd -> \\mathrm{d}，\\SI{a}{b} -> a\\,\\mathrm{b}
  6. 打印所有"需要人工确认"的残留项。

注意: 第 4 步的 \\ref{thm:...} 引用（非公式）会被替换成 [定理](#thm-x)，
      编号无法自动还原，脚本会在最后把它们列出来供人工润色。
"""

import io
import os
import re
import subprocess
import sys

PANDOC = r"D:\Quarto\bin\tools\pandoc.exe"

THEOREM_ENVS = (
    "definition|theorem|property|proof|example|remark|lemma|corollary|proposition|axiom"
)


def run_pandoc(tex_path):
    """调用 pandoc 做第一遍转换，返回 markdown 文本。"""
    proc = subprocess.run(
        [PANDOC, "-f", "latex", "-t", "markdown", "--wrap=none", tex_path],
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit("pandoc 转换失败")
    return proc.stdout.decode("utf-8")


def unwrap_equations(text):
    """$$\\begin{equation} ... \\label{..} ... \\end{equation}$$ -> $$ ... $$ {#eq-x}"""
    report = {"equations": 0, "labelled": 0}

    def repl(m):
        body = m.group(1)
        report["equations"] += 1
        label = None
        lm = re.search(r"\\label\{((?:eq|eqn):[^}]+)\}", body)
        if lm:
            label = lm.group(1)
            body = body.replace(lm.group(0), "")
            report["labelled"] += 1
        body = body.strip()
        if label:
            return "$$\n%s\n$$ {#%s}" % (body, label.replace(":", "-"))
        return "$$\n%s\n$$" % body

    text = re.sub(
        r"\$\$\s*\\begin\{(?:equation|equation\*|align|align\*|gather|gather\*)\}"
        r"([\s\S]*?)"
        r"\\end\{(?:equation|equation\*|align|align\*|gather|gather\*)\}\s*\$\$",
        repl,
        text,
    )
    return text, report


def convert_divs(text):
    """::: definition + []{#def:x} -> ::: {.definition #def-x}"""
    report = {"divs": 0, "with_id": 0}

    pat = re.compile(
        r"(?m)^:::\s*(%s)\s*\n([\s\S]*?)^:::\s*$" % THEOREM_ENVS
    )

    def repl(m):
        kind, inner = m.group(1), m.group(2)
        report["divs"] += 1
        div_id = ""
        lm = re.search(r"\[\]\{#([^\s}]+)[^}]*\}", inner)
        if lm:
            div_id = " #" + lm.group(1).replace(":", "-")
            inner = inner[: lm.start()] + inner[lm.end():]
            report["with_id"] += 1
        return "::: {.%s%s}\n%s\n:::" % (kind, div_id, inner.strip("\n"))

    # 反复替换以处理相邻的多个 div
    prev = None
    while prev != text:
        prev = text
        text = pat.sub(repl, text)
    return text, report


def convert_references(text):
    """pandoc 的属性式引用 -> Quarto / 简洁锚点链接。返回 (text, refs, eqrefs)"""
    eqrefs, refs = [], []

    def eq_repl(m):
        label = m.group(1).replace(":", "-")
        eqrefs.append(label)
        # 必须用方括号形式：中文紧跟 @eq-x 时 pandoc 会把汉字吞进 ID
        # （实测 `式@eq-fac和` -> "Unable to resolve crossref @eq-fac和"）
        return "[@" + label + "]"

    text = re.sub(
        r"\[\\\[(eq:[^\]]+)\\\]\]\(#[^)]*\)\{reference-type=\"eqref\"[^}]*\}",
        eq_repl,
        text,
    )

    def ref_repl(m):
        label = m.group(1)
        refs.append(label)
        word = "定义" if label.startswith("def:") else "定理" if label.startswith("thm:") else "前文"
        return "[%s](#%s)" % (word, label.replace(":", "-"))

    text = re.sub(
        r"\[\\\[([\w:\-]+)\\\]\]\(#[^)]*\)\{reference-type=\"ref\"[^}]*\}",
        ref_repl,
        text,
    )

    # 去重：LaTeX 原文常写「定理~\ref{thm:x}」，链接文字又生成一遍「定理」，
    # 会渲染成「定理 定理」。三种情形：
    #   前词 == 链接文字 -> 吃掉前词，只留链接
    #   链接文字是兜底的「前文」-> 用前词当链接文字（如「见定义~\ref{coeff:eos}」）
    #   两者不同 -> 原样保留，不猜
    def dedup_repl(m):
        prev, inner, label = m.group(1), m.group(2), m.group(3)
        if prev == inner or inner == "前文":
            return "[%s](#%s)" % (prev, label)
        return "%s [%s](#%s)" % (prev, inner, label)

    text = re.sub(r"(定理|定义)\s*\[(定理|定义|前文)\]\(#([\w\-]+)\)", dedup_repl, text)

    # 「\ref{x} 的」被 pandoc 转成链接 + 空格，中文里不该有空格
    text = re.sub(r"(\(#[^)]+\))\s+的", r"\1的", text)
    return text, refs, eqrefs


def fix_macros(text):
    """MathJax 不认识的宏。"""
    report = {}
    report["dd"] = len(re.findall(r"\\dd\b", text))
    text = re.sub(r"\\dd\b", r"\\mathrm{d}", text)

    def si_repl(m):
        num, unit = m.group(1), m.group(2)
        num = re.sub(r"e([-+]?\d+)", r"\\times 10^{\1}", num)
        return r"%s\,\mathrm{%s}" % (num, unit)

    report["si"] = len(re.findall(r"\\SI\{[^}]*\}\{[^}]*\}", text))
    text = re.sub(r"\\SI\{([^}]*)\}\{([^}]*)\}", si_repl, text)
    return text, report


def strip_label_spans(text):
    """剩下没被 div 吃掉的 []{#x label="x"} 空锚点。"""
    leftovers = re.findall(r"\[\]\{#([^\s}]+)[^}]*\}", text)
    text = re.sub(r"\[\]\{#([^\s}]+)[^}]*\}", "", text)
    return text, leftovers


def build_frontmatter(title, subtitle):
    return (
        "---\n"
        'title: "%s"\n'
        'subtitle: "%s"\n'
        "format:\n"
        "  html:\n"
        "    number-equations: true\n"
        "    toc: true\n"
        "crossref:\n"
        '  eq-prefix: ""\n'
        "---\n\n"
    ) % (title, subtitle)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)

    tex_path, out_path = sys.argv[1], sys.argv[2]
    title = "未命名"
    subtitle = "课程笔记"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    if "--subtitle" in sys.argv:
        subtitle = sys.argv[sys.argv.index("--subtitle") + 1]

    text = run_pandoc(tex_path)
    text, eq = unwrap_equations(text)
    text, div = convert_divs(text)
    text, refs, eqrefs = convert_references(text)
    text, mac = fix_macros(text)
    text, spans = strip_label_spans(text)

    text = re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"
    out = build_frontmatter(title, subtitle) + text

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(out)

    print("已写出: %s (%d 字符)" % (out_path, len(out)))
    print("  公式拆壳     : %d 个（其中带编号 %d）" % (eq["equations"], eq["labelled"]))
    print("  定理类环境   : %d 个（其中带 id %d）" % (div["divs"], div["with_id"]))
    print("  公式引用 ->@ : %d 处" % len(eqrefs))
    print("  \\dd 替换    : %d 处" % mac["dd"])
    print("  \\SI 替换    : %d 处" % mac["si"])
    print("  残留空锚点   : %d 个 %s" % (len(spans), spans[:8]))
    if refs:
        print("\n  [需人工润色] 定理/定义引用 %d 处（编号无法自动还原）:" % len(refs))
        for r in refs:
            print("      - %s" % r)
    for bad in ("\\label{", "\\begin{equation}", "\\SI{", "\\eqref{"):
        n = text.count(bad)
        if n:
            print("  [残留] %s x%d" % (bad, n))


if __name__ == "__main__":
    main()
