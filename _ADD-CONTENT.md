# 内容维护指南

> 这份文件名以 `_` 开头，Quarto 不会把它渲染成网页，也不会发布到线上，纯粹是给你自己看的备忘。
> 每次要加新内容时打开这份文件，复制对应模板即可。

---

## 一、目录结构

```
My-Blog/
├── index.qmd                 首页（Hero 标题 + Research Interests + 三个板块入口）
├── about.qmd                 About 页
├── styles.css                全站样式（所有颜色、间距、版式都在这里）
├── _quarto.yml               网站配置（导航栏、主题、页脚、lang）
├── _ADD-CONTENT.md           本文件，不发布
│
├── _tools/                   本地小工具，不发布
│   └── tex2qmd.py            LaTeX → Quarto 转换器（热统笔记用）
│
├── physics-notes/
│   ├── index.qmd             Physics Notes 落地页（课程 / 单篇笔记列表）
│   │
│   ├── group-theory/         群论（一门课 = 一个文件夹）
│   │   ├── index.qmd         课程落地页（课程信息 + 讲次列表）
│   │   └── ch02-groups-basic-definitions.qmd
│   │
│   └── thermodynamics/       热力学与统计物理
│       ├── index.qmd         课程落地页
│       └── ch01-basic-laws.qmd
│
├── projects/
│   ├── index.qmd             Projects 落地页（方向卡片 + 项目列表）
│   │
│   ├── vicsek-model/         Vicsek 模型复现（一个项目 = 一个文件夹）
│   │   ├── index.qmd         项目正文
│   │   ├── README.md         项目说明（给看代码的人）
│   │   ├── images/           配图
│   │   └── code/             核心源码
│   │
│   └── quantum-ai/
│       └── index.qmd
│
└── thoughts/
    ├── index.qmd             Thoughts 落地页（随笔列表）
    └── <新随笔>.qmd           以后新建的单篇随笔放这里
```

加内容分两种规模：

- **单篇**（一篇随笔、一篇独立笔记）：在板块文件夹里新建一个 `.qmd` 文件 → 回到该板块的 `index.qmd`，在列表里加一条链接。
- **整门课 / 整个项目**：新建一个**文件夹**，里面放 `index.qmd`（课程落地页或项目正文）、必要时加 `README.md`、`images/`、`code/`。这样一个单元自成一体，想下线时删掉整个文件夹即可，不会牵连别的页面。

文件夹落地后，记得在上一层的 `index.qmd` 列表里加一条指向 `文件夹名/index.html` 的链接。

---

## 二、新笔记模板（Physics Notes）

在 `physics-notes/` 里新建文件，例如 `fluctuation-response.qmd`：

````markdown
---
title: "涨落与响应：从配分函数出发"
date: 2026-03-01
---

正文从这里开始。数学公式直接用 LaTeX：

行内公式：$\langle (\Delta E)^2 \rangle = k_B T^2 C_V$

独立成行：

$$
\chi = \frac{\partial \langle M \rangle}{\partial H}
$$

需要推导步骤可以用对齐环境：

$$
\begin{aligned}
Z &= \sum_i e^{-\beta E_i} \\
F &= -\frac{1}{\beta} \ln Z
\end{aligned}
$$

代码块和普通 Markdown 一样：

```python
import numpy as np

def partition_function(energies, beta):
    return np.sum(np.exp(-beta * np.asarray(energies)))
```
````

小提示：

- `date` 会显示在标题下方，写 `YYYY-MM-DD`。
- 公式默认用 MathJax 渲染，`$...$` 行内、`$$...$$` 独立成行，开箱可用，不需要额外配置。
- 文件名用英文小写加连字符（会直接变成网址），标题用中文没问题。

---

## 三、在列表里加一条（三个板块都一样）

打开对应板块的 `index.qmd`，找到 `<div class="entry-list">`，把 `empty-note` 那一行删掉，换成：

```html
<a class="entry-item" href="fluctuation-response.html">
<div class="entry-meta">2026 · Statistical Physics</div>
<h3 class="entry-title">涨落与响应：从配分函数出发</h3>
<p class="entry-desc">一句话说明这篇在讲什么。</p>
</a>
```

四个可替换的地方：

| 位置 | 写什么 | 例子 |
|---|---|---|
| `href` | 文件名，`.qmd` 换成 `.html` | `fluctuation-response.html` |
| `entry-meta` | 年份 · 主题 | `2026 · Statistical Physics` |
| `entry-title` | 标题 | `涨落与响应：从配分函数出发` |
| `entry-desc` | 一句话摘要 | `从配分函数推出涨落-耗散定理。` |

主题词建议统一用这几个，方便以后按主题筛选：

- **Physics Notes**：`Statistical Physics` / `Quantum Mechanics` / `Phase Transitions` / `Mathematical Physics`
- **Projects**：`Numerical Simulation` / `Computational Physics` / `AI`
- **Thoughts**：`Learning` / `Science` / `Notes`

⚠️ 一个坑：HTML 块**不要缩进到 4 个空格以上**，否则 Pandoc 会把它当成代码块，标签会原样显示在页面上（首页之前踩过这个坑）。

新条目加在 `<div class="entry-list">` 里的**最上面**，列表就是按时间倒序。

---

## 四、新项目模板（Projects）

````markdown
---
title: "二维 Ising 模型的蒙特卡洛模拟"
date: 2026-03-01
---

## 问题

想验证二维方格子 Ising 模型在 $T_c \approx 2.269\,J/k_B$ 附近的相变行为。

## 方法

Metropolis 单自旋翻转算法，用 Python + NumPy 实现，格子尺寸 $L = 16, 32, 64$。

## 结果

磁化强度在临界温度附近快速下降，比热容出现峰值……

## 代码

```python
# 关键部分
```
````

项目页建议固定用「问题 / 方法 / 结果 / 代码」四段，比笔记更强调"做完了什么"。

### 项目是文件夹，不是单个文件

一个项目建一个文件夹，例如 `projects/ising-simulation/`：

```
projects/ising-simulation/
├── index.qmd       项目正文（上面这个模板），渲染成 index.html
├── README.md       给看代码的人看的说明，不渲染成网页
├── images/         配图，正文里用 ![](images/xxx.png) 引用
└── code/           核心源码
```

外层 `projects/index.qmd` 的链接写 `href="ising-simulation/index.html"`。

两点经验：

- **`code/` 里只放核心源码**，不要放编译产物（`.exe`）、模拟轨迹（`.bin`）、日志（`.log` / `.err`）。这些动辄几百 MB，进 git 就再也删不干净。
- **外部论文的 PDF / 全文 / 截图不要进仓库**（arXiv 和 APS 都有版权）。引用写成文献条目即可。

---

## 五、新随笔模板（Thoughts）

```markdown
---
title: "关于「重新推导一遍」这件事"
date: 2026-03-01
---

正文。这类内容不需要公式，也不用分小节，直接写就好。
```

---

## 六、本地预览

在 `D:\My-Blog` 下打开终端：

```powershell
quarto preview
```

会自动打开浏览器并**热重载** —— 改完文件保存，页面立刻更新，不用手动刷新。看完按 `Ctrl + C` 停止。

只想构建一次、不起服务器：

```powershell
quarto render
```

---

## 七、发布到线上

```powershell
cd D:\My-Blog
git add -A
git commit -m "Add note: 涨落与响应"
git push
```

推送后 GitHub Actions 会**自动**重新构建并部署，大约 1–2 分钟。之后刷新
<https://wenjium88-png.github.io/my-blog/> 就能看到。

> 网络不走代理连不上 github.com 时，push 前先执行：
> `$env:HTTPS_PROXY = "http://127.0.0.1:7897"`

---

## 八、可用的样式类（想微调时用）

| class | 用途 | 用在哪 |
|---|---|---|
| `entry-list` | 列表容器 | 三个落地页 |
| `entry-item` | 一条内容（整块可点） | 列表里 |
| `entry-meta` | 小字元信息（日期 · 主题） | 条目内 |
| `entry-title` | 条目标题 | 条目内 |
| `entry-desc` | 条目摘要 | 条目内 |
| `topic-grid` / `topic-card` | 主题卡片网格 | 落地页顶部 |
| `page-lead` | 页面引导语 | 落地页顶部 |
| `interest-item` / `interest-number` | 首页带编号的兴趣条目 | 首页 |
| `hero-title` | 首页大标题 | 首页 |
| `empty-note` | 空状态提示块 | 列表里没内容时 |

全局配色变量在 `styles.css` 顶部的 `:root` 里，改主题色只要改 `--accent-color`。

---

## 九、把 LaTeX 笔记搬上网页

热力学笔记原本是 LaTeX，用 `_tools/tex2qmd.py` 转换：

```
python _tools\tex2qmd.py <输入.tex> <输出.qmd> --title "标题" --subtitle "副标题"
```

它做六件事：调 pandoc（自动展开导言区自定义宏）、拆掉公式外层壳并接上 `{#eq-x}` 编号、把 `definition`/`theorem` 等环境转成 Quarto 的 `::: {.theorem #thm-x}`、把 `\eqref`/`\ref` 转成可点击交叉引用、替换 MathJax 不认识的宏（`\dd`、`\SI`）、加 YAML 头。转换后会在终端提醒哪些定理引用需要人工看一眼。

### 交叉引用的三条规矩

这几条都是踩过坑才定下来的，写新笔记时照做即可：

1. **引用公式必须用方括号形式**：写 `由式[@eq-fab]导出`，**不要**写 `由式@eq-fab导出`。
   中文直接跟在 `@eq-fab` 后面时，pandoc 会把汉字一起当成引用 ID，报
   `Unable to resolve crossref @eq-fab导`。
2. **定理 / 定义用围栏 div**：`::: {.theorem #thm-boyle}` … `:::`。
   正文里不要自己写「定理 定理」这种重复——链接文字本身已经带了类型词。
3. **`_quarto.yml` 里的 `lang: zh` 不要删**。它让 Quarto 自动输出中文标签
   （定义 / 定理 / 证明 / 引理 / 例 / 注记 / 命题 / 推论），同时本地化目录、搜索框等界面文字。
   公式编号前缀已在 `_quarto.yml` 里统一设成 `crossref: eq-prefix: ""`（全局生效），
   引用只会显示纯数字，正文自己写「式」字（`由式[@eq-vicsek]导出`）。新页面不用再单独配。

### 一个反复踩到的坑

**HTML 块里任何一行都不要缩进到 4 个空格以上。** Pandoc 会把这种行当成「缩进代码块」，
于是 `<h3>`、`<p>` 这些标签会以等宽字体原样显示在页面上，而不是渲染成标题和段落。
这个 bug 在首页 Research Interests 区块真的发生过一次。
