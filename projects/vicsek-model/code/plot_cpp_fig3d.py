# -*- coding: utf-8 -*-
"""把 C++ 版（400x400，论文规格盒子）的 Fig 3(d) ramp 结果画成对照图。

读 cpp/results/cpp_d_{up,down}_s*.csv（列：rho,phi,vm,dvar,nb,N），
按 seed 平均，画 Δρ²∥ vs ρ0 的两个滞后支，并叠加论文截图的读数值。
"""
import glob
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# 论文 Fig 3(d) 截图读数（目测）：上行支在 4.2 附近崩塌、下行支在 3.4 附近跃起
PAPER_UP = {2.0: 2.10, 2.5: 2.05, 3.0: 1.75, 3.5: 1.20, 4.0: 0.45, 4.4: 0.15}
PAPER_DOWN = {3.5: 0.20, 4.0: 0.15, 4.5: 0.10}


def _save(fig, out, dpi=170):
    """保存图片；若目标被占用（Windows 常见），回退到 *_new.png。"""
    try:
        fig.savefig(out, dpi=dpi)
    except OSError:
        alt = out.replace(".png", "_new.png")
        fig.savefig(alt, dpi=dpi)
        print(f"警告：{os.path.basename(out)} 被占用，已写入 {os.path.basename(alt)}")
        out = alt
    print("written", out)
    return out


def load_branch(tag):
    """优先级：修正协议 slow_up（8000 步/点 + nb=3，6 seeds）
             > fine（4000 步/点）
             > cpp_d（第一轮 1500 步/点）。
    不同长度的 run 用插值对齐到最短网格，因此未跑完的 run 也能参与平均。"""
    if tag == "up":
        cands = [("slow_up", "slow_up_s*.csv"), ("fine", "fine_up_s*.csv"),
                 ("cpp_d", "cpp_d_up_s*.csv")]
    else:
        cands = [("fine", "fine_down_s*.csv"), ("cpp_d", "cpp_d_down_s*.csv")]
    for prefix, pattern in cands:
        runs = []
        for f in sorted(glob.glob(os.path.join(RES, pattern))):
            try:
                d = np.genfromtxt(f, delimiter=",", names=True)
                if d.size:
                    runs.append((np.atleast_1d(d["rho"]), np.atleast_1d(d["dvar"]),
                                 np.atleast_1d(d["vm"])))
            except Exception as e:
                print("skip", f, e)
        if not runs:
            continue
        # 统一按 ρ 升序（下行 ramp 的 ρ 是递减的，np.interp 要求 xp 递增）
        runs = [tuple(np.asarray(a)[np.argsort(np.asarray(r))]
                      for a in (r, dv_, vm_))
                for (r, dv_, vm_) in runs]
        n = min(len(r[0]) for r in runs)
        ref = [r for r in runs if len(r[0]) == n][0][0]
        dv = np.mean([np.interp(ref, r[0], r[1]) for r in runs], axis=0)
        vm = np.mean([np.interp(ref, r[0], r[2]) for r in runs], axis=0)
        return (ref, dv, vm, len(runs), prefix)
    return None


def main():
    up = load_branch("up")
    down = load_branch("down")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    src = up[4] if up is not None else (down[4] if down is not None else "?")
    if up is not None:
        ax.plot(up[0], up[1], "o-", color="#1a3fbf", ms=5, lw=1.6,
                label=f"C++ up ({up[3]} seeds, 400×400, {src})")
    if down is not None:
        ax.plot(down[0], down[1], "o-", color="#2e8b3d", ms=5, lw=1.6,
                label=f"C++ down ({down[3]} seeds)")
    pu = sorted(PAPER_UP)
    ax.plot(pu, [PAPER_UP[k] for k in pu], "--", color="#1a3fbf", lw=1.1, alpha=0.55,
            label="paper up (read-off)")
    pd = sorted(PAPER_DOWN)
    ax.plot(pd, [PAPER_DOWN[k] for k in pd], "--", color="#2e8b3d", lw=1.1, alpha=0.55,
            label="paper down (read-off)")
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$\Delta\rho_\parallel^2$")
    ax.set_title(r"(d) microphase $\leftrightarrow$ liquid, $\eta=0.4$", fontsize=10)
    ax.set_xlim(1.9, 4.8)
    ax.set_ylim(0, 3.9)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    if up is not None:
        ax.plot(up[0], up[2], "o-", color="#1a3fbf", ms=5, lw=1.5, label="C++ up")
    if down is not None:
        ax.plot(down[0], down[2], "o-", color="#2e8b3d", ms=5, lw=1.5, label="C++ down")
    ax.set_xlabel(r"$\rho_0$")
    ax.set_ylabel(r"$|v|$")
    ax.set_title(r"(c-style) mean velocity $|v|$", fontsize=10)
    ax.set_xlim(1.9, 4.8)
    ax.legend(fontsize=7.5)

    fig.suptitle("Fig. 3(d) rerun in C++ at the paper's box size (400×400, $v_0=0.5$)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(RES, "fig3d_cpp_400.png")
    _save(fig, out, dpi=170)
    plt.close(fig)
    if up is not None:
        print("up  : rho=", np.array2string(up[0], precision=2))
        print("      dvar=", np.array2string(up[1], precision=3))
    if down is not None:
        print("down: rho=", np.array2string(down[0], precision=2))
        print("      dvar=", np.array2string(down[1], precision=3))


if __name__ == "__main__":
    main()
