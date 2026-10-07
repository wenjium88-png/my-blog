# -*- coding: utf-8 -*-
"""相图结果分析：读取 scan_summary.csv，输出图与临界噪声估计。

输出：
  results/figures/phi_vs_eta.png        每个密度下 phi(eta)，多尺寸对比（展示有限尺寸移动）
  results/figures/heatmap_phi_L{L}.png  每个尺寸的 <phi> 热图
  results/figures/phase_diagram.png     综合相图：<phi> 热图 + 带相标记 + eta_c(L) 线
  results/eta_c.csv                     临界噪声估计（chi 峰 + phi 中点）
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_THRESHOLD = 0.35       # 密度不均匀度超过该值标记为带相候选（L>=128 时适用；
                            # 小 L 下有序相自身的巨数涨落会把衬度抬高到 ~0.3，见 README）
PHI_LEVELS = [0.25, 0.5, 0.75]


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:   # utf-8-sig 兼容 BOM
        for r in csv.DictReader(f):
            rows.append({k: float(v) if k not in ("seed", "N", "n_transient",
                         "n_measure", "subsample") else int(float(v)) for k, v in r.items()})
    return rows


def group_by(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["L"], r["rho"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["eta"])
    return g


def eta_c_estimates(etas, phi, chi):
    """eta_c 的两个估计：chi 峰位置；phi 从高到低的中点（线性插值）。"""
    i = int(np.argmax(chi))
    e_chi = float(etas[i])
    lo, hi = float(np.min(phi)), float(np.max(phi))
    e_mid = float(np.interp(0.5 * (lo + hi), phi[::-1], etas[::-1])) if hi - lo > 1e-12 else np.nan
    return e_chi, e_mid


def pivot_matrix(g, L, field):
    """把 (rho, eta) 数据整理成矩阵；缺失组合填 NaN。"""
    rhos = sorted({k[1] for k in g if k[0] == L})
    etas = sorted({r["eta"] for r in g[(L, rhos[0])]})
    m = np.full((len(rhos), len(etas)), np.nan)
    for ir, rho in enumerate(rhos):
        for ie, eta in enumerate(etas):
            for r in g[(L, rho)]:
                if r["eta"] == eta:
                    m[ir, ie] = r[field]
    return rhos, etas, m


def log_edges(values):
    """对数轴下的一组网格边界（几何中点外推）。"""
    lv = np.log(values)
    mid = 0.5 * (lv[:-1] + lv[1:])
    lo = lv[0] - (mid[0] - lv[0])
    hi = lv[-1] + (lv[-1] - mid[-1])
    return np.exp(np.concatenate(([lo], mid, [hi])))


def plot_heatmap(ax, rhos, etas, m, cmap="viridis", vmin=0.0, vmax=1.0,
                 levels=None, band_mask=None, colorbar=True):
    etas = np.asarray(etas, dtype=float)
    de = 0.5 * (etas[1] - etas[0]) if len(etas) > 1 else 0.025
    eta_edges = np.concatenate(([etas[0] - de], etas[:-1] + de, [etas[-1] + de]))
    rho_edges = log_edges(np.array(rhos))
    X, Y = np.meshgrid(eta_edges, rho_edges)
    mm = np.ma.masked_invalid(m)
    pc = ax.pcolormesh(X, Y, mm, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    if levels:
        xc = 0.5 * (X[:-1, :-1] + X[1:, 1:])                    # 单元中心（线性 eta 轴）
        yc = np.exp(0.5 * (np.log(Y[:-1, :-1]) + np.log(Y[1:, 1:])))  # 对数 rho 轴中心
        cs = ax.contour(xc, yc, m, levels=levels,
                        colors="k", linewidths=0.8, linestyles="-")
        ax.clabel(cs, fmt="%.2f", fontsize=7)
    if band_mask is not None:
        rr, ee = np.meshgrid(np.array(rhos), np.array(etas), indexing="ij")
        ax.scatter(ee[band_mask], rr[band_mask], marker="x", color="white",
                   s=26, linewidths=1.2, label="band (contrast>%.1f)" % BAND_THRESHOLD,
                   zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(eta_edges[0], eta_edges[-1])
    ax.set_ylim(rho_edges[0], rho_edges[-1])
    if colorbar:
        plt.colorbar(pc, ax=ax, label=r"$\langle\phi\rangle$")
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("results", "scan_summary.csv"))
    ap.add_argument("--outdir", default=os.path.join("results", "figures"))
    ap.add_argument("--band-threshold", type=float, default=BAND_THRESHOLD)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(args.csv)
    g = group_by(rows)
    if not g:
        raise SystemExit(f"没有数据：{args.csv}")
    sizes = sorted({L for L, _ in g})
    rhos_all = sorted({rho for _, rho in g})
    print(f"数据点 {len(rows)} 个，尺寸 {sizes}，密度 {rhos_all}")

    # ---------- eta_c 估计 ----------
    eta_c_rows = []
    for (L, rho), rs in sorted(g.items()):
        etas = np.array([r["eta"] for r in rs])
        phi = np.array([r["phi_mean"] for r in rs])
        chi = np.array([r["chi"] for r in rs])
        if len(rs) < 3:
            continue
        e_chi, e_mid = eta_c_estimates(etas, phi, chi)
        eta_c_rows.append({"L": L, "rho": rho, "eta_c_chi": e_chi, "eta_c_mid": e_mid,
                           "phi_eta0": phi[0], "phi_last": phi[-1],
                           "contrast_max": max(r["contrast"] for r in rs)})
        print(f"L={L:<5} rho={rho:<4} eta_c(chi)={e_chi:<5} eta_c(mid)={e_mid:<5} "
              f"phi(eta0)={phi[0]:.3f} phi(last)={phi[-1]:.3f}")
    eta_c_path = os.path.join(os.path.dirname(args.csv), "eta_c.csv")
    with open(eta_c_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eta_c_rows[0].keys()))
        w.writeheader()
        w.writerows(eta_c_rows)
    print(f"eta_c 已写入 {eta_c_path}")

    # ---------- 图 1：phi(eta) 按密度分面板 ----------
    ncol = min(len(rhos_all), 3)
    nrow = int(np.ceil(len(rhos_all) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             squeeze=False)
    for ax, rho in zip(axes.flat, rhos_all):
        for L in sizes:
            if (L, rho) not in g:
                continue
            rs = g[(L, rho)]
            etas = [r["eta"] for r in rs]
            phi = [r["phi_mean"] for r in rs]
            err = [r["phi_err"] for r in rs]
            ax.errorbar(etas, phi, yerr=err, fmt="o-", ms=4, capsize=2,
                        label=f"L={int(L)}")
        ax.set_title(rf"$\rho$ = {rho}")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\langle\phi\rangle$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(rhos_all):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "phi_vs_eta.png"), dpi=150)
    plt.close(fig)

    # ---------- 图 2：每个尺寸的 <phi> 热图 ----------
    for L in sizes:
        rhos, etas, m = pivot_matrix(g, L, "phi_mean")
        _, _, cm = pivot_matrix(g, L, "contrast")
        band = ~np.isnan(cm) & (cm > args.band_threshold)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        plot_heatmap(ax, rhos, etas, m, levels=PHI_LEVELS, band_mask=band)
        ax.set_xlabel(r"noise $\eta$")
        ax.set_ylabel(r"density $\rho$")
        ax.set_title(rf"Vicsek order parameter, $L={int(L)}$ "
                     rf"($v_0=0.5,\ R=1$)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"heatmap_phi_L{int(L)}.png"), dpi=150)
        plt.close(fig)

    # ---------- 图 3：综合相图 ----------
    L_main = max(sizes)
    rhos, etas, m = pivot_matrix(g, L_main, "phi_mean")
    _, _, cm = pivot_matrix(g, L_main, "contrast")
    band = ~np.isnan(cm) & (cm > args.band_threshold)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    plot_heatmap(ax, rhos, etas, m, levels=PHI_LEVELS, band_mask=band)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(sizes)))
    for c, L in zip(colors, sizes):
        lr = np.array([r["rho"] for r in eta_c_rows if r["L"] == L])
        le = np.array([r["eta_c_mid"] for r in eta_c_rows if r["L"] == L])
        if len(le) > 1:
            ax.plot(le, lr, "o--", color=c, ms=5, lw=1.4,
                    label=f"eta_c, L={int(L)}")
    ax.set_xlabel(r"noise $\eta$")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title(rf"Vicsek model phase diagram ($L={int(L_main)}$, "
                 rf"$v_0=0.5$, $R=1$, angular noise)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "phase_diagram.png"), dpi=150)
    plt.close(fig)
    print(f"图已写入 {args.outdir}")


if __name__ == "__main__":
    main()
