# -*- coding: utf-8 -*-
"""Vicsek 模型（角噪声 / scalar-angular noise）+ 相图扫描观测量。

模型定义（与文献标准实现一致）
------------------------------
    θ_i(t+1) = Arg( Σ_{j∈S_i} v_j(t) ) + η ξ_i,   ξ_i ~ U(-π, π)
    x_i(t+1) = x_i(t) + v0 e_i(t+1)          (mod L, 周期边界)

即：取邻域速度矢量和的方向，再叠加幅值为 ηπ 的均匀角度噪声。
注意术语：按 Ginelli & Chaté, Eur. Phys. J. ST 225, 2099 (2016) 的定义，
这是 **scalar/angular noise**（其式 (2)，也是 Solon-Chaté-Tailleur,
PRL 114, 068101 (2015) 式 (1) 用的版本）；所谓 vectorial noise 是把随机
矢量在归一化前加到邻居和上（其式 (6)，幅值 ∝ 邻居数），两者有限尺寸行为
略有差别但渐近性质相同。

其中 S_i 是与粒子 i 距离 <= R 的所有粒子（含自身 i）；平行更新，dt = 1。
标准参数：v0 = 0.5, R = 1.0。

参考：
  G. Grégoire & H. Chaté, PRL 92, 025702 (2004)
  H. Chaté, Annu. Rev. Condens. Matter Phys. 11, 189 (2020)
  A. P. Solon, H. Chaté, J. Tailleur, PRL 114, 068101 (2015)

用法：
  python vicsek.py --selftest              # 正确性自检（邻域搜索 vs 暴力求和）
  python vicsek.py 128 2.0 0.5             # 单点运行：L=128, rho=2, eta=0.5
"""
import argparse
import time

import numpy as np


# ----------------------------------------------------------------------------
# 核心：精确邻域搜索（矢量化单元列表法 + 团簇退化自动切换 cKDTree）
# ----------------------------------------------------------------------------
def neighbor_velocity_sums(pos, vel, L, R=1.0):
    """对每个粒子 i 精确求和 sum_{j: |r_i - r_j| <= R} vel_j（最小镜像约定，含自身）。

    算法：单元列表（cell size = R），对 3x3 邻域单元的候选对做距离过滤。
    3x3 覆盖保证了所有距离 <= R 的粒子对一定被候选到（单元尺寸 R，轴向最多
    相差 1 个单元），距离过滤保证不把距离 > R 的粒子算进来。

    注意：粒子聚成致密团簇时（如 eta=0 低密度），候选对数随局部密度平方增长，
    此时自动切换到 x 排序循环窗口法（对团簇稳健）。正常情形不触发，性能无影响。
    """
    N = pos.shape[0]
    nc = int(round(L / R))
    cell = np.floor(pos / R).astype(np.int64)
    cell %= nc
    cid = cell[:, 0] * nc + cell[:, 1]

    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    vel_s = vel[order]
    pos_s = pos[order]
    counts = np.bincount(cid_s, minlength=nc * nc)
    starts = np.zeros(nc * nc, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    rows = cid_s // nc
    cols = cid_s % nc

    # 团簇退化检测：候选对总数远超均匀情形的 ~9*rho*N 时切换 x 排序窗口法
    cand_total = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nbr = ((rows + dx) % nc) * nc + ((cols + dy) % nc)
            cand_total += int(counts[nbr].sum())
    rho = N / (L * L)
    if cand_total > N * max(200.0, 90.0 * rho):
        return _neighbor_velocity_sums_xsort(pos, vel, L, R)

    idx = np.arange(N, dtype=np.int64)
    sum_v = np.zeros((N, 2), dtype=np.float64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nbr = ((rows + dx) % nc) * nc + ((cols + dy) % nc)
            n_counts = counts[nbr]                    # 每个粒子的候选数
            total = int(n_counts.sum())
            if total == 0:
                continue
            csum = np.cumsum(n_counts)
            seg_off = np.arange(total, dtype=np.int64) \
                - np.repeat(csum - n_counts, n_counts)
            cand = np.repeat(starts[nbr], n_counts) + seg_off   # 候选在排序数组中的下标
            owner = np.repeat(idx, n_counts)                     # 候选属于哪个粒子
            d = pos_s[cand] - pos_s[owner]
            d -= L * np.round(d / L)                 # 最小镜像（跨周期边界的邻居）
            mask = (d * d).sum(axis=1) <= R * R
            o, c = owner[mask], cand[mask]
            sum_v[:, 0] += np.bincount(o, weights=vel_s[c, 0], minlength=N)
            sum_v[:, 1] += np.bincount(o, weights=vel_s[c, 1], minlength=N)

    inv = np.empty(N, dtype=np.int64)
    inv[order] = idx
    return sum_v[inv]


def _neighbor_velocity_sums_xsort(pos, vel, L, R):
    """x 排序循环窗口法：精确邻域求和，对团簇构型比单元列表快数倍。

    按 x 排序后，与粒子 i 距离 <= R 的邻居必须满足 |dx| <= R，即落在排序数组的
    一个连续窗口内。用三重复制（x-L, x, x+L）把周期边界下的窗口变成连续区间
    （窗口宽 2R < L 保证每个粒子的三个副本中至多一个落入窗口，无重复计数），
    再对窗口内候选做一次完整距离过滤（y 方向用最小镜像）。
    """
    N = len(pos)
    order = np.argsort(pos[:, 0], kind="stable")
    xs = pos[order, 0]
    ys = pos[order, 1]
    vel_s = vel[order]
    xs3 = np.concatenate([xs - L, xs, xs + L])
    left = np.searchsorted(xs3, xs - R, side="left")
    right = np.searchsorted(xs3, xs + R, side="right")
    widths = right - left
    total = int(widths.sum())
    idx = np.arange(N, dtype=np.int64)
    seg_off = np.arange(total, dtype=np.int64) \
        - np.repeat(np.cumsum(widths) - widths, widths)
    cand = np.repeat(left, widths) + seg_off          # 在 xs3 上的下标
    owner = np.repeat(idx, widths)
    cand_mod = cand % N                               # 映射回 0..N-1
    dx = xs3[cand] - xs[owner]                        # |dx| <= R 已由窗口保证
    dy = ys[cand_mod] - ys[owner]
    dy -= L * np.round(dy / L)                        # y 方向最小镜像
    mask = dx * dx + dy * dy <= R * R
    o, c = owner[mask], cand_mod[mask]
    sums = np.zeros((N, 2), dtype=np.float64)
    sums[:, 0] = np.bincount(o, weights=vel_s[c, 0], minlength=N)
    sums[:, 1] = np.bincount(o, weights=vel_s[c, 1], minlength=N)
    inv = np.empty(N, dtype=np.int64)
    inv[order] = idx
    return sums[inv]


# ----------------------------------------------------------------------------
# 一步更新
# ----------------------------------------------------------------------------
def step(pos, vel, v0, R, L, eta, rng):
    local = neighbor_velocity_sums(pos, vel, L, R)
    norm = np.linalg.norm(local, axis=1)          # 邻域含自身 => norm > 0 恒成立
    u = local / norm[:, None]
    ang = rng.uniform(-eta * np.pi, eta * np.pi, size=len(pos))
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    ux = cos_a * u[:, 0] - sin_a * u[:, 1]
    uy = sin_a * u[:, 0] + cos_a * u[:, 1]
    vel_new = v0 * np.column_stack((ux, uy))
    pos_new = (pos + vel_new) % L
    return pos_new, vel_new


# ----------------------------------------------------------------------------
# 观测量
# ----------------------------------------------------------------------------
def order_parameter(vel, v0):
    """全局极性序参量 phi = |sum v_i| / (N v0)。"""
    return float(np.linalg.norm(vel.sum(axis=0)) / (len(vel) * v0))


def density_contrast(pos, vel, L, n_bins=64, rho=None):
    """沿全局平均运动方向的密度不均匀度 C = sqrt(Var(rho_bin)/rho^2 - 散粒噪声)。

    对任意方向的周期方盒，用解析弦长分布做几何基线校正得到每个 bin 的
    局域密度 rho_bin；其相对方差扣除泊松散粒噪声（每个 bin 的期望贡献
    1/(bin 面积 x rho)）。均匀相时 C ~ 0（无 bin 尺寸的假信号），
    带相（微相分离，密度双峰）时 C ~ O(1)。
    """
    vmean = vel.mean(axis=0)
    speed = np.linalg.norm(vmean)
    if speed < 1e-12:
        return float("nan")
    u = vmean / speed
    s = pos @ u
    s_lo = L * (min(u[0], 0.0) + min(u[1], 0.0))
    s = s - s_lo                                   # 归一化到 [0, L*(|ux|+|uy|))
    ux, uy = abs(u[0]), abs(u[1])
    smax = L * (ux + uy)
    edges = np.linspace(0.0, smax, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    if ux < 1e-10 or uy < 1e-10:                   # 方向与坐标轴平行：基线为常数
        base = np.full(n_bins, L)
    else:
        a = L * min(ux, uy)
        b = L * max(ux, uy)
        base = np.where(
            mids < a,
            mids / (ux * uy),
            np.where(mids <= b, a / (ux * uy), (smax - mids) / (ux * uy)),
        )
    counts, _ = np.histogram(s, bins=edges)
    area = (smax / n_bins) * base
    profile = counts / area                        # 局域密度估计 rho_bin
    rho_mean = rho if rho is not None else len(pos) / (L * L)
    var_rel = float(np.var(profile)) / (rho_mean ** 2)
    shot = float(np.mean(1.0 / (area * rho_mean)))  # 泊松散粒噪声的理论贡献
    return float(np.sqrt(max(var_rel - shot, 0.0)))


# ----------------------------------------------------------------------------
# 单次运行：瞬态 + 测量，返回全部观测量统计
# ----------------------------------------------------------------------------
def run(L, rho, eta, v0=0.5, R=1.0, n_transient=10000, n_measure=10000,
        subsample=10, snapshot_every=100, n_bins=64, seed=0, n_blocks=20):
    t0 = time.perf_counter()
    N = int(round(rho * L * L))
    rng = np.random.default_rng(seed)

    # 随机初态：均匀位置 + 随机方向（"从无序出发"协议）
    pos = rng.random((N, 2)) * L
    ang0 = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vel = v0 * np.column_stack((np.cos(ang0), np.sin(ang0)))

    for _ in range(n_transient):
        pos, vel = step(pos, vel, v0, R, L, eta, rng)

    phi_list, contrast_list = [], []
    phi_acc = phi2_acc = phi4_acc = 0.0
    for t in range(1, n_measure + 1):
        pos, vel = step(pos, vel, v0, R, L, eta, rng)
        if t % subsample == 0:
            phi = order_parameter(vel, v0)
            phi_list.append(phi)
            phi_acc += phi
            phi2_acc += phi * phi
            phi4_acc += phi * phi * phi * phi
        if t % snapshot_every == 0:
            contrast_list.append(density_contrast(pos, vel, L, n_bins, rho))

    phi_series = np.asarray(phi_list)
    n = len(phi_series)
    phi_mean = phi_acc / n
    phi2_mean = phi2_acc / n
    phi4_mean = phi4_acc / n
    chi = N * (phi2_mean - phi_mean ** 2)
    binder = 1.0 - phi4_mean / (2.0 * phi2_mean ** 2)

    # 分块（blocking）误差估计，消除时间关联对误差的低估
    n_b = min(n_blocks, n // 2)
    k = n // n_b
    blk = phi_series[: n_b * k].reshape(n_b, k)
    b1 = blk.mean(axis=1)
    b2 = (blk * blk).mean(axis=1)
    b4 = (blk ** 4).mean(axis=1)
    chi_blocks = N * (b2 - b1 ** 2)
    g_blocks = 1.0 - b4 / (2.0 * b2 ** 2)

    def jk(v):
        m = float(v.mean())
        e = float(np.sqrt(((v - m) ** 2).sum() * (n_b - 1) / n_b))
        return m, e

    chi_m, chi_e = jk(chi_blocks)
    g_m, g_e = jk(g_blocks)
    phi_m = float(b1.mean())
    phi_e = float(b1.std(ddof=1) / np.sqrt(n_b))
    contrast_arr = np.asarray(contrast_list, dtype=float)
    contrast = float(np.nanmean(contrast_arr)) if contrast_arr.size else float("nan")
    contrast_e = (float(np.nanstd(contrast_arr, ddof=1) / np.sqrt(contrast_arr.size))
                  if contrast_arr.size > 1 else float("nan"))

    return {
        "L": float(L), "rho": float(rho), "eta": float(eta), "v0": float(v0),
        "R": float(R), "N": N, "n_transient": n_transient, "n_measure": n_measure,
        "subsample": subsample, "seed": seed,
        "phi_mean": phi_m, "phi_err": phi_e,
        "phi2": phi2_mean, "phi4": phi4_mean,
        "chi": chi_m, "chi_err": chi_e,
        "binder": g_m, "binder_err": g_e,
        "contrast": contrast, "contrast_err": contrast_e,
        "walltime_s": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)

    # 1) 邻域搜索与暴力 O(N^2) 交叉验证（均匀构型 -> 单元列表路径）
    L, R, N = 16.0, 1.0, 512
    pos = rng.random((N, 2)) * L
    vel = rng.normal(size=(N, 2))
    fast = neighbor_velocity_sums(pos, vel, L, R)
    ref = np.zeros_like(fast)
    for i in range(N):
        d = pos - pos[i]
        d -= L * np.round(d / L)
        ref[i] = vel[(d * d).sum(axis=1) <= R * R].sum(axis=0)
    err = float(np.abs(fast - ref).max())
    print(f"[selftest] neighbor search (uniform) max abs err = {err:.3e}")
    assert err < 1e-9, "邻域搜索与暴力求和不一致"

    # 1b) 团簇构型 -> x 排序窗口路径的交叉验证
    L2, N2 = 64.0, 2048
    pos2 = rng.random((N2, 2)) * 6.0             # 90% 粒子挤在小区域
    pos2[:180] = rng.random((180, 2)) * L2
    vel2 = rng.normal(size=(N2, 2))
    fast2 = neighbor_velocity_sums(pos2, vel2, L2, R)
    ref2 = np.zeros_like(fast2)
    for i in range(N2):
        d = pos2 - pos2[i]
        d -= L2 * np.round(d / L2)
        ref2[i] = vel2[(d * d).sum(axis=1) <= R * R].sum(axis=0)
    err2 = float(np.abs(fast2 - ref2).max())
    print(f"[selftest] neighbor search (clustered) max abs err = {err2:.3e}")
    assert err2 < 1e-9, "团簇构型下邻域搜索与暴力求和不一致"

    # 2) 定性物理：低噪声应有序、高噪声应无序
    res = {}
    for eta in (0.1, 0.8):
        res[eta] = run(L=32.0, rho=2.0, eta=eta,
                       n_transient=3000, n_measure=2000, seed=7)
        r = res[eta]
        print(f"[selftest] L=32 rho=2 eta={eta}: "
              f"phi={r['phi_mean']:.4f}+/-{r['phi_err']:.4f} "
              f"chi={r['chi']:.3f} binder={r['binder']:.3f} "
              f"contrast={r['contrast']:.3f} ({r['walltime_s']:.1f}s)")
    assert res[0.1]["phi_mean"] > 0.8, "eta=0.1 应进入极性有序相"
    assert res[0.8]["phi_mean"] < 0.1, "eta=0.8 应为无序气相"
    print("[selftest] OK")


# ----------------------------------------------------------------------------
# 命令行：单点运行
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vicsek 模型（矢量噪声）单点运行")
    ap.add_argument("--selftest", action="store_true", help="运行自检后退出")
    ap.add_argument("L", nargs="?", type=float, default=None)
    ap.add_argument("rho", nargs="?", type=float, default=None)
    ap.add_argument("eta", nargs="?", type=float, default=None)
    ap.add_argument("--transient", type=int, default=10000)
    ap.add_argument("--measure", type=int, default=10000)
    ap.add_argument("--v0", type=float, default=0.5)
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.L is None or args.rho is None or args.eta is None:
        ap.error("需要 L rho eta 三个参数（或 --selftest）")
    res = run(args.L, args.rho, args.eta, v0=args.v0, R=args.R,
              n_transient=args.transient, n_measure=args.measure, seed=args.seed)
    keys = ["L", "rho", "eta", "phi_mean", "phi_err", "chi", "chi_err",
            "binder", "binder_err", "contrast", "contrast_err", "walltime_s"]
    print("  ".join(f"{k}={res[k]}" for k in keys))


if __name__ == "__main__":
    main()
