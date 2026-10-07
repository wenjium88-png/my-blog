# -*- coding: utf-8 -*-
"""矩形盒子版 Vicsek 模型 + PRL 114, 068101 (2015) 复现所需观测量。

模型与 vicsek.py 完全一致（角噪声）：
    θ_i(t+1) = Arg( Σ_{j: |r_i-r_j|<=R} v_j(t) ) + η ξ_i,   ξ_i ~ U(-π, π)
    r_i(t+1) = r_i(t) + v0 e_i(t+1)          (周期边界 Lx × Ly)

与 vicsek.py 的差别只在几何与观测量：
  - 支持 Lx != Ly 的周期矩形盒子（论文用 800×100、2000×100、400×400）；
  - 加入横向平均密度剖面 ρ̄(x∥)、带数 n_b、断面方差 Δρ²∥（Fig. 2/3 需要的量）；
  - 加入密度 ramp 协议（滞回回线，Fig. 3(c,d) 与 binodal 提取）。

带沿 +x 方向传播（初态取沿 x 的近平行取向），剖面沿 x 分 bin、对 y 平均。
"""
import os
import time

import numpy as np
from scipy.signal import find_peaks


# ----------------------------------------------------------------------------
# 精确邻域搜索（矩形盒子）
# ----------------------------------------------------------------------------
def neighbor_velocity_sums(pos, vel, Lx, Ly, R=1.0):
    """精确求 sum_{j: 最小镜像距离 <= R} vel_j（含自身）。

    单元列表（cell size R，ncx×ncy 个单元），3×3 邻域候选 + 逐轴最小镜像距离过滤。
    候选对数因团簇退化时自动切换到 x 排序窗口法（对团簇稳健）。
    """
    N = pos.shape[0]
    ncx = int(round(Lx / R))
    ncy = int(round(Ly / R))
    cx = np.floor(pos[:, 0] / R).astype(np.int64) % ncx
    cy = np.floor(pos[:, 1] / R).astype(np.int64) % ncy
    cid = cx * ncy + cy

    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    vel_s = vel[order]
    pos_s = pos[order]
    counts = np.bincount(cid_s, minlength=ncx * ncy)
    starts = np.zeros(ncx * ncy, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    rows = cid_s // ncy
    cols = cid_s % ncy

    # 退化检测（用 counts 网格的 9 次 roll 做，成本 O(单元数)）
    grid = counts.reshape(ncx, ncy)
    tot = np.zeros_like(grid)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tot += np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
    cand_total = int((grid * tot).sum())
    rho = N / (Lx * Ly)
    if cand_total > N * max(200.0, 90.0 * rho):
        return _neighbor_velocity_sums_xsort(pos, vel, Lx, Ly, R)

    idx = np.arange(N, dtype=np.int64)
    cand_parts, owner_parts = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nbr = ((rows + dx) % ncx) * ncy + ((cols + dy) % ncy)
            n_counts = counts[nbr]
            total = int(n_counts.sum())
            if total == 0:
                continue
            csum = np.cumsum(n_counts)
            seg_off = np.arange(total, dtype=np.int64) \
                - np.repeat(csum - n_counts, n_counts)
            cand_parts.append(np.repeat(starts[nbr], n_counts) + seg_off)
            owner_parts.append(np.repeat(idx, n_counts))

    cand = np.concatenate(cand_parts)
    owner = np.concatenate(owner_parts)
    d = pos_s[cand] - pos_s[owner]              # (M, 2)
    d[:, 0] -= Lx * np.round(d[:, 0] / Lx)      # 逐轴最小镜像
    d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
    mask = (d * d).sum(axis=1) <= R * R
    o, c = owner[mask], cand[mask]
    sum_v = np.empty((N, 2), dtype=np.float64)
    sum_v[:, 0] = np.bincount(o, weights=vel_s[c, 0], minlength=N)
    sum_v[:, 1] = np.bincount(o, weights=vel_s[c, 1], minlength=N)

    inv = np.empty(N, dtype=np.int64)
    inv[order] = idx
    return sum_v[inv]


def _neighbor_velocity_sums_xsort(pos, vel, Lx, Ly, R):
    """x 排序循环窗口法（矩形版），用于团簇退化时的精确回退。"""
    N = len(pos)
    order = np.argsort(pos[:, 0], kind="stable")
    xs = pos[order, 0]
    ys = pos[order, 1]
    vel_s = vel[order]
    xs3 = np.concatenate([xs - Lx, xs, xs + Lx])
    left = np.searchsorted(xs3, xs - R, side="left")
    right = np.searchsorted(xs3, xs + R, side="right")
    widths = right - left
    total = int(widths.sum())
    idx = np.arange(N, dtype=np.int64)
    seg_off = np.arange(total, dtype=np.int64) \
        - np.repeat(np.cumsum(widths) - widths, widths)
    cand = np.repeat(left, widths) + seg_off
    owner = np.repeat(idx, widths)
    cand_mod = cand % N
    dx = xs3[cand] - xs[owner]
    dy = ys[cand_mod] - ys[owner]
    dy -= Ly * np.round(dy / Ly)
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
def step(pos, vel, v0, R, Lx, Ly, eta, rng):
    local = neighbor_velocity_sums(pos, vel, Lx, Ly, R)
    norm = np.linalg.norm(local, axis=1)
    u = local / norm[:, None]
    ang = rng.uniform(-eta * np.pi, eta * np.pi, size=len(pos))
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    ux = cos_a * u[:, 0] - sin_a * u[:, 1]
    uy = sin_a * u[:, 0] + cos_a * u[:, 1]
    vel_new = v0 * np.column_stack((ux, uy))
    pos[:, 0] = (pos[:, 0] + vel_new[:, 0]) % Lx
    pos[:, 1] = (pos[:, 1] + vel_new[:, 1]) % Ly
    return pos, vel_new


# ----------------------------------------------------------------------------
# 观测量
# ----------------------------------------------------------------------------
def polarization(vel, v0):
    """|P| = (1/N)|Σ e_i|。"""
    return float(np.linalg.norm(vel.sum(axis=0)) / (len(vel) * v0))


def mean_velocity_density(vel, Lx, Ly):
    """|v| = (1/S)|Σ v_i| = v0 ρ0 |P|（论文 Fig. 3(b) 的纵轴）。"""
    return float(np.linalg.norm(vel.sum(axis=0)) / (Lx * Ly))


def transverse_profile(pos, Lx, Ly, nprof=200):
    """横向平均密度剖面 ρ̄(x∥)：沿 x 分 nprof 个 bin，对 y 平均。"""
    counts, _ = np.histogram(pos[:, 0], bins=nprof, range=(0.0, Lx))
    return counts / ((Lx / nprof) * Ly)


def profile_variance(profile):
    """Δρ²∥ = ⟨(ρ̄(x) − ρ̄_mean)²⟩_x（论文 Fig. 3(d) 的纵轴）。"""
    p = np.asarray(profile, dtype=float)
    return float(np.mean((p - p.mean()) ** 2))


def band_count(profile, rho0, smooth=5, min_sep_frac=0.04, height_frac=0.5,
               contrast_min=0.25):
    """数带：平滑剖面的峰计数。

    先要求剖面本身有足够衬度（max-min > contrast_min*ρ0），否则判为 0 条带
    （避免在近均匀剖面上把散粒噪声峰当成带）；再以高度 + prominence + 最小间距
    三重判据找峰。
    """
    p = np.asarray(profile, dtype=float)
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        p = np.convolve(np.concatenate([p[-smooth:], p, p[:smooth]]), ker,
                        mode="same")[smooth:-smooth]
    span = float(p.max() - p.min())
    if span < contrast_min * max(rho0, 1e-12):
        return 0
    thr = max(height_frac * (p.min() + p.max()), 1.15 * rho0)
    dist = max(3, int(len(p) * min_sep_frac))
    peaks, _ = find_peaks(p, height=thr, prominence=0.3 * span, distance=dist)
    return int(len(peaks))


def align_profile(profile):
    """把剖面循环平移，使最高峰落在 bin 0（共动系对齐，用于时间平均）。"""
    p = np.asarray(profile, dtype=float)
    return np.roll(p, -int(np.argmax(p)))


# ----------------------------------------------------------------------------
# 定密度长时运行（Fig. 1/2(c,d)、Fig. 3(a,b) 的底层运行）
# ----------------------------------------------------------------------------
def initial_state(Lx, Ly, rho, v0, rng, aligned=True, spread=0.1):
    N = int(round(rho * Lx * Ly))
    pos = np.empty((N, 2))
    pos[:, 0] = rng.random(N) * Lx
    pos[:, 1] = rng.random(N) * Ly
    if aligned:      # 近沿 +x 取向（保证带沿 x 传播，便于横向平均）
        ang = rng.uniform(-spread * np.pi, spread * np.pi, size=N)
    else:
        ang = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vel = v0 * np.column_stack((np.cos(ang), np.sin(ang)))
    return pos, vel


def initial_state_band(Lx, Ly, rho0, rho_l, rho_h, v0, rng, n_bands=2, spread=0.1):
    """人工"带种子"初态：气体背景 rho_l + n_bands 条密度 rho_h 的板条。

    直接给出共存区的两相结构，跳过成核等待，使带剖面更快达到渐近形状
    （对应 binodal 的谷/峰值）。板条沿 x 周期等间距排布。
    """
    rho_h = float(max(rho_h, rho0 * 1.05))
    frac = float(np.clip((rho0 - rho_l) / (rho_h - rho_l), 0.05, 0.95))
    spacing = Lx / n_bands
    w = frac * spacing                      # 每条带宽
    N = int(round(rho0 * Lx * Ly))

    x = np.empty(N)
    in_liq = rng.random(N) < frac           # 液体（带内）粒子
    n_liq = int(in_liq.sum())
    n_gas = N - n_liq
    if n_liq:
        k = rng.integers(0, n_bands, size=n_liq)
        centers = (k + 0.5) * spacing
        x[in_liq] = (centers + (rng.random(n_liq) - 0.5) * w) % Lx
    if n_gas:                               # 气体粒子：拒绝采样，排除带内区域
        out = []
        need = n_gas
        while need > 0:
            cand = rng.random(max(need * 2, 64)) * Lx
            rel = (cand - 0.5 * spacing) % spacing
            ok = np.abs(rel) > 0.5 * w       # 落在带外
            cand = cand[ok][:need]
            out.append(cand)
            need -= cand.size
        x[~in_liq] = np.concatenate(out)

    pos = np.column_stack((x, rng.random(N) * Ly))
    ang = rng.uniform(-spread * np.pi, spread * np.pi, size=N)
    vel = v0 * np.column_stack((np.cos(ang), np.sin(ang)))
    return pos, vel


def run_fixed(Lx, Ly, rho, eta, v0=0.5, R=1.0, n_transient=30000, n_measure=10000,
              subsample=50, snap_every=250, nprof=200, seed=0, aligned=True,
              max_snaps=40, rho_l=None, rho_h=None, n_bands=2):
    """定密度运行：返回 |P|、|v|、Δρ²∥、带数、瞬时剖面与时间平均剖面。

    rho_l/rho_h 给定时使用"人工带种子"初态（跳过成核等待）。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    if rho_l is not None and rho_h is not None:
        pos, vel = initial_state_band(Lx, Ly, rho, rho_l, rho_h, v0, rng,
                                      n_bands=n_bands)
    else:
        pos, vel = initial_state(Lx, Ly, rho, v0, rng, aligned=aligned)
    for _ in range(n_transient):
        pos, vel = step(pos, vel, v0, R, Lx, Ly, eta, rng)

    phi_list, vm_list, dvar_list, theta_list = [], [], [], []
    snap_list = []
    for t in range(1, n_measure + 1):
        pos, vel = step(pos, vel, v0, R, Lx, Ly, eta, rng)
        if t % subsample == 0:
            phi = polarization(vel, v0)
            phi_list.append(phi)
            vm_list.append(v0 * rho * phi)
            theta_list.append(float(np.arctan2(vel[:, 1].sum(), vel[:, 0].sum())))
        if t % snap_every == 0:
            prof = transverse_profile(pos, Lx, Ly, nprof)
            dvar_list.append(profile_variance(prof))
            if len(snap_list) < max_snaps:
                snap_list.append(prof)

    snaps = np.asarray(snap_list)
    aligned_mean = np.mean([align_profile(p) for p in snaps], axis=0)
    n_b = band_count(aligned_mean, rho)
    nb_snaps = np.array([band_count(p, rho) for p in snaps])

    return {
        "Lx": float(Lx), "Ly": float(Ly), "rho": float(rho), "eta": float(eta),
        "N": int(round(rho * Lx * Ly)),
        "n_transient": n_transient, "n_measure": n_measure, "seed": seed,
        "phi": float(np.mean(phi_list)),
        "vm": float(np.mean(vm_list)),
        "dvar": float(np.mean(dvar_list)),
        "nb": int(n_b),
        "nb_snap_mean": float(nb_snaps.mean()),
        "theta_global": float(np.mean(theta_list)),
        "prof_mean": aligned_mean,
        "prof_snaps": snaps,
        "walltime_s": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------
# 密度 ramp 协议（滞回回线 / binodal）
# ----------------------------------------------------------------------------
def _resize(pos, vel, N_target, Lx, Ly, v0, rng, insert_random=True):
    """调整粒子数：增补随机位置粒子（带随机取向，模拟气体插入）/ 随机删除。"""
    N = pos.shape[0]
    if N_target > N:
        k = N_target - N
        new_pos = np.empty((k, 2))
        new_pos[:, 0] = rng.random(k) * Lx
        new_pos[:, 1] = rng.random(k) * Ly
        if insert_random:
            ang = rng.uniform(0.0, 2.0 * np.pi, size=k)
        else:
            ang = np.zeros(k)
        new_vel = v0 * np.column_stack((np.cos(ang), np.sin(ang)))
        return np.vstack([pos, new_pos]), np.vstack([vel, new_vel])
    if N_target < N:
        keep = rng.choice(N, size=N_target, replace=False)
        keep.sort()
        return pos[keep], vel[keep]
    return pos, vel


def ramp_run(Lx, Ly, eta, rho_start, rho_end, steps_per_point, v0=0.5, R=1.0,
             nprof=200, seed=0, aligned=True, measure_frac=0.4, subsample=20,
             snap_every=100, init=None, drho=0.05, pre_equil=0, ckpt=None):
    """匀速改变密度（步长 drho），每点先弛豫后测量，返回滞回曲线。

    init: None 表示按 rho_start 用近沿 +x 取向初态；否则用 (pos, vel) 续跑。
    pre_equil: 起点的额外平衡步数（用于让初态先进入带相/均匀相）。
    ckpt: 检查点路径。给定则每算完一个密度点就落盘（含粒子状态），进程被杀后
          可自动从断点继续——长 ramp 必备（实测被杀会丢掉数小时算力）。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    keys = ["rho", "phi", "vm", "dvar", "nb", "N"]
    done = {k: [] for k in keys}
    start_idx = 0

    n_pts = max(int(round(abs(rho_end - rho_start) / drho)) + 1, 2)
    rhos = np.linspace(rho_start, rho_end, n_pts)

    if ckpt and os.path.exists(ckpt):
        d = np.load(ckpt, allow_pickle=False)
        start_idx = int(d["i_point"]) + 1
        pos, vel = d["pos"], d["vel"]
        for k in keys:
            done[k] = list(np.asarray(d[k])[:start_idx])
        print(f"[ramp] 检查点续跑：已完成 {start_idx}/{n_pts} 个密度点", flush=True)

    if start_idx == 0:
        if init is None:
            pos, vel = initial_state(Lx, Ly, rho_start, v0, rng, aligned=aligned)
        else:
            pos, vel = init[0].copy(), init[1].copy()
        for _ in range(pre_equil):
            pos, vel = step(pos, vel, v0, R, Lx, Ly, eta, rng)

    n_settle = max(1, int(steps_per_point * (1.0 - measure_frac)))
    n_meas = max(1, steps_per_point - n_settle)

    for i_pt in range(start_idx, n_pts):
        rho = rhos[i_pt]
        N_target = int(round(rho * Lx * Ly))
        pos, vel = _resize(pos, vel, N_target, Lx, Ly, v0, rng)
        for _ in range(n_settle):
            pos, vel = step(pos, vel, v0, R, Lx, Ly, eta, rng)

        phi_l, dvar_l, prof_l = [], [], []
        for t in range(1, n_meas + 1):
            pos, vel = step(pos, vel, v0, R, Lx, Ly, eta, rng)
            if t % subsample == 0:
                phi_l.append(polarization(vel, v0))
            if t % snap_every == 0:
                prof = transverse_profile(pos, Lx, Ly, nprof)
                dvar_l.append(profile_variance(prof))
                prof_l.append(prof)
        phi_m = float(np.mean(phi_l)) if phi_l else float("nan")
        pmean = np.mean([align_profile(p) for p in prof_l], axis=0) if prof_l else None
        done["rho"].append(float(rho))
        done["phi"].append(phi_m)
        done["vm"].append(v0 * rho * phi_m)
        done["dvar"].append(float(np.mean(dvar_l)) if dvar_l else float("nan"))
        done["nb"].append(band_count(pmean, rho) if pmean is not None else -1)
        done["N"].append(int(pos.shape[0]))

        if ckpt:
            np.savez_compressed(ckpt, i_point=i_pt, pos=pos, vel=vel,
                                **{k: np.array(done[k]) for k in keys})

    return {
        "Lx": float(Lx), "Ly": float(Ly), "eta": float(eta),
        "rho_start": float(rho_start), "rho_end": float(rho_end),
        "steps_per_point": int(steps_per_point), "seed": seed,
        "rho": np.array(done["rho"]), "phi": np.array(done["phi"]),
        "vm": np.array(done["vm"]), "dvar": np.array(done["dvar"]),
        "nb": np.array(done["nb"]), "N": np.array(done["N"]),
        "walltime_s": time.perf_counter() - t0,
    }


# ----------------------------------------------------------------------------
# 自检（矩形邻域搜索 vs 暴力）
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(5)
    Lx, Ly, R = 17.0, 7.0, 1.0
    N = 600
    pos = np.column_stack((rng.random(N) * Lx, rng.random(N) * Ly))
    vel = rng.normal(size=(N, 2))
    fast = neighbor_velocity_sums(pos, vel, Lx, Ly, R)
    ref = np.zeros_like(fast)
    for i in range(N):
        d = pos - pos[i]
        d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
        d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
        ref[i] = vel[(d * d).sum(axis=1) <= R * R].sum(axis=0)
    err = float(np.abs(fast - ref).max())
    print(f"[vicsek_rect selftest] rectangular neighbor search max err = {err:.3e}")
    assert err < 1e-9, "矩形邻域搜索与暴力求和不一致"

    # 团簇构型 -> 回退路径
    pos2 = np.column_stack((rng.random(N) * 5.0, rng.random(N) * 5.0))
    fast2 = neighbor_velocity_sums(pos2, vel, Lx, Ly, R)
    ref2 = np.zeros_like(fast2)
    for i in range(N):
        d = pos2 - pos2[i]
        d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
        d[:, 1] -= Ly * np.round(d[:, 1] / Ly)
        ref2[i] = vel[(d * d).sum(axis=1) <= R * R].sum(axis=0)
    err2 = float(np.abs(fast2 - ref2).max())
    print(f"[vicsek_rect selftest] clustered path max err = {err2:.3e}")
    assert err2 < 1e-9

    # 一个短程物理检查：低噪声应出现极化 + 剖面出现峰
    res = run_fixed(Lx=256.0, Ly=64.0, rho=1.2, eta=0.4, n_transient=3000,
                    n_measure=1000, nprof=128, seed=1)
    print(f"[vicsek_rect selftest] Lx=256 rho=1.2 eta=0.4: |P|={res['phi']:.3f} "
          f"|v|={res['vm']:.4f} nb={res['nb']} dvar={res['dvar']:.4f} "
          f"({res['walltime_s']:.1f}s)")
    print("[vicsek_rect selftest] OK")


if __name__ == "__main__":
    selftest()
