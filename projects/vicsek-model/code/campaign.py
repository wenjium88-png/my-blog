# -*- coding: utf-8 -*-
"""PRL 114, 068101 (2015) Fig 2(a) / Fig 3(c,d) 的复现任务驱动。

复现目标（用户提供的论文截图）：
  截图 1 = Fig 3(c)：|v| vs ρ0（ρ0 ∈ [0.4,1.2]，η=0.4）气体↔微相分离滞后回线；
          Fig 3(d)：Δρ²∥ vs ρ0（ρ0 ∈ [2.0,4.5]）微相分离↔极性液体滞后回线。
  截图 2 = Fig 2(a)：binodal 相图 ρ_ℓ(η)（低密度/气体侧）与 ρ_h(η)（高密度/液体侧）。

任务类型：
  profile —— 在共存区内定密度长程运行，测量带对齐后的时间平均剖面：
             谷值 = ρ_gas = ρ_ℓ(η)，峰值 = ρ_band = ρ_h(η)（论文正文：
             带剖面在热力学极限下与 ρ0、系统尺寸无关）。
  ramp    —— 密度 ramp 协议，给出滞后回线（|v| 与 Δρ²∥ 两支）。

用法：
  python campaign.py --dry-run                 # 打印任务清单与预算
  python campaign.py --smoke                   # 小尺度快速验证（分钟级）
  python campaign.py --jobs 8                  # 生产运行（默认多进程）
  python campaign.py --task-stride 6 --task-offset 0 --jobs 1   # 分块（受限环境）
已完成任务（已有 npz）自动跳过。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vicsek_rect import ramp_run, run_fixed  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SUMMARY = os.path.join(OUTDIR, "summary.csv")
V0, R = 0.5, 1.0

# ---------------------------------------------------------------------------
# 生产任务定义
# ---------------------------------------------------------------------------
# 带质量/横向单位长度的估计（由论文 Fig 1/2 数据反推：800x100 盒子、η=0.4 时
# ρ0=1.05 有 1 条带、1.93 有 3 条带 => μ = L∥(ρ0-ρgas)/nb ≈ 360）。一条带能塞进
# 盒子的条件约为 L∥(ρ0-ρgas) ≳ μ，因此共存区取点需保证有 1~3 条带。
MU = 360.0
# 论文 Fig 2(a) 红线（低密度 binodal）的粗略读数，用于选取共存区内的取样密度
RHO_L_GUESS = {0.20: 0.06, 0.25: 0.10, 0.30: 0.18, 0.35: 0.28,
               0.40: 0.45, 0.45: 0.66, 0.50: 1.30, 0.53: 2.20}


def profile_tasks(smoke=False):
    """Fig 2(a)：每个 η 一次共存区长程运行 -> 带剖面 -> 两个 binodal 密度。"""
    if smoke:
        return [dict(kind="profile", tag="smoke_prof", Lx=128.0, Ly=32.0, rho=1.2,
                     eta=0.40, n_transient=800, n_measure=400, subsample=20,
                     snap_every=50, nprof=64, seed=1)]
    Lx, Ly = 400.0, 64.0
    tasks = []
    for eta, rl in sorted(RHO_L_GUESS.items()):
        # 取样密度：让盒子里约有 ~2 条带（excess = 2*mu/Lx ≈ 1.8）
        rho0 = round(rl + 2.0 * MU / Lx, 3)
        tasks.append(dict(kind="profile", tag=f"binodal_eta{eta:g}",
                          Lx=Lx, Ly=Ly, rho=rho0, eta=eta,
                          n_transient=15000, n_measure=10000, subsample=50,
                          snap_every=250, nprof=200, seed=int(1000 * eta)))
    return tasks


def ramp_tasks(smoke=False):
    """Fig 3(c,d)：η=0.4 的两条滞后回线。"""
    if smoke:
        return [
            dict(kind="ramp", tag="smoke_c_up", Lx=128.0, Ly=32.0, eta=0.40,
                 rho_start=0.40, rho_end=1.20, drho=0.20, steps_per_point=400,
                 pre_equil=0, seed=1),
            dict(kind="ramp", tag="smoke_c_down", Lx=128.0, Ly=32.0, eta=0.40,
                 rho_start=1.20, rho_end=0.40, drho=0.20, steps_per_point=400,
                 pre_equil=2000, seed=1),
            dict(kind="ramp", tag="smoke_d_up", Lx=128.0, Ly=32.0, eta=0.40,
                 rho_start=2.00, rho_end=4.60, drho=0.40, steps_per_point=400,
                 pre_equil=2000, seed=1),
            dict(kind="ramp", tag="smoke_d_down", Lx=128.0, Ly=32.0, eta=0.40,
                 rho_start=4.60, rho_end=2.00, drho=0.40, steps_per_point=400,
                 pre_equil=2000, seed=1),
        ]
    tasks = []
    for seed in (1, 2):
        # Fig 3(c)：气体 <-> 微相分离（L∥=400 与论文 400x400 一致，Ly 缩小省算力）
        tasks.append(dict(kind="ramp", tag="fig3c_up", Lx=400.0, Ly=100.0, eta=0.40,
                          rho_start=0.40, rho_end=1.25, drho=0.05,
                          steps_per_point=1500, pre_equil=0, seed=seed))
        tasks.append(dict(kind="ramp", tag="fig3c_down", Lx=400.0, Ly=100.0, eta=0.40,
                          rho_start=1.25, rho_end=0.40, drho=0.05,
                          steps_per_point=1500, pre_equil=3000, seed=seed))
        # Fig 3(d)：微相分离 <-> 极性液体（Ly 由 400 缩到 48 省算力：该转变由
        # 纵向带结构控制，横向尺寸不改变带剖面；高密度端 N 降到 ~1/8）
        tasks.append(dict(kind="ramp", tag="fig3d_up", Lx=400.0, Ly=48.0, eta=0.40,
                          rho_start=2.00, rho_end=4.60, drho=0.20,
                          steps_per_point=1200, pre_equil=3000, seed=seed))
        tasks.append(dict(kind="ramp", tag="fig3d_down", Lx=400.0, Ly=48.0, eta=0.40,
                          rho_start=4.60, rho_end=2.00, drho=0.20,
                          steps_per_point=1200, pre_equil=4000, seed=seed))
    return tasks


def all_tasks(smoke=False):
    tasks = ramp_tasks(smoke) + profile_tasks(smoke)
    if not smoke:
        tasks.sort(key=lambda t: t["Lx"] * t["Ly"]
                   * t.get("rho", t.get("rho_start", 0.0))
                   * (t.get("steps_per_point", 1000)
                      * (abs(t.get("rho_end", 1) - t.get("rho_start", 0)) / t.get("drho", 0.1) + 1)
                      if t["kind"] == "ramp"
                      else t.get("n_measure", 1000)))
    return tasks


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def run_task(t):
    name = t["tag"] + (f"_s{t['seed']}" if "seed" in t else "")
    path = os.path.join(OUTDIR, name + ".npz")
    if os.path.exists(path):
        return {"tag": t["tag"], "seed": t.get("seed", 0), "file": name + ".npz",
                "kind": t["kind"], "Lx": t.get("Lx"), "Ly": t.get("Ly"),
                "eta": t.get("eta"), "rho0": np.nan, "phi": np.nan, "vm": np.nan,
                "dvar": np.nan, "nb": np.nan, "prof_min": np.nan,
                "prof_max": np.nan, "skipped": 1, "walltime_s": 0.0}
    t0 = time.perf_counter()
    if t["kind"] == "profile":
        res = run_fixed(t["Lx"], t["Ly"], t["rho"], t["eta"], v0=V0, R=R,
                        n_transient=t["n_transient"], n_measure=t["n_measure"],
                        subsample=t["subsample"], snap_every=t["snap_every"],
                        nprof=t["nprof"], seed=t["seed"], aligned=True)
        np.savez_compressed(
            path, kind="profile", tag=t["tag"],
            Lx=res["Lx"], Ly=res["Ly"], rho=res["rho"], eta=res["eta"],
            phi=res["phi"], vm=res["vm"], dvar=res["dvar"], nb=res["nb"],
            theta_global=res["theta_global"],
            prof_mean=res["prof_mean"], prof_snaps=res["prof_snaps"])
        row = {"tag": t["tag"], "seed": t["seed"], "file": name + ".npz",
               "kind": "profile", "Lx": t["Lx"], "Ly": t["Ly"], "eta": t["eta"],
               "rho0": t["rho"], "phi": res["phi"], "vm": res["vm"],
               "dvar": res["dvar"], "nb": res["nb"],
               "prof_min": float(res["prof_mean"].min()),
               "prof_max": float(res["prof_mean"].max()),
               "walltime_s": res["walltime_s"], "skipped": 0}
    else:
        ckpt = os.path.join(OUTDIR, name + ".ckpt.npz")
        res = ramp_run(t["Lx"], t["Ly"], t["eta"], t["rho_start"], t["rho_end"],
                       t["steps_per_point"], v0=V0, R=R, nprof=int(t["Lx"] // 2),
                       seed=t["seed"], measure_frac=0.4, subsample=20,
                       snap_every=100, drho=t["drho"], pre_equil=t["pre_equil"],
                       ckpt=ckpt)
        np.savez_compressed(
            path, kind="ramp", tag=t["tag"], Lx=res["Lx"], Ly=res["Ly"],
            eta=res["eta"], rho=res["rho"], phi=res["phi"], vm=res["vm"],
            dvar=res["dvar"], nb=res["nb"], N=res["N"],
            rho_start=res["rho_start"], rho_end=res["rho_end"])
        row = {"tag": t["tag"], "seed": t["seed"], "file": name + ".npz",
               "kind": "ramp", "Lx": t["Lx"], "Ly": t["Ly"], "eta": t["eta"],
               "rho0": f"{t['rho_start']}->{t['rho_end']}",
               "phi": float(np.max(res["phi"])), "vm": float(np.max(res["vm"])),
               "dvar": float(np.max(res["dvar"])), "nb": int(np.max(res["nb"])),
               "prof_min": np.nan, "prof_max": np.nan,
               "walltime_s": res["walltime_s"], "skipped": 0}
    row["walltime_s"] = time.perf_counter() - t0
    return row


FIELDS = ["tag", "seed", "kind", "Lx", "Ly", "eta", "rho0", "phi", "vm", "dvar",
          "nb", "prof_min", "prof_max", "walltime_s", "skipped", "file"]


def append_row(row):
    os.makedirs(OUTDIR, exist_ok=True)
    new = not os.path.exists(SUMMARY)
    with open(SUMMARY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2 or 1))
    ap.add_argument("--smoke", action="store_true", help="小尺度快速验证")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", choices=["processes", "threads"], default="processes")
    ap.add_argument("--task-stride", type=int, default=1)
    ap.add_argument("--task-offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = all_tasks(args.smoke)
    if args.task_stride > 1:
        tasks = tasks[args.task_offset:: args.task_stride]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"任务数 {len(tasks)}")

    if args.dry_run:
        for t in tasks:
            print("  " + json.dumps(t, ensure_ascii=False))
        return

    os.makedirs(OUTDIR, exist_ok=True)
    if args.smoke or args.jobs == 1:
        for t in tasks:
            row = run_task(t)
            append_row(row)
            if row.get("skipped"):
                print(f"[skip] {row['tag']} seed={row['seed']} (已完成)", flush=True)
                continue
            print(f"[done] {row['tag']} seed={row['seed']} "
                  f"({row['walltime_s']:.0f}s) phi={row.get('phi', float('nan')):.3f} "
                  f"vm={row.get('vm', float('nan')):.4f} "
                  f"dvar={row.get('dvar', float('nan')):.3f} nb={row.get('nb')}",
                  flush=True)
        return

    if args.backend == "threads":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(run_task, t) for t in tasks]
            for fut in as_completed(futs):
                row = fut.result()
                append_row(row)
                print(f"[done] {row['tag']} seed={row['seed']} ({row['walltime_s']:.0f}s)",
                      flush=True)
    else:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for row in pool.imap_unordered(run_task, tasks, chunksize=1):
                append_row(row)
                print(f"[done] {row['tag']} seed={row['seed']} ({row['walltime_s']:.0f}s)",
                      flush=True)
    print("全部任务完成")


if __name__ == "__main__":
    main()
