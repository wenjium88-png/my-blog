# -*- coding: utf-8 -*-
"""相图扫描驱动程序：并行扫 (L, rho, eta) 网格，断点续跑，结果逐点落盘 CSV。

用法：
  python scan.py --jobs 8                     # 标准网格（约一晚）
  python scan.py --jobs 8 --quick             # 快速预览：粗网格 + 1/4 步数
  python scan.py --sizes 64 --jobs 8          # 小尺寸演示（几分钟）
  python scan.py --dry-run                    # 只打印任务清单与耗时预算
  python scan.py --backend threads --jobs 8   # 线程后端（进程创建受限时）

输出：results/scan_summary.csv（每行一个 (L, rho, eta) 点的全部观测量）
已完成的点会自动跳过（断点续跑），重跑某点需删除对应 CSV 行。
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from vicsek import run

# ---------------- 默认网格 ----------------
DENSITIES = [0.5, 1.0, 2.0, 4.0]
# eta=0 是奇异点（无噪声 -> 低密度无限凝聚、无稳态且计算退化），默认从 0.05 起；
# 需要 eta=0 时用 --eta0（仅对 rho>=1 添加，那里成本正常）。
ETAS = np.arange(0.05, 0.800001, 0.05)        # 0.05 ~ 0.80, 步长 0.05
QUICK_ETAS = np.arange(0.05, 0.800001, 0.1)   # 快速模式：步长 0.10

# 尺寸 → (允许的密度上限列表, (瞬态步数, 测量步数))
SIZE_PLAN = {
    128.0: (DENSITIES, (10000, 10000)),
    256.0: ([0.5, 1.0, 2.0], (10000, 10000)),
    512.0: ([1.0, 2.0], (5000, 5000)),
}
# 演示/任意尺寸的默认步数（L=64 演示用）
FALLBACK_STEPS = (1000, 1000)

FIELDS = ["L", "rho", "eta", "v0", "R", "N", "n_transient", "n_measure",
          "subsample", "seed", "phi_mean", "phi_err", "phi2", "phi4",
          "chi", "chi_err", "binder", "binder_err",
          "contrast", "contrast_err", "walltime_s"]


def make_tasks(args):
    dens = list(DENSITIES)
    if args.dense:
        dens += [8.0]
    etas = QUICK_ETAS if args.quick else ETAS
    if args.eta0:
        etas = np.concatenate(([0.0], etas))
    sizes = [float(s) for s in args.sizes] if args.sizes else sorted(SIZE_PLAN)
    tasks = []
    for L in sizes:
        if L in SIZE_PLAN:
            dens_allowed, (tr, ms) = SIZE_PLAN[L]
            tr = max(tr // 4, 500) if args.quick else tr
            ms = max(ms // 4, 500) if args.quick else ms
        else:
            dens_allowed, (tr, ms) = dens, FALLBACK_STEPS
            tr = max(tr // 4, 500) if args.quick else tr
            ms = max(ms // 4, 500) if args.quick else ms
        for rho in dens:
            if rho not in dens_allowed:
                continue
            for eta in etas:
                if eta == 0.0 and rho < 1.0:
                    continue            # eta=0 低密度会无限凝聚，成本退化
                tasks.append({
                    "L": L, "rho": rho, "eta": float(eta),
                    "v0": 0.5, "R": 1.0,
                    "n_transient": tr, "n_measure": ms,
                    "subsample": 10, "snapshot_every": 100,
                    "n_bins": 64,
                    # 稳定可复现的种子（不用 hash()，其受 PYTHONHASHSEED 影响）
                    "seed": int(L * 1_000_000 + round(rho * 10_000) + round(eta * 1000)),
                })
    # 小任务（粒子数少）先跑，进度与 ETA 更平滑
    tasks.sort(key=lambda t: (t["rho"] * t["L"] ** 2 * t["n_measure"], t["L"]))
    return tasks


def worker(params):
    return run(**params)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:   # 兼容 BOM
            for row in csv.DictReader(f):
                done.add((float(row["L"]), float(row["rho"]), float(row["eta"]),
                          int(row["n_transient"]), int(row["n_measure"]),
                          int(row["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser(description="Vicsek 相图并行扫描")
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2 or 1))
    ap.add_argument("--quick", action="store_true", help="粗网格 + 短运行（快速预览）")
    ap.add_argument("--dense", action="store_true", help="加入 rho=8 的稠密点")
    ap.add_argument("--eta0", action="store_true",
                    help="加入 eta=0 的点（仅 rho>=1；rho<1 时该点无限凝聚、成本退化）")
    ap.add_argument("--sizes", nargs="+", type=float, help="只跑指定尺寸，如 64 128")
    ap.add_argument("--limit", type=int, default=0, help="最多跑前 K 个任务（调试用）")
    ap.add_argument("--task-stride", type=int, default=1,
                    help="每 K 个任务取 1 个（配合 --task-offset 做多进程分块）")
    ap.add_argument("--task-offset", type=int, default=0,
                    help="任务下标起点（0 起），配合 --task-stride")
    ap.add_argument("--dry-run", action="store_true", help="只打印任务清单")
    ap.add_argument("--backend", choices=["processes", "threads"],
                    default="processes", help="并行后端（默认多进程）")
    ap.add_argument("--out", default=os.path.join("results", "scan_summary.csv"))
    args = ap.parse_args()

    tasks = make_tasks(args)
    done = load_done(args.out)
    todo = [t for t in tasks
            if (t["L"], t["rho"], t["eta"], t["n_transient"], t["n_measure"], t["seed"])
            not in done]
    n_skip = len(tasks) - len(todo)
    if args.task_stride > 1:
        todo = todo[args.task_offset:: args.task_stride]
    if args.limit:
        todo = todo[: args.limit]

    est_steps = sum(t["rho"] * t["L"] ** 2 * (t["n_transient"] + t["n_measure"])
                    for t in todo)
    print(f"任务总数 {len(tasks)}（跳过已完成 {n_skip}，待跑 {len(todo)}），"
          f"粒子-步总数 ≈ {est_steps:.2e}")
    if args.dry_run:
        for t in todo:
            print(f"  L={t['L']:<6} rho={t['rho']:<4} eta={t['eta']:<5} "
                  f"N={int(t['rho']*t['L']**2):<8} "
                  f"steps={t['n_transient']}+{t['n_measure']}")
        return
    if not todo:
        print("全部任务已完成。")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)

    def append_row(res):
        nonlocal write_header
        with open(args.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(res)

    n_done = 0
    t_start = time.perf_counter()

    def on_result(res):
        nonlocal n_done
        append_row(res)
        n_done += 1
        elapsed = time.perf_counter() - t_start
        eta_sec = elapsed / n_done * (len(todo) - n_done)
        print(f"[{n_done}/{len(todo)}] L={res['L']:<6} rho={res['rho']:<4} "
              f"eta={res['eta']:<5} phi={res['phi_mean']:.4f} "
              f"contrast={res['contrast']:.3f} ({res['walltime_s']:.0f}s)  "
              f"ETA {eta_sec/60:.0f} min", flush=True)

    if args.backend == "threads":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = [pool.submit(worker, t) for t in todo]
            for fut in as_completed(futs):      # 乱序落盘，慢任务不阻塞其它点写入
                on_result(fut.result())
    else:
        with Pool(args.jobs) as pool:
            for res in pool.imap_unordered(worker, todo, chunksize=1):
                on_result(res)

    print(f"扫描完成：{n_done} 个新点已写入 {args.out}")


if __name__ == "__main__":
    main()
