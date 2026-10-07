// Vicsek 模型（角噪声）密度 ramp —— C++ 高性能版
//
// 与 Python 版 (vicsek_rect.ramp_run) 同一物理与同一协议：
//   θ_i(t+1) = Arg( Σ_{j: |r_i-r_j|<=R} v_j(t) ) + η ξ_i,  ξ_i ~ U(-π,π)
//   x_i(t+1) = x_i(t) + v0 e_i(t+1)                       (周期 Lx × Ly)
// 密度 ramp：逐点增删粒子（增补的粒子随机位置 + 随机取向），每点先弛豫后测量。
//
// 性能设计：
//   * SoA float 数组（x, y, cosθ, sinθ），内存流量减半
//   * 链表式单元列表 + 9 邻域单元索引 LUT（内层循环无取模、无分支）
//   * std::thread 按粒子分块并行（每粒子只写自己的输出槽，无竞争）
//   * xoshiro256** 每线程独立 RNG
//   * 无 atan2：直接用归一化邻居和旋转噪声角得到新的 cos/sin
//   * 逐密度点检查点（二进制），进程被杀可续跑
//
// 编译：
//   g++ -O3 -march=native -std=c++17 -pthread vicsek_ramp.cpp -o vicsek_ramp.exe
//
// 用法：
//   ./vicsek_ramp.exe --Lx 400 --Ly 48 --eta 0.4 \
//       --rho-start 4.6 --rho-end 2.0 --drho 0.2 --steps 1200 --pre-equil 4000 \
//       --seed 1 --out fig3d_down_s1.csv --ckpt fig3d_down_s1.bin [--threads 16]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------ RNG
struct Rng {                       // xoshiro256** （快且质量好，可复现）
    uint64_t s[4];
    explicit Rng(uint64_t seed) {
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s[i] = x ^ (x >> 31);
        }
    }
    static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t r = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }
    inline float u01() {            // [0,1)
        return (float)((next() >> 40) * 0x1.0p-24);
    }
    inline int below(int n) { return (int)(u01() * (float)n) % (n > 0 ? n : 1); }
};

// ------------------------------------------------------------------ 配置
struct Cfg {
    float Lx = 400.f, Ly = 48.f, R = 1.f, v0 = 0.5f;
    float eta = 0.4f;
    float rho_start = 4.6f, rho_end = 2.0f, drho = 0.2f;
    int steps = 1200, pre_equil = 4000;
    float measure_frac = 0.4f;
    int subsample = 20, snap_every = 100, nprof = 200;
    int seed = 1, threads = 0;      // 0 = hardware_concurrency
    float spread = 0.1f;            // 初态取向展宽（× π）
    bool seed_band = false;         // 人工带种子初态
    float rho_l = 0.5f, rho_h = 0.f;
    int n_bands = 2;
    std::string out = "ramp.csv", ckpt = "";
    bool aligned_ic = true;
};

// ------------------------------------------------------------------ 快速数学
// 关键修正：内层循环里 nearbyintf/floorf 若不被内联会退化成 libm 调用，
// 每次候选对 2 次调用 → 数十 ns/次，实测把 400x400 拖到 1.03 s/步。
// 这里用强制内联的取整/回绕版本（|v| < 0.5 与 |dx| < 1.5L 下完全等价）。
static inline float fast_round(float v) {
    return (float)(int)(v + (v >= 0.f ? 0.5f : -0.5f));
}
static inline float wrap_period(float v, float L) {
    if (v < 0.f) v += L;
    else if (v >= L) v -= L;
    return v;
}

// ------------------------------------------------------------------ 三角函数查表
// 噪声角均匀分布，用 4096 项表 + 线性插值（误差 ~1e-7），省掉每粒子每步 2 次 libm 调用
struct Trig {
    static const int NB = 4096;
    std::vector<float> sinT, cosT;
    bool use_std = false;
    Trig() : sinT(NB + 1), cosT(NB + 1) {
        use_std = (getenv("VICSEK_NOLUT") != nullptr);
        for (int i = 0; i <= NB; ++i) {
            double a = 2.0 * 3.14159265358979323846 * i / NB;
            sinT[i] = (float)std::sin(a);
            cosT[i] = (float)std::cos(a);
        }
    }
    inline void sincos(float ang, float& sa, float& ca) const {
        if (use_std) { sa = std::sin(ang); ca = std::cos(ang); return; }
        // 表坐标 = ang * NB/(2π)，再折回 [0, NB) —— 注意只能乘一次 NB！
        float t = ang * (float)(NB / (2.0 * 3.14159265358979323846));
        t -= NB * std::floor(t * (1.0f / NB));
        int i = (int)t;
        if (i >= NB) { i = NB - 1; }
        const float f = t - i;
        sa = sinT[i] + f * (sinT[i + 1] - sinT[i]);
        ca = cosT[i] + f * (cosT[i + 1] - cosT[i]);
    }
};

// ------------------------------------------------------------------ 单元网格（计数排序版）
struct Grid {
    int ncx = 0, ncy = 0, nc = 0;
    std::vector<int> nbr;        // 9 邻域单元索引 LUT
    std::vector<int> cnt, cstart;   // 计数 / 单元起始（cstart 长度 nc+1）
    // 每步复用的缓冲
    std::vector<int> cell, order;
    std::vector<float> xs, ys, cts, sts, nct, nst;

    void init(float Lx, float Ly, float R, int N) {
        ncx = (int)std::lround(Lx / R);
        ncy = (int)std::lround(Ly / R);
        nc = ncx * ncy;
        nbr.assign(9 * nc, 0);
        for (int cy = 0; cy < ncy; ++cy)
            for (int cx = 0; cx < ncx; ++cx) {
                int c = cx + ncx * cy, k = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ax = (cx + dx + ncx) % ncx, ay = (cy + dy + ncy) % ncy;
                        nbr[9 * c + k++] = ax + ncx * ay;
                    }
            }
        cnt.assign(nc, 0);
        cstart.assign(nc + 1, 0);
        cell.assign(N, 0); order.assign(N, 0);
        xs.assign(N, 0); ys.assign(N, 0); cts.assign(N, 0); sts.assign(N, 0);
        nct.assign(N, 0); nst.assign(N, 0);
    }
    void ensure(int N) {
        if ((int)cell.size() < N) {
            cell.assign(N, 0); order.assign(N, 0);
            xs.assign(N, 0); ys.assign(N, 0); cts.assign(N, 0); sts.assign(N, 0);
            nct.assign(N, 0); nst.assign(N, 0);
        }
    }
};

// ------------------------------------------------------------------ 状态
struct State {
    std::vector<float> x, y, ct, st;       // 位置 + cosθ/sinθ
    std::vector<int> cell;
    int N = 0;
};

static inline int cellOf(const Cfg& c, const Grid& g, float x, float y) {
    int cx = (int)(x / c.R); if (cx >= g.ncx) cx = g.ncx - 1; if (cx < 0) cx = 0;
    int cy = (int)(y / c.R); if (cy >= g.ncy) cy = g.ncy - 1; if (cy < 0) cy = 0;
    return cx + g.ncx * cy;
}

static void buildGrid(const Cfg& c, Grid& g, State& s) {
    // 兼容保留：仅用于需要单独建表的场合（当前主循环直接在 stepOnce 内建表）
    const int N = s.N;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) g.cell[i] = cellOf(c, g, s.x[i], s.y[i]);
}

// ------------------------------------------------------------------ 初态
// 均匀取向初态（aligned）或"人工带种子"：气体背景 rho_l + n_bands 条密度 rho_h 的板条，
// 直接给出共存两相结构，跳过成核等待（Python 版实测把收敛加速约 2 倍）。
static void initState(const Cfg& c, State& s, Rng& rng) {
    const int N0 = (int)std::lround(c.rho_start * c.Lx * c.Ly);
    s.N = N0;
    s.x.assign(N0, 0.f); s.y.assign(N0, 0.f);
    s.ct.assign(N0, 0.f); s.st.assign(N0, 0.f); s.cell.assign(N0, 0);
    if (!c.seed_band) {
        for (int i = 0; i < N0; ++i) {
            s.x[i] = rng.u01() * c.Lx;
            s.y[i] = rng.u01() * c.Ly;
            float ang = (c.aligned_ic ? (2.f * rng.u01() - 1.f) * c.spread
                                      : (2.f * rng.u01() - 1.f)) * 3.14159265358979323846f;
            s.ct[i] = std::cos(ang); s.st[i] = std::sin(ang);
        }
        return;
    }
    const float rho_h = std::max(c.rho_h > 0 ? c.rho_h : c.rho_start * 1.5f,
                                 c.rho_start * 1.05f);
    float frac = (c.rho_start - c.rho_l) / (rho_h - c.rho_l);
    frac = std::min(0.95f, std::max(0.05f, frac));
    const float spacing = c.Lx / c.n_bands;
    const float w = frac * spacing;
    int i = 0;
    while (i < N0) {
        const bool in_liq = (rng.u01() < frac);
        float x, y = rng.u01() * c.Ly;
        if (in_liq) {
            const int k = rng.below(c.n_bands);
            x = (k + 0.5f) * spacing + (rng.u01() - 0.5f) * w;
            x -= c.Lx * std::floor(x / c.Lx);
        } else {
            bool ok = false; float cand = 0.f;
            for (int tryi = 0; tryi < 64 && !ok; ++tryi) {
                cand = rng.u01() * c.Lx;
                float rel = cand - 0.5f * spacing;
                rel -= spacing * std::floor(rel / spacing);
                ok = std::fabs(rel) > 0.5f * w;
            }
            x = ok ? cand : rng.u01() * c.Lx;
        }
        s.x[i] = x; s.y[i] = y;
        const float ang = (2.f * rng.u01() - 1.f) * c.spread * 3.14159265358979323846f;
        s.ct[i] = std::cos(ang); s.st[i] = std::sin(ang);
        ++i;
    }
}

// ------------------------------------------------------------------ 一步更新
// 结构：① 计算每个粒子的单元号 ② 计数排序，把状态按单元重排到连续内存
//       ③ 邻居循环只访问连续区间（顺序访存、可预取）④ 更新位置
// 关键：②让 ③ 的访存从"随机散布"变成"连续"，这是大 N 下的主要性能来源。
static void stepOnce(const Cfg& c, Grid& g, State& s, std::vector<Rng>& rngs,
                     int nthreads, const Trig& trig) {
    const int N = s.N;
    const float R2 = c.R * c.R, invLx = 1.f / c.Lx, invLy = 1.f / c.Ly;
    const float noiseAmp = c.eta * 3.14159265358979323846f;
    g.ensure(N);

    // ① 单元号
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) g.cell[i] = cellOf(c, g, s.x[i], s.y[i]);

    // ② 计数排序（cell-major），并把状态搬进连续缓冲
    std::fill(g.cnt.begin(), g.cnt.begin() + g.nc, 0);
    for (int i = 0; i < N; ++i) ++g.cnt[g.cell[i]];
    int acc = 0;
    for (int k = 0; k < g.nc; ++k) { int t = g.cnt[k]; g.cstart[k] = acc; acc += t; }
    g.cstart[g.nc] = acc;
    // 串行 scatter（顺序访存，O(N)）
    {
        static thread_local std::vector<int> cur;
        cur.assign(g.cstart.begin(), g.cstart.begin() + g.nc);
        for (int i = 0; i < N; ++i) {
            int cc = g.cell[i];
            int p = cur[cc]++;
            g.xs[p] = s.x[i]; g.ys[p] = s.y[i];
            g.cts[p] = s.ct[i]; g.sts[p] = s.st[i];
            g.order[p] = cc;
        }
    }

    // ③ 邻居循环（连续内存）
    static std::vector<long long> tdbg;          // 诊断计数（静态，避免每步 malloc）
    if ((int)tdbg.size() < nthreads * 3) tdbg.assign(nthreads * 3, 0);
    std::fill(tdbg.begin(), tdbg.begin() + nthreads * 3, 0LL);
    auto worker = [&](int t) {
        Rng& rng = rngs[t];
        const int lo = (int)((long long)N * t / nthreads);
        const int hi = (int)((long long)N * (t + 1) / nthreads);
        const int* cstart = g.cstart.data();
        const int* nbr = g.nbr.data();
        const float* xs = g.xs.data(); const float* ys = g.ys.data();
        const float* cts = g.cts.data(); const float* sts = g.sts.data();
        long long cand = 0, acc_nb = 0, nzero = 0;
        for (int p = lo; p < hi; ++p) {
            const float xi = xs[p], yi = ys[p];
            float sx = 0.f, sy = 0.f;
            const int* nb = nbr + 9 * g.order[p];
            for (int k = 0; k < 9; ++k) {
                const int q0 = cstart[nb[k]], q1 = cstart[nb[k] + 1];
                for (int q = q0; q < q1; ++q) {
                    ++cand;
                    float dx = xs[q] - xi;  dx -= c.Lx * fast_round(dx * invLx);
                    float dy = ys[q] - yi;  dy -= c.Ly * fast_round(dy * invLy);
                    if (dx * dx + dy * dy <= R2) { sx += cts[q]; sy += sts[q]; ++acc_nb; }
                }
            }
            const float norm = std::sqrt(sx * sx + sy * sy);
            if (norm <= 0.f) ++nzero;
            const float ang = noiseAmp * (2.f * rng.u01() - 1.f);
            float sa, ca;
            trig.sincos(ang, sa, ca);
            float ux, uy;
            if (norm > 0.f) { ux = (sx * ca - sy * sa) / norm; uy = (sx * sa + sy * ca) / norm; }
            else            { ux = ca; uy = sa; }
            g.nct[p] = ux; g.nst[p] = uy;
        }
        tdbg[3 * t] = cand; tdbg[3 * t + 1] = acc_nb; tdbg[3 * t + 2] = nzero;
    };
    std::vector<std::thread> th;
    th.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) th.emplace_back(worker, t);
    for (auto& z : th) z.join();
    if (getenv("VICSEK_DEBUG")) {
        long long cand = 0, acc_nb = 0, nzero = 0;
        for (int t = 0; t < nthreads; ++t) {
            cand += tdbg[3 * t]; acc_nb += tdbg[3 * t + 1]; nzero += tdbg[3 * t + 2];
        }
        double ssum = 0, ssum2 = 0, sabs = 0;
        {
            Rng rr(999);
            const int M = 200000;
            for (int i = 0; i < M; ++i) {
                double x = 2.0 * rr.u01() - 1.0;
                ssum += x; ssum2 += x * x; sabs += std::fabs(x);
            }
            ssum /= M; ssum2 /= M; sabs /= M;
        }
        fprintf(stderr, "[dbg] N=%d 候选/粒子=%.1f 接受/粒子=%.2f 零邻居=%lld(%.1f%%)\n",
                N, (double)cand / N, (double)acc_nb / N, nzero, 100.0 * nzero / N);
        fprintf(stderr, "[dbg] 噪声抽样: mean=%.5f E[x^2]=%.5f(理论0.33333) mean|x|=%.5f(理论0.5)\n",
                ssum, ssum2, sabs);
        const float test_ang = noiseAmp * 0.5f;
        float tsa, tca;
        trig.sincos(test_ang, tsa, tca);
        fprintf(stderr, "[dbg] noiseAmp=%.4f  查表 sin(%.4f)=%.5f (std=%.5f) cos=%.5f (std=%.5f)\n",
                noiseAmp, test_ang, tsa, std::sin(test_ang), tca, std::cos(test_ang));
    }

    // ④ 位置更新 + 回绕（状态保持 cell-major 顺序，下一轮重新排序）
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        s.ct[i] = g.nct[i]; s.st[i] = g.nst[i];
        s.x[i] = wrap_period(g.xs[i] + c.v0 * g.nct[i], c.Lx);
        s.y[i] = wrap_period(g.ys[i] + c.v0 * g.nst[i], c.Ly);
    }
}

// ------------------------------------------------------------------ 观测量
struct Prof {
    std::vector<double> bin;         // 每 bin 的粒子数
};

static void measure(const Cfg& c, const State& s, double& phi, double& vm,
                    Prof& prof) {
    double sx = 0, sy = 0;
#pragma omp parallel for reduction(+ : sx, sy) schedule(static)
    for (int i = 0; i < s.N; ++i) { sx += s.ct[i]; sy += s.st[i]; }
    phi = std::sqrt(sx * sx + sy * sy) / (double)s.N;
    vm = phi * c.v0 * (double)s.N / ((double)c.Lx * c.Ly);

    prof.bin.assign(c.nprof, 0.0);
    const double binw = (double)c.Lx / c.nprof;
#pragma omp parallel
    {
        std::vector<double> loc(c.nprof, 0.0);
#pragma omp for schedule(static) nowait
        for (int i = 0; i < s.N; ++i) {
            int b = (int)(s.x[i] / binw);
            if (b < 0) b = 0; if (b >= c.nprof) b = c.nprof - 1;
            loc[b] += 1.0;
        }
#pragma omp critical
        for (int b = 0; b < c.nprof; ++b) prof.bin[b] += loc[b];
    }
}

static double profileVar(const Cfg& c, const Prof& p) {
    const double binw = (double)c.Lx / c.nprof, Ly = c.Ly;
    double mean = 0; std::vector<double> rho(p.bin.size());
    for (size_t b = 0; b < p.bin.size(); ++b) { rho[b] = p.bin[b] / (binw * Ly); mean += rho[b]; }
    mean /= (double)rho.size();
    double v = 0; for (double r : rho) v += (r - mean) * (r - mean);
    return v / (double)rho.size();
}

static void alignAdd(const Cfg& c, const Prof& p, std::vector<double>& acc, int& n) {
    const double binw = (double)c.Lx / c.nprof, Ly = c.Ly;
    std::vector<double> rho(p.bin.size());
    for (size_t b = 0; b < p.bin.size(); ++b) rho[b] = p.bin[b] / (binw * Ly);
    int arg = 0; for (size_t b = 1; b < rho.size(); ++b) if (rho[b] > rho[arg]) arg = (int)b;
    const int M = (int)rho.size();
    for (int b = 0; b < M; ++b) acc[b] += rho[(b + arg) % M];
    ++n;
}

// 带计数：平滑后的局部极大值，带衬度门槛与最小间距（与 Python 版同判据）
static int bandCount(const Cfg& c, const std::vector<double>& prof, double rho0) {
    const int M = (int)prof.size();
    if (M < 8) return 0;
    const int sm = 5;
    std::vector<double> p(M, 0.0);
    for (int b = 0; b < M; ++b) {
        double a = 0;
        for (int k = -sm / 2; k <= sm / 2; ++k) a += prof[((b + k) % M + M) % M];
        p[b] = a / sm;
    }
    double mn = p[0], mx = p[0];
    for (double v : p) { mn = std::min(mn, v); mx = std::max(mx, v); }
    if (mx - mn < 0.25 * rho0) return 0;
    const double thr = std::max(0.5 * (mn + mx), 1.15 * rho0);
    const int dist = std::max(3, (int)(M * 0.04));
    int n = 0, last = -1000000;
    for (int b = 0; b < M; ++b) {
        if (p[b] < thr) continue;
        double left = b - 1, right = b + 1;
        bool ismax = true;
        for (int k = 1; k <= dist && ismax; ++k) {
            if (p[((b - k) % M + M) % M] > p[b] || p[(b + k) % M] > p[b]) ismax = false;
        }
        if (ismax && b - last >= dist) { ++n; last = b; }
        (void)left; (void)right;
    }
    return n;
}

// ------------------------------------------------------------------ 粒子增删
static void resizeState(const Cfg& c, State& s, int Nt, Rng& rng) {
    if (Nt > s.N) {
        const int add = Nt - s.N, old = s.N;
        s.x.resize(Nt); s.y.resize(Nt); s.ct.resize(Nt); s.st.resize(Nt);
        s.cell.resize(Nt);
        for (int i = old; i < Nt; ++i) {
            s.x[i] = rng.u01() * c.Lx;
            s.y[i] = rng.u01() * c.Ly;
            const float ang = 2.f * 3.14159265358979323846f * rng.u01();
            s.ct[i] = std::cos(ang); s.st[i] = std::sin(ang);
        }
        s.N = Nt;
    } else if (Nt < s.N) {
        // 随机保留 Nt 个（原地部分 Fisher–Yates，避免 O(N) 选择开销）
        const int N = s.N;
        for (int i = 0; i < Nt; ++i) {
            int j = i + rng.below(N - i);
            std::swap(s.x[i], s.x[j]); std::swap(s.y[i], s.y[j]);
            std::swap(s.ct[i], s.ct[j]); std::swap(s.st[i], s.st[j]);
        }
        s.x.resize(Nt); s.y.resize(Nt); s.ct.resize(Nt); s.st.resize(Nt);
        s.cell.resize(Nt);
        s.N = Nt;
    }
}

// ------------------------------------------------------------------ 检查点
static void saveCkpt(const std::string& path, const State& s, int i_point,
                     const std::vector<double>& rho, const std::vector<double>& phi,
                     const std::vector<double>& vm, const std::vector<double>& dvar,
                     const std::vector<int>& nb, const std::vector<int>& Ns) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    const int n = (int)rho.size();
    fwrite("VSKC", 1, 4, f);
    fwrite(&i_point, sizeof(int), 1, f);
    fwrite(&s.N, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(s.x.data(), sizeof(float), s.N, f);
    fwrite(s.y.data(), sizeof(float), s.N, f);
    fwrite(s.ct.data(), sizeof(float), s.N, f);
    fwrite(s.st.data(), sizeof(float), s.N, f);
    fwrite(rho.data(), sizeof(double), n, f);
    fwrite(phi.data(), sizeof(double), n, f);
    fwrite(vm.data(), sizeof(double), n, f);
    fwrite(dvar.data(), sizeof(double), n, f);
    fwrite(nb.data(), sizeof(int), n, f);
    fwrite(Ns.data(), sizeof(int), n, f);
    fclose(f);
}

static bool loadCkpt(const std::string& path, State& s, int& i_point,
                     std::vector<double>& rho, std::vector<double>& phi,
                     std::vector<double>& vm, std::vector<double>& dvar,
                     std::vector<int>& nb, std::vector<int>& Ns) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4] = {0};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "VSKC", 4) != 0) { fclose(f); return false; }
    int n = 0;
    fread(&i_point, sizeof(int), 1, f);
    fread(&s.N, sizeof(int), 1, f);
    fread(&n, sizeof(int), 1, f);
    s.x.resize(s.N); s.y.resize(s.N); s.ct.resize(s.N); s.st.resize(s.N);
    s.cell.resize(s.N);
    fread(s.x.data(), sizeof(float), s.N, f);
    fread(s.y.data(), sizeof(float), s.N, f);
    fread(s.ct.data(), sizeof(float), s.N, f);
    fread(s.st.data(), sizeof(float), s.N, f);
    rho.resize(n); phi.resize(n); vm.resize(n); dvar.resize(n); nb.resize(n); Ns.resize(n);
    fread(rho.data(), sizeof(double), n, f);
    fread(phi.data(), sizeof(double), n, f);
    fread(vm.data(), sizeof(double), n, f);
    fread(dvar.data(), sizeof(double), n, f);
    fread(nb.data(), sizeof(int), n, f);
    fread(Ns.data(), sizeof(int), n, f);
    fclose(f);
    return true;
}

// ------------------------------------------------------------------ main
int main(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* k) { return std::string(argv[++i]); };
        if (a == "--Lx") c.Lx = std::stof(val("Lx"));
        else if (a == "--Ly") c.Ly = std::stof(val("Ly"));
        else if (a == "--R") c.R = std::stof(val("R"));
        else if (a == "--v0") c.v0 = std::stof(val("v0"));
        else if (a == "--eta") c.eta = std::stof(val("eta"));
        else if (a == "--rho-start") c.rho_start = std::stof(val("rho-start"));
        else if (a == "--rho-end") c.rho_end = std::stof(val("rho-end"));
        else if (a == "--drho") c.drho = std::stof(val("drho"));
        else if (a == "--steps") c.steps = std::stoi(val("steps"));
        else if (a == "--pre-equil") c.pre_equil = std::stoi(val("pre-equil"));
        else if (a == "--measure-frac") c.measure_frac = std::stof(val("measure-frac"));
        else if (a == "--subsample") c.subsample = std::stoi(val("subsample"));
        else if (a == "--snap-every") c.snap_every = std::stoi(val("snap-every"));
        else if (a == "--nprof") c.nprof = std::stoi(val("nprof"));
        else if (a == "--seed") c.seed = std::stoi(val("seed"));
        else if (a == "--threads") c.threads = std::stoi(val("threads"));
        else if (a == "--spread") c.spread = std::stof(val("spread"));
        else if (a == "--seed-band") c.seed_band = true;
        else if (a == "--no-seed-band") c.seed_band = false;
        else if (a == "--rho-l") c.rho_l = std::stof(val("rho-l"));
        else if (a == "--rho-h") c.rho_h = std::stof(val("rho-h"));
        else if (a == "--n-bands") c.n_bands = std::stoi(val("n-bands"));
        else if (a == "--out") c.out = val("out");
        else if (a == "--ckpt") c.ckpt = val("ckpt");
    }
    int nthreads = c.threads > 0 ? c.threads
                                 : (int)std::max(1u, std::thread::hardware_concurrency());
    std::vector<Rng> rngs;
    for (int t = 0; t < nthreads; ++t) rngs.emplace_back((uint64_t)c.seed * 1000003ULL + t);

    State s;
    int i_done = -1;
    std::vector<double> vrho, vphi, vvm, vdvar;
    std::vector<int> vnb, vN;
    bool resumed = false;
    if (!c.ckpt.empty())
        resumed = loadCkpt(c.ckpt, s, i_done, vrho, vphi, vvm, vdvar, vnb, vN);

    const int npts = std::max(2, (int)std::lround(std::fabs(c.rho_end - c.rho_start) / c.drho) + 1);
    std::vector<float> rho_grid(npts);
    for (int k = 0; k < npts; ++k)
        rho_grid[k] = c.rho_start + (c.rho_end - c.rho_start) * k / (npts - 1);

    if (!resumed) {
        Rng r0((uint64_t)c.seed);
        initState(c, s, r0);
        i_done = -1;
    }
    Grid g; g.init(c.Lx, c.Ly, c.R, std::max(s.N, (int)std::lround(c.rho_end * c.Lx * c.Ly) + 64));
    const Trig trig;
    if (!resumed) {
        for (int t = 0; t < c.pre_equil; ++t) stepOnce(c, g, s, rngs, nthreads, trig);
    } else {
        fprintf(stderr, "[ckpt] 续跑：已完成 %d/%d 点\n", i_done + 1, npts);
    }

    const int nsettle = std::max(1, (int)(c.steps * (1.f - c.measure_frac)));
    const int nmeas = std::max(1, c.steps - nsettle);
    std::vector<double> profacc(c.nprof, 0.0);
    int nacc = 0;
    Rng rmain((uint64_t)c.seed * 7919ULL + 13);

    for (int k = i_done + 1; k < npts; ++k) {
        const double rho = rho_grid[k];
        const int Nt = (int)std::lround(rho * c.Lx * c.Ly);
        resizeState(c, s, Nt, rmain);
        for (int t = 0; t < nsettle; ++t) stepOnce(c, g, s, rngs, nthreads, trig);

        std::vector<Prof> snaps;
        double phisum = 0; int phin = 0;
        std::vector<double> dvars;
        for (int t = 1; t <= nmeas; ++t) {
            stepOnce(c, g, s, rngs, nthreads, trig);
            if (t % c.subsample == 0) { double p, v; Prof pr; measure(c, s, p, v, pr); phisum += p; ++phin; }
            if (t % c.snap_every == 0) {
                double p, v; Prof pr; measure(c, s, p, v, pr);
                dvars.push_back(profileVar(c, pr));
                snaps.push_back(pr);
                alignAdd(c, pr, profacc, nacc);
            }
        }
        const double phi = phin ? phisum / phin : 0.0;
        double dv = 0; for (double z : dvars) dv += z; dv = dvars.empty() ? 0.0 : dv / dvars.size();
        // 用对齐后的时间平均剖面数带
        double profmean = 0; std::vector<double> pm(c.nprof, 0.0);
        for (int b = 0; b < c.nprof; ++b) { pm[b] = profacc[b] / std::max(1, nacc); profmean += pm[b]; }
        const int nb = bandCount(c, pm, rho);
        (void)profmean;
        vrho.push_back(rho); vphi.push_back(phi);
        vvm.push_back(phi * c.v0 * (double)s.N / ((double)c.Lx * c.Ly));
        vdvar.push_back(dv); vnb.push_back(nb); vN.push_back(s.N);
        printf("  rho0=%.3f N=%d phi=%.4f |v|=%.4f dvar=%.4f nb=%d\n",
               rho, s.N, phi, vvm.back(), dv, nb);
        fflush(stdout);
        if (!c.ckpt.empty())
            saveCkpt(c.ckpt, s, k, vrho, vphi, vvm, vdvar, vnb, vN);
    }

    FILE* fo = fopen(c.out.c_str(), "w");
    if (fo) {
        fprintf(fo, "rho,phi,vm,dvar,nb,N\n");
        for (size_t k = 0; k < vrho.size(); ++k)
            fprintf(fo, "%.6f,%.8f,%.8f,%.8f,%d,%d\n",
                    vrho[k], vphi[k], vvm[k], vdvar[k], vnb[k], vN[k]);
        fclose(fo);
    }
    // 对齐时间平均剖面
    FILE* fp = fopen((c.out + ".prof").c_str(), "w");
    if (fp) {
        for (int b = 0; b < c.nprof; ++b)
            fprintf(fp, "%.6f\n", profacc[b] / std::max(1, nacc));
        fclose(fp);
    }
    printf("完成：%zu 个密度点 -> %s\n", vrho.size(), c.out.c_str());
    return 0;
}
