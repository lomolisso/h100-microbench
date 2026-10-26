/*
 * common12.hpp — experiment 12: pieces shared by atom_lat, atom_tput and dsmem_async.
 *
 * Chain encoding (K1 dependent chains). A chain element holds the position of the next element in a form that the op's
 * identity operand leaves unchanged, so a chain can be walked again (pass 2) without being rewritten. Position x = byte
 * offset / U (U = 4 smem, 8 global); next address = base + x·U (one IMAD per hop):
 *   32-bit words   0x4000_0000 | x                  (an f32 in [2, 4): normal; positive as s32; the tag wraps out of a
 *                                                     32-bit shared address, global bases are lowered by 2^33)
 *   64-bit words   0x4000_0000_0000_0000 | x        (an f64 in [2, 4); the address uses the low word)
 *   f16x2/bf16x2   two halves 0x2000 + 13 bits of x each (normal, finite in both formats), x < 2^26
 *   identity12(op) the operand that leaves a word unchanged: add +0, min.s32 INT_MAX, max.u32 0, and.b32 ~0; inc.u32
 *                  with bound ~0 and the add +1 control add 1, which moves the next access one unit further inside the
 *                  same element (their elements hold the encoding in every word: 64 B in smem, 128-B lines in global);
 *                  exch writes the previous element's word (the chain is reversed; walking it again from its end
 *                  restores it); cas compares with 0, which never matches (memory unchanged: "cas_fail").
 * Partitions. The L2 partition of every SM on this card comes from K0a (results/12__atomics__sm_partition.csv, latest
 *   run): Parts::part[smid]. Experiment 05's GPC → partition table (bw_common.hpp, measured on another card) is not used.
 * Home labels (K0b; every binary invocation labels its own buffer, because a line's home depends on its physical page).
 *   label_lines(): per probed line, L2 flush → whole-line write by SM A (one pinned warp) → one timed ld.global.cg by SM B
 *   (one pinned warp, dependent loads, t0 = %clock64, t1 = read_clock64_after(value)); then A and B swapped. A and B lie
 *   in different partitions (SMs 0 and 2 by default). Near for B and far for A ⇒ home = part[B]; far for B and near for
 *   A ⇒ home = part[A]; otherwise ambiguous (−1). Threshold = midpoint of a 2-means over all samples ≤ 1500 cycles.
 *   Homes are constant per 4 KiB on this GPU family, so large buffers are labelled from the first line of every 4 KiB page
 *   and the full-line labels of a prefix check that every line of a page agrees.
 * Launch bookkeeping: Discards12 (results/12__atomics__discards.csv), clock-checked launches, CSV helpers.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "atomics.cuh"
#include "../../04__memory_hierarchy_latency/src/pchase_chain.hpp"
#include "../../05__memory_hierarchy_bandwidth/src/bw_common.hpp"
#include "../../09__dsmem_latency_vs_cluster_size/src/common09.hpp"

static const char* EXP = "12__dsmem_atomics_and_async_stores";
constexpr uint32_t kLine = 128, kPage = 4096, kLinesPerPage = kPage / kLine;
constexpr uint64_t kMiB = 1ull << 20;
constexpr int kMaxClaimRetries = 50;

// ---------------------------------------------------------------- chain encoding
// x = the element's byte offset from the region base in units U (U = 4 for smem, 8 for global): the next address is
// base + x·U, one IMAD per hop. 32-bit words carry tag bit 30 (an f32 in [2, 4), positive as s32): in a 32-bit shared
// address 2^30·4 wraps to 0; for global the base is lowered by 2^30·8 = 2^33 (chain_base_adj). 64-bit words carry the
// tag in their high word and the address uses the low word. f16x2/bf16x2 words hold x as two halves 0x2000 + 13 bits
// (normal and finite in both formats), decoded with a few more instructions (their own same-encoding ld reference).
constexpr uint32_t kTag32 = 0x40000000u;
__host__ __device__ __forceinline__ bool op_is16x2(int op) {
    return op == AO_ADD_F16X2 || op == AO_ADD_BF16X2 || op == AO_ADD_V2F16X2 || op == AO_ADD_V4F16X2;
}
__host__ __device__ __forceinline__ uint64_t enc12(int op, uint32_t x) {
    if (op_is16x2(op)) return (uint64_t)((0x2000u + (x >> 13)) << 16 | (0x2000u + (x & 0x1FFFu)));
    if (op_is64(op)) return 0x4000000000000000ull | x;
    return kTag32 | x;
}
// the value a hop uses as its next x (32-bit words: tagged)
__host__ __device__ __forceinline__ uint32_t dec12(int op, uint64_t w) {
    if (op_is16x2(op)) return (uint32_t)(((w >> 16) & 0x1FFFu) << 13 | (w & 0x1FFFu));
    return (uint32_t)w;
}
__host__ __device__ __forceinline__ uint32_t tag12(int op) { return op_is16x2(op) || op_is64(op) ? 0u : kTag32; }
// global base so that base + x·8 is the address for tagged 32-bit words
__host__ __device__ __forceinline__ uint64_t chain_base_adj(int op, uint64_t buf) { return tag12(op) ? buf - (1ull << 33) : buf; }
// identity operand; add_one = the +1 control of add.u32. inc.u32 with bound ~0 adds 1 and add +1 adds 1: the next
// address moves by one unit inside the element (elements of ≥ 16 units hold the encoding in every word).
__host__ __device__ __forceinline__ uint64_t identity12(int op, bool add_one = false) {
    switch (op) {
        case AO_ADD_U32: return add_one ? 1u : 0u;
        case AO_INC_U32: return 0xFFFFFFFFu;
        case AO_MIN_S32: return 0x7FFFFFFFu;
        case AO_AND_B32: return 0xFFFFFFFFu;
        default: return 0;  // add (+0 in every format), max.u32 0, cas: new value (never written, compare 0 never matches)
    }
}

// ---------------------------------------------------------------- CSV / device helpers
inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
inline std::vector<std::string> split_csv(const std::string& l) {
    std::vector<std::string> v;
    std::stringstream ss(l);
    for (std::string c; std::getline(ss, c, ',');) v.push_back(c);
    return v;
}

struct Discards12 {
    hb::Run& run;
    hb::CsvWriter csv;
    int n = 0;
    explicit Discards12(hb::Run& r) : run(r), csv(hb::Run::results("12__atomics__discards.csv"), hb::with_mandatory({"label", "reason"})) {}
    void add(const std::string& label, const char* reason, long long li, const hb::ClockSample& s) {
        csv.add(run.prefix(li, s) + "," + label + "," + reason);
        csv.sync();
        ++n;
        std::printf("  discarded launch (%s): %s\n", reason, label.c_str());
    }
};

// ---------------------------------------------------------------- partitions (K0a output)
struct Parts {
    std::vector<int> gpc, part;  // per smid
    int gpc_part[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::string run_id;
    int first_sm(int p, int skip_gpc = -1) const {
        for (size_t s = 0; s < part.size(); ++s)
            if (part[s] == p && gpc[s] != skip_gpc) return (int)s;
        hb::die("no SM in partition " + std::to_string(p));
    }
    uint32_t gpc_mask(int p) const {
        uint32_t m = 0;
        for (int g = 0; g < 8; ++g) if (gpc_part[g] == p) m |= 1u << g;
        return m;
    }
};
inline Parts load_parts(int sm_count, bool require = true) {
    Parts P;
    P.gpc = load_gpc_map(sm_count);
    P.part.assign(sm_count, -1);
    std::ifstream f(hb::Run::results("12__atomics__sm_partition.csv"));
    if (!f) {
        if (require) hb::die("results/12__atomics__sm_partition.csv missing: run `atom_lat k0a` first");
        return P;
    }
    std::string line;
    std::getline(f, line);
    const auto h = split_csv(line);
    auto col = [&](const char* n) { for (size_t i = 0; i < h.size(); ++i) if (h[i] == n) return (int)i; hb::die(std::string("sm_partition lacks ") + n); };
    const int c_run = col("run_id"), c_sm = col("smid"), c_p = col("partition");
    std::vector<std::vector<std::string>> rows;
    while (std::getline(f, line)) {
        auto r = split_csv(line);
        if ((int)r.size() <= c_p) continue;
        if (r[c_run] > P.run_id) P.run_id = r[c_run];
        rows.push_back(std::move(r));
    }
    for (const auto& r : rows)
        if (r[c_run] == P.run_id) P.part[std::stoi(r[c_sm])] = std::stoi(r[c_p]);
    for (int s = 0; s < sm_count; ++s) {
        if (P.part[s] < 0) { if (require) hb::die("no partition for smid " + std::to_string(s)); continue; }
        int& gp = P.gpc_part[P.gpc[s]];
        if (gp >= 0 && gp != P.part[s]) std::printf("note: GPC %d has SMs in both partitions (smid %d)\n", P.gpc[s], s);
        gp = P.part[s];
    }
    return P;
}

// ---------------------------------------------------------------- labelling kernels (one pinned warp)
// Lines first + k·stride, k < n (line = 128 B). Whole-line writes (8 lanes × 16 B per line), value 0.
__global__ void __launch_bounds__(32) touch12(char* buf, uint64_t first, uint32_t stride, uint32_t n, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    for (uint32_t q = threadIdx.x; q < n * 8; q += 32) {
        const uint64_t a = (uint64_t)buf + (first + (uint64_t)(q / 8) * stride) * kLine + (q % 8) * 16;
        st_v4<C_WB>(a, 0u);
    }
}
// One timed dependent ld.global.cg.u32 per line (timer pair included in the value).
__global__ void __launch_bounds__(32) probe12(const char* buf, uint64_t first, uint32_t stride, uint32_t n, uint32_t target, uint32_t* claim,
                                              uint64_t zero, uint16_t* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    uint64_t dep = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t a = (uint64_t)buf + (first + (uint64_t)i * stride) * kLine + (dep & zero);
        const uint64_t t0 = read_clock64();
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
        const uint64_t t1 = read_clock64_after(v, zero);
        out[i] = (uint16_t)(t1 - t0 > 65535 ? 65535 : t1 - t0);
        dep = v;
    }
}

// The same for an explicit list of lines (indices into buf).
__global__ void __launch_bounds__(32) touch_list12(char* buf, const uint32_t* lines, uint32_t n, uint32_t target, uint32_t* claim) {
    if (!claim_target_sm(target, claim)) return;
    for (uint32_t q = threadIdx.x; q < n * 8; q += 32)
        st_v4<C_WB>((uint64_t)buf + (uint64_t)lines[q / 8] * kLine + (q % 8) * 16, 0u);
}
// t0 waits for the list entry (read before the timed load), so the list read is outside the timed interval.
__global__ void __launch_bounds__(32) probe_list12(const char* buf, const uint32_t* lines, uint32_t n, uint32_t target, uint32_t* claim,
                                                   uint64_t zero, uint16_t* out) {
    if (!claim_target_sm(target, claim)) return;
    if (threadIdx.x != 0) return;
    uint64_t dep = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t ln = lines[i + (uint32_t)(dep & zero)];
        const uint64_t a = (uint64_t)buf + (uint64_t)ln * kLine;
        const uint64_t t0 = read_clock64_after(ln, zero);
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
        const uint64_t t1 = read_clock64_after(v, zero);
        out[i] = (uint16_t)(t1 - t0 > 65535 ? 65535 : t1 - t0);
        dep = v;
    }
}

// Pinned one-warp launch with retries (claim_target_sm); F launches the kernel with the claim pointer.
template <class F>
inline void pinned(hb::SmClaim& claim, int sm_count, uint32_t sm, const char* what, F&& launch) {
    for (int t = 0; t < kMaxClaimRetries; ++t) {
        claim.reset();
        launch(hb::SmClaim::blocks(sm_count), claim.ptr());
        cudaError_t e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) hb::die(std::string(what) + ": " + cudaGetErrorString(e));
        if (claim.ok(sm)) return;
    }
    hb::die(std::string(what) + ": could not pin SM " + std::to_string(sm));
}

inline void two_means12(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, double cap, double& lo, double& hi) {
    std::vector<uint64_t> hist(65536, 0);
    for (auto v : a) ++hist[v];
    for (auto v : b) ++hist[v];
    lo = 200; hi = 500;
    for (int it = 0; it < 100; ++it) {
        const double thr = (lo + hi) / 2;
        double s0 = 0, n0 = 0, s1 = 0, n1 = 0;
        for (int v = 0; v <= cap; ++v) {
            if (!hist[v]) continue;
            if (v <= thr) { s0 += double(v) * hist[v]; n0 += hist[v]; } else { s1 += double(v) * hist[v]; n1 += hist[v]; }
        }
        const double nlo = n0 ? s0 / n0 : lo, nhi = n1 ? s1 / n1 : hi;
        if (std::fabs(nlo - lo) < 1e-6 && std::fabs(nhi - hi) < 1e-6) break;
        lo = nlo; hi = nhi;
    }
}

// Home labels of a buffer: page[] for every 4 KiB page (from its first line), line[] for every line of a prefix.
struct Homes {
    std::vector<int8_t> page, line;
    std::vector<uint16_t> pa, pb, la, lb;  // cycles: touch B probe A / touch A probe B, for pages and prefix lines
    double thr = 0, c_near = 0, c_far = 0;
    uint32_t A = 0, B = 0;
    int partA = 0, partB = 1;
    uint64_t mixed_pages = 0, line_page_disagree = 0, ambiguous_pages = 0;
    // pages homed in partition p within [first_page, first_page + n)
    std::vector<uint32_t> pages_of(int p, uint32_t first_page, uint32_t n) const {
        std::vector<uint32_t> v;
        for (uint32_t i = first_page; i < first_page + n && i < page.size(); ++i) if (page[i] == p) v.push_back(i);
        return v;
    }
};

// Label `n_pages` pages (first line of each) and the first `n_lines_full` lines of `buf`; chunks of 65 536 probes.
inline Homes label_buffer(hb::Run& run, hb::SmClaim& claim, hb::L2Flusher& flush, const Parts& P, char* buf, uint32_t n_pages,
                          uint32_t n_lines_full, uint32_t A = 0, uint32_t B = 2) {
    Homes H;
    H.A = A; H.B = B; H.partA = P.part[A]; H.partB = P.part[B];
    if (H.partA == H.partB || H.partA < 0) hb::die("labelling SMs must lie in different partitions");
    const int sm = run.dev().sm_count;
    constexpr uint32_t kChunk = 65536;
    uint16_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, kChunk * sizeof(uint16_t)));
    auto pass = [&](uint64_t first, uint32_t stride, uint32_t n, uint32_t toucher, uint32_t prober, std::vector<uint16_t>& dst) {
        dst.resize(n);
        for (uint32_t c = 0; c < n; c += kChunk) {
            const uint32_t m = std::min(kChunk, n - c);
            const uint64_t f0 = first + (uint64_t)c * stride;
            flush();
            pinned(claim, sm, toucher, "touch12", [&](int nb, uint32_t* cl) { touch12<<<nb, 32>>>(buf, f0, stride, m, toucher, cl); });
            pinned(claim, sm, prober, "probe12", [&](int nb, uint32_t* cl) { probe12<<<nb, 32>>>(buf, f0, stride, m, prober, cl, 0, d_out); });
            CUDA_CHECK(cudaMemcpy(dst.data() + c, d_out, m * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        }
    };
    pass(0, kLinesPerPage, n_pages, A, B, H.pb);
    pass(0, kLinesPerPage, n_pages, B, A, H.pa);
    pass(0, 1, n_lines_full, A, B, H.lb);
    pass(0, 1, n_lines_full, B, A, H.la);
    CUDA_CHECK(cudaFree(d_out));
    std::vector<uint16_t> all_b = H.pb, all_a = H.pa;
    all_b.insert(all_b.end(), H.lb.begin(), H.lb.end());
    all_a.insert(all_a.end(), H.la.begin(), H.la.end());
    two_means12(all_b, all_a, 1500, H.c_near, H.c_far);
    H.thr = (H.c_near + H.c_far) / 2;
    auto lab = [&](uint16_t cb, uint16_t ca) -> int8_t {
        const bool far_b = cb > H.thr, far_a = ca > H.thr;
        return (!far_b && far_a) ? (int8_t)H.partB : (far_b && !far_a) ? (int8_t)H.partA : (int8_t)-1;
    };
    H.page.resize(n_pages);
    for (uint32_t i = 0; i < n_pages; ++i) { H.page[i] = lab(H.pb[i], H.pa[i]); H.ambiguous_pages += H.page[i] < 0; }
    H.line.resize(n_lines_full);
    for (uint32_t i = 0; i < n_lines_full; ++i) H.line[i] = lab(H.lb[i], H.la[i]);
    for (uint32_t pg = 0; pg * kLinesPerPage < n_lines_full; ++pg) {
        bool h0 = false, h1 = false;
        for (uint32_t i = pg * kLinesPerPage; i < (pg + 1) * kLinesPerPage && i < n_lines_full; ++i) {
            h0 |= H.line[i] == 0; h1 |= H.line[i] == 1;
            H.line_page_disagree += H.line[i] >= 0 && H.page[pg] >= 0 && H.line[i] != H.page[pg];
        }
        H.mixed_pages += h0 && h1;
    }
    uint64_t n0 = 0, n1 = 0;
    for (auto h : H.page) { n0 += h == 0; n1 += h == 1; }
    std::printf("homes: near %.1f far %.1f thr %.1f; pages home0 %llu home1 %llu ambiguous %llu; prefix: 4 KiB pages with both homes "
                "%llu, lines disagreeing with their page %llu of %u\n",
                H.c_near, H.c_far, H.thr, (unsigned long long)n0, (unsigned long long)n1, (unsigned long long)H.ambiguous_pages,
                (unsigned long long)H.mixed_pages, (unsigned long long)H.line_page_disagree, n_lines_full);
    return H;
}

// One summary row per labelling (every invocation) → 12__atomics__homes_runs.csv
inline void write_homes_summary(hb::Run& run, const Homes& H, const std::string& buffer, uint64_t bytes) {
    hb::CsvWriter csv(hb::Run::results("12__atomics__homes_runs.csv"),
                      hb::with_mandatory({"buffer", "buffer_bytes", "toucher_a", "toucher_b", "part_a", "part_b", "c_near_cycles",
                                          "c_far_cycles", "thr_cycles", "pages", "pages_home0", "pages_home1", "pages_ambiguous",
                                          "prefix_lines", "prefix_mixed_pages", "prefix_line_page_disagree"}));
    uint64_t n0 = 0, n1 = 0;
    for (auto h : H.page) { n0 += h == 0; n1 += h == 1; }
    csv.add(run.prefix(-1, run.dev().sample()) +
            fmt(",%s,%llu,%u,%u,%d,%d,%.2f,%.2f,%.2f,%zu,%llu,%llu,%llu,%zu,%llu,%llu", buffer.c_str(), (unsigned long long)bytes, H.A, H.B,
                H.partA, H.partB, H.c_near, H.c_far, H.thr, H.page.size(), (unsigned long long)n0, (unsigned long long)n1,
                (unsigned long long)H.ambiguous_pages, H.line.size(), (unsigned long long)H.mixed_pages, (unsigned long long)H.line_page_disagree));
    csv.sync();
}

// 2 MiB-aligned device buffer (cudaMalloc returns 2 MiB-aligned blocks for large sizes; checked).
inline char* alloc_aligned(uint64_t bytes) {
    char* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    if ((uint64_t)p % (2 * kMiB)) hb::die("buffer not 2 MiB aligned");
    return p;
}
