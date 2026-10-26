/*
 * common11.hpp — experiment 11: traffic patterns, operation names, discard log.
 *
 * Work decomposition (host and device, identical code): an issuing CTA of rank r moves `payload` bytes per pass,
 * as E = payload / 16 elements of 16 B. plan_groups() splits its warps into groups; group g has one operation
 * (ld / st / st_async / tma_put / tma_red), one target rank (or kChunkTarget: a new partner per 16 KiB chunk), the
 * element range [e0, e1) it moves per pass and the warps [w0, w1) that execute it (TMA groups: one issuing thread,
 * lane 0 of warp w0). Patterns (cs = cluster size, k = pattern parameter):
 *   pair_uni        rank 0 → rank 1                                (others idle)
 *   pair_bi         0 → 1 and 1 → 0
 *   ring            r → (r + k) mod cs
 *   packed_pairs    stride k: r with (r / k) even → r + k            (k = 1: TPC mates; k = cs/2: across the cluster)
 *   broadcast_in    ranks 1 … k → rank 0                            (ld: k readers of rank 0; st/TMA: k writers into rank 0)
 *   broadcast_out   rank 0 → each of ranks 1 … k, the full payload to each (k warp groups; st/TMA only)
 *   all_to_all      every rank, cs − 1 warp groups, group g → rank (r + 1 + g) mod cs, 1/(cs − 1) of the payload each
 *   all_to_all_full as all_to_all, but every group moves the full payload (ld only)
 *   randperm        r → perm[r], perm a random derangement drawn per launch (same for all clusters of the launch)
 *   uniform_random  r → a new random rank ≠ r for every 16 KiB chunk and pass (hash of seed, sample, pass, r, chunk)
 *   mixed_same_dir  rank 0 warps [0,16) load from rank 1; rank 1 warps [16,32) store (or TMA-put) into rank 0
 *   mixed_opposite  rank 0 warps [0,16) load from rank 1; rank 0 warps [16,32) store (or TMA-put) into rank 1
 * Group warp ranges: single-group patterns use all warps of the CTA; multi-group patterns give floor(nwarps / G)
 * warps to each group (remaining warps idle); the TMA issuer of every group is lane 0 of its first warp.
 * The target side of st_async / TMA learns how many bytes every source sends it per pass by evaluating
 * plan_groups() for every source rank (incoming_bytes()).
 */
#pragma once
#include <cstdint>
#include <string>

#include "host.hpp"

enum Op11 { OP_LD = 0, OP_ST, OP_STA, OP_TMA, OP_TRED, OP_MIX_ST, OP_MIX_TMA, OP_N };
static const char* kOpName[] = {"ld", "st", "st_async", "tma_put", "tma_red", "mixed_ld_st", "mixed_ld_tma"};
enum Pat11 { PT_PAIR_UNI = 0, PT_PAIR_BI, PT_RING, PT_PACKED, PT_BCAST_IN, PT_BCAST_OUT, PT_A2A, PT_A2A_FULL, PT_RANDPERM,
             PT_UNIFORM, PT_MIX_SAME, PT_MIX_OPP, PT_N };
static const char* kPatName[] = {"pair_uni", "pair_bi", "ring", "packed_pairs", "broadcast_in", "broadcast_out", "all_to_all",
                                 "all_to_all_full", "randperm", "uniform_random", "mixed_same_dir", "mixed_opposite"};

constexpr int kMaxGroups = 15;
constexpr uint32_t kChunkTarget = 0xffu;          // per-16-KiB-chunk partner (uniform_random)
constexpr uint32_t kChunkElems = 16384u / 16u;    // elements per 16 KiB chunk

struct Group11 { uint32_t op, target, e0, e1, w0, w1; };
struct Perm16 { uint8_t p[16]; };

__host__ __device__ inline bool op_async(uint32_t op) { return op == OP_STA || op == OP_TMA || op == OP_TRED; }

__host__ __device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
// uniform_random partner of (rank, chunk) in pass `pass` of sample `sample`: uniformly one of the cs − 1 other ranks
__host__ __device__ inline uint32_t chunk_target(uint32_t seed, uint32_t sample, uint32_t pass, uint32_t rank, uint32_t chunk, uint32_t cs) {
    const uint32_t h = hash32(seed ^ hash32(sample * 0x9e3779b9u ^ hash32(pass * 0x85ebca6bu ^ hash32(rank * 977u + chunk))));
    return (rank + 1u + h % (cs - 1u)) % cs;
}

// Groups of the CTA with rank r (returns their number, 0 = idle as an issuer). op = kernel operation, nelem = payload / 16.
__host__ __device__ inline int plan_groups(uint32_t op, uint32_t pat, uint32_t cs, uint32_t k, uint32_t r, uint32_t nwarps, uint32_t nelem,
                                           const Perm16& perm, Group11* g) {
    auto one = [&](uint32_t o, uint32_t t, uint32_t w0, uint32_t w1) { g[0] = Group11{o, t, 0, nelem, w0, w1}; return 1; };
    switch (pat) {
    case PT_PAIR_UNI: return r == 0 ? one(op, 1, 0, nwarps) : 0;
    case PT_PAIR_BI: return r < 2 ? one(op, r ^ 1u, 0, nwarps) : 0;
    case PT_RING: return one(op, (r + k) % cs, 0, nwarps);
    case PT_PACKED: return ((r / k) % 2 == 0) ? one(op, r + k, 0, nwarps) : 0;
    case PT_BCAST_IN: return (r >= 1 && r <= k) ? one(op, 0, 0, nwarps) : 0;
    case PT_RANDPERM: return one(op, perm.p[r], 0, nwarps);
    case PT_UNIFORM: return one(op, kChunkTarget, 0, nwarps);
    case PT_BCAST_OUT:
    case PT_A2A:
    case PT_A2A_FULL: {
        if (pat == PT_BCAST_OUT && r != 0) return 0;
        const uint32_t G = pat == PT_BCAST_OUT ? k : cs - 1u, wpg = nwarps / G;
        for (uint32_t i = 0; i < G; ++i) {
            const uint32_t t = pat == PT_BCAST_OUT ? 1u + i : (r + 1u + i) % cs;
            const bool split = pat == PT_A2A;
            g[i] = Group11{op, t, split ? (uint32_t)((uint64_t)nelem * i / G) : 0u, split ? (uint32_t)((uint64_t)nelem * (i + 1) / G) : nelem,
                           i * wpg, (i + 1) * wpg};
        }
        return (int)G;
    }
    case PT_MIX_SAME:
    case PT_MIX_OPP: {
        const uint32_t second = op == OP_MIX_TMA ? (uint32_t)OP_TMA : (uint32_t)OP_ST, half = nwarps / 2;
        int n = 0;
        if (r == 0) g[n++] = Group11{OP_LD, 1, 0, nelem, 0, half};
        if (pat == PT_MIX_SAME && r == 1) g[n++] = Group11{second, 0, 0, nelem, half, nwarps};
        if (pat == PT_MIX_OPP && r == 0) g[n++] = Group11{second, 1, 0, nelem, half, nwarps};
        return n;
    }
    }
    return 0;
}

// Bytes per pass that CTA `me` receives from rank `src` through async operations (st_async / TMA): sum over src's groups.
__host__ __device__ inline uint32_t incoming_bytes(uint32_t op, uint32_t pat, uint32_t cs, uint32_t k, uint32_t src, uint32_t me,
                                                   uint32_t nwarps, uint32_t nelem, const Perm16& perm, uint32_t seed, uint32_t sample,
                                                   uint32_t pass) {
    Group11 g[kMaxGroups];
    const int n = plan_groups(op, pat, cs, k, src, nwarps, nelem, perm, g);
    uint32_t b = 0;
    for (int i = 0; i < n; ++i) {
        if (!op_async(g[i].op)) continue;
        if (g[i].target == kChunkTarget) {
            for (uint32_t c = g[i].e0 / kChunkElems; c * kChunkElems < g[i].e1; ++c)
                if (chunk_target(seed, sample, pass, src, c, cs) == me) {
                    const uint32_t a = c * kChunkElems > g[i].e0 ? c * kChunkElems : g[i].e0;
                    const uint32_t e = (c + 1) * kChunkElems < g[i].e1 ? (c + 1) * kChunkElems : g[i].e1;
                    b += (e - a) * 16u;
                }
        } else if (g[i].target == me) {
            b += (g[i].e1 - g[i].e0) * 16u;
        }
    }
    return b;
}

// ---------------------------------------------------------------- host: discarded launches
struct Discards11 {
    hb::Run& run;
    hb::CsvWriter csv;
    int n = 0;
    explicit Discards11(hb::Run& r)
        : run(r), csv(hb::Run::results("11__dsmem_bw__discards.csv"), hb::with_mandatory({"label", "reason"})) {}
    void add(const std::string& label, const char* reason, long long li, const hb::ClockSample& s) {
        csv.add(run.prefix(li, s) + "," + label + "," + reason);
        csv.sync();
        ++n;
        std::printf("  discarded launch (%s): %s\n", reason, label.c_str());
    }
};
