/*
 * pchase_chain.hpp — the random cyclic pointer chain of experiment 04 (K2/K3), reused by 06 so that generic- and
 * async-proxy latency curves chase exactly the same order.
 *
 *   sattolo_cycle(n, seed)  next[i] for a uniformly random permutation of 0..n-1 that is ONE cycle of length n
 *                           (Sattolo's algorithm; mt19937_64(seed); bounded draws by Lemire's multiply-shift,
 *                           so the result depends only on (n, seed), not on the standard library).
 *   chase_order(next, h)    the first h elements visited from element 0: order[k] = element of hop k.
 *   kLineBytes = 128        one pointer per 128 B line (skip = 16 qwords): every hop touches a new L1/L2 line.
 *   kChainSeed              the seed used by every run of 04.
 * A chain over a buffer `base` stores, at qword 16·i, the device address base + 128·next[i]; the chase starts at
 * line 0, so line 0 is hop 0 and hop k reads line order[k].
 */
#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace chain {

constexpr uint64_t kLineBytes = 128;
constexpr uint64_t kQwordsPerLine = kLineBytes / sizeof(uint64_t);
constexpr uint64_t kChainSeed = 0x04ull << 56 | 20261023ull;

inline std::vector<uint32_t> sattolo_cycle(uint32_t n, uint64_t seed = kChainSeed) {
    std::vector<uint32_t> a(n);
    for (uint32_t i = 0; i < n; ++i) a[i] = i;
    std::mt19937_64 rng(seed);
    for (uint32_t i = n - 1; i >= 1; --i) {
        const uint32_t j = (uint32_t)(((unsigned __int128)rng() * i) >> 64);  // uniform in [0, i)
        std::swap(a[i], a[j]);
    }
    return a;  // a[i] = successor of i
}

inline std::vector<uint32_t> chase_order(const std::vector<uint32_t>& next, uint64_t hops) {
    std::vector<uint32_t> order(hops);
    uint32_t e = 0;
    for (uint64_t k = 0; k < hops; ++k) { order[k] = e; e = next[e]; }
    return order;
}

}  // namespace chain
