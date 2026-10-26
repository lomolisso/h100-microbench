/*
 * atomics.cuh — templated atom / red wrappers for every (operation, state space, semantics) used by experiment 12.
 *
 *   atomx<OP, SP, SEM>(a, v, c)  atom{SEM}.{SP}.{OP}: returns the old value as bits (32-bit ops in the low word).
 *                                v = operand bits, c = compare value (cas only). Vector ops return word 0 in the low
 *                                32 bits and the XOR of the other returned words in the high 32 bits, so that every
 *                                destination register is used (ptxas turns an atom whose results are unused into RED).
 *   redx<OP, SP, SEM>(a, v)      red{SEM}.{SP}.{OP} (no return; not defined for exch / cas).
 *   SP   AS_CTA = .shared::cta, AS_CLUSTER = .shared::cluster (32-bit addresses; mapa for peers), AS_GLOBAL = .global
 *        (64-bit generic address of global memory).
 *   SEM  index into {"", .relaxed.{cta,cluster,gpu,sys}, .acq_rel.{cta,cluster,gpu,sys}}; red uses .release in place
 *        of .acq_rel (red has no acquire side). "" = the default (relaxed, .gpu scope).
 *   OP   the operations swept by experiment 12 (kOpName gives the CSV label): add.u32/u64, inc.u32 (bound operand),
 *        min.s32, max.u32, and.b32, exch.b32, cas.b32/b64, add.f32/f64, add.noftz.f16x2/bf16x2 (packed in a .b32),
 *        and the global-only vector forms add.v2/v4.f32, add.noftz.v2/v4.f16x2.
 * Every form was assembled with ptxas 12.9 for sm_90a (experiment 12 notebook lists the SASS each one becomes: native
 * ATOMS / ATOM.E / ATOMG / REDG, or a compare-and-swap loop). Each call is one asm volatile statement with a "memory"
 * clobber; floating-point operands are passed through the matching register class (f32 "f", f64 "d").
 */
#pragma once
#include <cstdint>

enum AtomSpace { AS_CTA = 0, AS_CLUSTER = 1, AS_GLOBAL = 2 };
enum AtomOp {
    AO_ADD_U32 = 0, AO_ADD_U64, AO_INC_U32, AO_MIN_S32, AO_MAX_U32, AO_AND_B32, AO_EXCH_B32, AO_CAS_B32, AO_CAS_B64,
    AO_ADD_F32, AO_ADD_F64, AO_ADD_F16X2, AO_ADD_BF16X2, AO_ADD_V2F32, AO_ADD_V4F32, AO_ADD_V2F16X2, AO_ADD_V4F16X2, AO_COUNT
};
static const char* const kOpName[AO_COUNT] = {"add.u32", "add.u64", "inc.u32", "min.s32", "max.u32", "and.b32", "exch.b32",
                                              "cas.b32", "cas.b64", "add.f32", "add.f64", "add.f16x2", "add.bf16x2",
                                              "add.v2.f32", "add.v4.f32", "add.v2.f16x2", "add.v4.f16x2"};
static const char* const kSemName[9] = {"default", "relaxed.cta", "relaxed.cluster", "relaxed.gpu", "relaxed.sys",
                                        "acq_rel.cta", "acq_rel.cluster", "acq_rel.gpu", "acq_rel.sys"};
// 64-bit memory words (chain encoding, strides)
__host__ __device__ constexpr bool op_is64(int op) { return op == AO_ADD_U64 || op == AO_CAS_B64 || op == AO_ADD_F64; }
__host__ __device__ constexpr bool op_has_red(int op) { return op != AO_EXCH_B32 && op != AO_CAS_B32 && op != AO_CAS_B64; }
__host__ __device__ constexpr bool op_is_vector(int op) { return op >= AO_ADD_V2F32; }
__host__ __device__ constexpr int op_vec_words(int op) {
    return op == AO_ADD_V2F32 || op == AO_ADD_V2F16X2 ? 2 : op == AO_ADD_V4F32 || op == AO_ADD_V4F16X2 ? 4 : 1; }
__host__ __device__ constexpr int op_bytes(int op) { return op_is64(op) ? 8 : 4 * op_vec_words(op); }

// ---- semantics × space dispatch: F(SEMSTR, SPACESTR, ADDR_CONSTRAINT, ADDR_VAR) is one asm statement
#define AT12_SEM(SEM, F, SPS, AC, AV)                                                   \
    if constexpr (SEM == 0) { F("", SPS, AC, AV); }                                     \
    else if constexpr (SEM == 1) { F(".relaxed.cta", SPS, AC, AV); }                    \
    else if constexpr (SEM == 2) { F(".relaxed.cluster", SPS, AC, AV); }                \
    else if constexpr (SEM == 3) { F(".relaxed.gpu", SPS, AC, AV); }                    \
    else if constexpr (SEM == 4) { F(".relaxed.sys", SPS, AC, AV); }                    \
    else if constexpr (SEM == 5) { F(".acq_rel.cta", SPS, AC, AV); }                    \
    else if constexpr (SEM == 6) { F(".acq_rel.cluster", SPS, AC, AV); }                \
    else if constexpr (SEM == 7) { F(".acq_rel.gpu", SPS, AC, AV); }                    \
    else { F(".acq_rel.sys", SPS, AC, AV); }
#define RD12_SEM(SEM, F, SPS, AC, AV)                                                   \
    if constexpr (SEM == 0) { F("", SPS, AC, AV); }                                     \
    else if constexpr (SEM == 1) { F(".relaxed.cta", SPS, AC, AV); }                    \
    else if constexpr (SEM == 2) { F(".relaxed.cluster", SPS, AC, AV); }                \
    else if constexpr (SEM == 3) { F(".relaxed.gpu", SPS, AC, AV); }                    \
    else if constexpr (SEM == 4) { F(".relaxed.sys", SPS, AC, AV); }                    \
    else if constexpr (SEM == 5) { F(".release.cta", SPS, AC, AV); }                    \
    else if constexpr (SEM == 6) { F(".release.cluster", SPS, AC, AV); }                \
    else if constexpr (SEM == 7) { F(".release.gpu", SPS, AC, AV); }                    \
    else { F(".release.sys", SPS, AC, AV); }
#define X12_SPACE(SEMSW, SP, SEM, F)                                                    \
    if constexpr (SP == AS_CTA) { SEMSW(SEM, F, "shared::cta", "r", a32) }              \
    else if constexpr (SP == AS_CLUSTER) { SEMSW(SEM, F, "shared::cluster", "r", a32) } \
    else { SEMSW(SEM, F, "global", "l", a) }

// ---- atom bodies (d32/d64/df/dd = destination, v32/... = operand, c32/c64 = compare)
#define A12_ADD_U32(S, P, AC, AV) asm volatile("atom" S "." P ".add.u32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_INC_U32(S, P, AC, AV) asm volatile("atom" S "." P ".inc.u32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_MIN_S32(S, P, AC, AV) asm volatile("atom" S "." P ".min.s32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_MAX_U32(S, P, AC, AV) asm volatile("atom" S "." P ".max.u32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_AND_B32(S, P, AC, AV) asm volatile("atom" S "." P ".and.b32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_EXCH_B32(S, P, AC, AV) asm volatile("atom" S "." P ".exch.b32 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_CAS_B32(S, P, AC, AV) asm volatile("atom" S "." P ".cas.b32 %0, [%1], %2, %3;" : "=r"(d32) : AC(AV), "r"(c32), "r"(v32) : "memory")
#define A12_ADD_F16X2(S, P, AC, AV) asm volatile("atom" S "." P ".add.noftz.f16x2 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_ADD_BF16X2(S, P, AC, AV) asm volatile("atom" S "." P ".add.noftz.bf16x2 %0, [%1], %2;" : "=r"(d32) : AC(AV), "r"(v32) : "memory")
#define A12_ADD_U64(S, P, AC, AV) asm volatile("atom" S "." P ".add.u64 %0, [%1], %2;" : "=l"(d64) : AC(AV), "l"(v) : "memory")
#define A12_CAS_B64(S, P, AC, AV) asm volatile("atom" S "." P ".cas.b64 %0, [%1], %2, %3;" : "=l"(d64) : AC(AV), "l"(c), "l"(v) : "memory")
#define A12_ADD_F32(S, P, AC, AV) asm volatile("atom" S "." P ".add.f32 %0, [%1], %2;" : "=f"(df) : AC(AV), "f"(vf) : "memory")
#define A12_ADD_F64(S, P, AC, AV) asm volatile("atom" S "." P ".add.f64 %0, [%1], %2;" : "=d"(dd) : AC(AV), "d"(vd) : "memory")
#define A12_ADD_V2F32(S, P, AC, AV) asm volatile("atom" S "." P ".add.v2.f32 {%0, %1}, [%2], {%3, %3};" \
                                                  : "=f"(f0), "=f"(f1) : AC(AV), "f"(vf) : "memory")
#define A12_ADD_V4F32(S, P, AC, AV) asm volatile("atom" S "." P ".add.v4.f32 {%0, %1, %2, %3}, [%4], {%5, %5, %5, %5};" \
                                                  : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3) : AC(AV), "f"(vf) : "memory")
#define A12_ADD_V2F16X2(S, P, AC, AV) asm volatile("atom" S "." P ".add.noftz.v2.f16x2 {%0, %1}, [%2], {%3, %3};" \
                                                    : "=r"(w0), "=r"(w1) : AC(AV), "r"(v32) : "memory")
#define A12_ADD_V4F16X2(S, P, AC, AV) asm volatile("atom" S "." P ".add.noftz.v4.f16x2 {%0, %1, %2, %3}, [%4], {%5, %5, %5, %5};" \
                                                    : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3) : AC(AV), "r"(v32) : "memory")

template <int OP, int SP, int SEM = 0>
__device__ __forceinline__ uint64_t atomx(uint64_t a, uint64_t v, uint64_t c = 0) {
    const uint32_t a32 = (uint32_t)a, v32 = (uint32_t)v, c32 = (uint32_t)c;
    uint32_t d32 = 0;
    uint64_t d64 = 0;
    (void)a32; (void)v32; (void)c32;
    if constexpr (OP == AO_ADD_U32) { X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_U32) return d32; }
    else if constexpr (OP == AO_INC_U32) { X12_SPACE(AT12_SEM, SP, SEM, A12_INC_U32) return d32; }
    else if constexpr (OP == AO_MIN_S32) { X12_SPACE(AT12_SEM, SP, SEM, A12_MIN_S32) return d32; }
    else if constexpr (OP == AO_MAX_U32) { X12_SPACE(AT12_SEM, SP, SEM, A12_MAX_U32) return d32; }
    else if constexpr (OP == AO_AND_B32) { X12_SPACE(AT12_SEM, SP, SEM, A12_AND_B32) return d32; }
    else if constexpr (OP == AO_EXCH_B32) { X12_SPACE(AT12_SEM, SP, SEM, A12_EXCH_B32) return d32; }
    else if constexpr (OP == AO_CAS_B32) { X12_SPACE(AT12_SEM, SP, SEM, A12_CAS_B32) return d32; }
    else if constexpr (OP == AO_ADD_F16X2) { X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_F16X2) return d32; }
    else if constexpr (OP == AO_ADD_BF16X2) { X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_BF16X2) return d32; }
    else if constexpr (OP == AO_ADD_U64) { X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_U64) return d64; }
    else if constexpr (OP == AO_CAS_B64) { X12_SPACE(AT12_SEM, SP, SEM, A12_CAS_B64) return d64; }
    else if constexpr (OP == AO_ADD_F32) {
        const float vf = __uint_as_float(v32);
        float df;
        X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_F32)
        return __float_as_uint(df);
    } else if constexpr (OP == AO_ADD_F64) {
        const double vd = __longlong_as_double((long long)v);
        double dd;
        X12_SPACE(AT12_SEM, SP, SEM, A12_ADD_F64)
        return (uint64_t)__double_as_longlong(dd);
    } else if constexpr (OP == AO_ADD_V2F32 || OP == AO_ADD_V4F32) {
        static_assert(SP == AS_GLOBAL, "vector atomics exist for .global only");
        const float vf = __uint_as_float(v32);
        float f0, f1, f2 = 0.f, f3 = 0.f;
        if constexpr (OP == AO_ADD_V2F32) { AT12_SEM(SEM, A12_ADD_V2F32, "global", "l", a) }
        else { AT12_SEM(SEM, A12_ADD_V4F32, "global", "l", a) }
        return (uint64_t)__float_as_uint(f0) |
               (uint64_t)(__float_as_uint(f1) ^ __float_as_uint(f2) ^ __float_as_uint(f3)) << 32;
    } else {
        static_assert(OP == AO_ADD_V2F16X2 || OP == AO_ADD_V4F16X2, "unknown op");
        static_assert(SP == AS_GLOBAL, "vector atomics exist for .global only");
        uint32_t w0, w1, w2 = 0, w3 = 0;
        if constexpr (OP == AO_ADD_V2F16X2) { AT12_SEM(SEM, A12_ADD_V2F16X2, "global", "l", a) }
        else { AT12_SEM(SEM, A12_ADD_V4F16X2, "global", "l", a) }
        return (uint64_t)w0 | (uint64_t)(w1 ^ w2 ^ w3) << 32;
    }
}

// ---- red bodies
#define R12_ADD_U32(S, P, AC, AV) asm volatile("red" S "." P ".add.u32 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_INC_U32(S, P, AC, AV) asm volatile("red" S "." P ".inc.u32 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_MIN_S32(S, P, AC, AV) asm volatile("red" S "." P ".min.s32 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_MAX_U32(S, P, AC, AV) asm volatile("red" S "." P ".max.u32 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_AND_B32(S, P, AC, AV) asm volatile("red" S "." P ".and.b32 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_ADD_F16X2(S, P, AC, AV) asm volatile("red" S "." P ".add.noftz.f16x2 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_ADD_BF16X2(S, P, AC, AV) asm volatile("red" S "." P ".add.noftz.bf16x2 [%0], %1;" :: AC(AV), "r"(v32) : "memory")
#define R12_ADD_U64(S, P, AC, AV) asm volatile("red" S "." P ".add.u64 [%0], %1;" :: AC(AV), "l"(v) : "memory")
#define R12_ADD_F32(S, P, AC, AV) asm volatile("red" S "." P ".add.f32 [%0], %1;" :: AC(AV), "f"(__uint_as_float(v32)) : "memory")
#define R12_ADD_F64(S, P, AC, AV) asm volatile("red" S "." P ".add.f64 [%0], %1;" :: AC(AV), "d"(__longlong_as_double((long long)v)) : "memory")
#define R12_ADD_V2F32(S, P, AC, AV) asm volatile("red" S "." P ".add.v2.f32 [%0], {%1, %1};" :: AC(AV), "f"(__uint_as_float(v32)) : "memory")
#define R12_ADD_V4F32(S, P, AC, AV) asm volatile("red" S "." P ".add.v4.f32 [%0], {%1, %1, %1, %1};" :: AC(AV), "f"(__uint_as_float(v32)) : "memory")
#define R12_ADD_V2F16X2(S, P, AC, AV) asm volatile("red" S "." P ".add.noftz.v2.f16x2 [%0], {%1, %1};" :: AC(AV), "r"(v32) : "memory")
#define R12_ADD_V4F16X2(S, P, AC, AV) asm volatile("red" S "." P ".add.noftz.v4.f16x2 [%0], {%1, %1, %1, %1};" :: AC(AV), "r"(v32) : "memory")

template <int OP, int SP, int SEM = 0>
__device__ __forceinline__ void redx(uint64_t a, uint64_t v) {
    static_assert(op_has_red(OP), "no red form for exch / cas");
    const uint32_t a32 = (uint32_t)a, v32 = (uint32_t)v;
    (void)a32; (void)v32;
    if constexpr (OP == AO_ADD_U32) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_U32) }
    else if constexpr (OP == AO_INC_U32) { X12_SPACE(RD12_SEM, SP, SEM, R12_INC_U32) }
    else if constexpr (OP == AO_MIN_S32) { X12_SPACE(RD12_SEM, SP, SEM, R12_MIN_S32) }
    else if constexpr (OP == AO_MAX_U32) { X12_SPACE(RD12_SEM, SP, SEM, R12_MAX_U32) }
    else if constexpr (OP == AO_AND_B32) { X12_SPACE(RD12_SEM, SP, SEM, R12_AND_B32) }
    else if constexpr (OP == AO_ADD_F16X2) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_F16X2) }
    else if constexpr (OP == AO_ADD_BF16X2) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_BF16X2) }
    else if constexpr (OP == AO_ADD_U64) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_U64) }
    else if constexpr (OP == AO_ADD_F32) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_F32) }
    else if constexpr (OP == AO_ADD_F64) { X12_SPACE(RD12_SEM, SP, SEM, R12_ADD_F64) }
    else if constexpr (OP == AO_ADD_V2F32) { static_assert(SP == AS_GLOBAL, ".global only"); RD12_SEM(SEM, R12_ADD_V2F32, "global", "l", a) }
    else if constexpr (OP == AO_ADD_V4F32) { static_assert(SP == AS_GLOBAL, ".global only"); RD12_SEM(SEM, R12_ADD_V4F32, "global", "l", a) }
    else if constexpr (OP == AO_ADD_V2F16X2) { static_assert(SP == AS_GLOBAL, ".global only"); RD12_SEM(SEM, R12_ADD_V2F16X2, "global", "l", a) }
    else { static_assert(OP == AO_ADD_V4F16X2, "unknown op"); static_assert(SP == AS_GLOBAL, ".global only"); RD12_SEM(SEM, R12_ADD_V4F16X2, "global", "l", a) }
}

// ---- plain loads / stores of the same paths (references for the chains), 32/64-bit
template <int SP>
__device__ __forceinline__ uint32_t ld32x(uint64_t a) {
    uint32_t v;
    if constexpr (SP == AS_CTA) asm volatile("ld.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"((uint32_t)a) : "memory");
    else if constexpr (SP == AS_CLUSTER) asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"((uint32_t)a) : "memory");
    else asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
    return v;
}
template <int SP>
__device__ __forceinline__ uint64_t ld64x(uint64_t a) {
    uint64_t v;
    if constexpr (SP == AS_CTA) asm volatile("ld.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"((uint32_t)a) : "memory");
    else if constexpr (SP == AS_CLUSTER) asm volatile("ld.shared::cluster.u64 %0, [%1];" : "=l"(v) : "r"((uint32_t)a) : "memory");
    else asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(a) : "memory");
    return v;
}
template <int SP>
__device__ __forceinline__ void st32x(uint64_t a, uint32_t v) {
    if constexpr (SP == AS_CTA) asm volatile("st.shared::cta.u32 [%0], %1;" :: "r"((uint32_t)a), "r"(v) : "memory");
    else if constexpr (SP == AS_CLUSTER) asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"((uint32_t)a), "r"(v) : "memory");
    else asm volatile("st.global.u32 [%0], %1;" :: "l"(a), "r"(v) : "memory");
}
