/*
 * Multi-buffer SHA-256: hash several INDEPENDENT messages in parallel.
 *
 * A single SHA-256 stream is a serial dependency chain, so one hash can't be
 * sped up much past the single-block cost. Throughput workloads (hashing many
 * files/chunks) can do far better by running independent messages side by side:
 *
 *   sha256_x4_neon : 4-way pure-software SIMD. Each NEON lane carries one
 *                    message's state word. No crypto extension required, so
 *                    this is the fast path on cores like the Pi 4 (A72).
 *   sha256_x2_ce   : 2-way interleave of the ARMv8 crypto instructions. The
 *                    two streams hide each other's SHA256H latency, beating
 *                    single-stream armv8-ce on wide cores (A76, Apple).
 *
 * Both one-shot helpers assume the N input messages have EQUAL length (the
 * common chunked-hashing case) so the padding layout is shared across lanes.
 */
#include "sha256.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#include <string.h>

static const uint32_t IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

static const uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

/* Build the padded tail block(s) for an equal-length message set. Returns the
 * number of tail blocks (1 or 2). `tail` must hold 128 bytes. */
static int build_tail(uint8_t *tail, const uint8_t *last, size_t rem, uint64_t total) {
    int blocks = (rem >= 56) ? 2 : 1;
    memset(tail, 0, 128);
    memcpy(tail, last, rem);
    tail[rem] = 0x80;
    uint64_t bits = total * 8;
    size_t lenpos = (size_t)blocks * 64 - 8;
    for (int i = 0; i < 8; i++)
        tail[lenpos + i] = (uint8_t)(bits >> (56 - 8 * i));
    return blocks;
}

// 4-way NEON software

#define ROTR4(x, n) vorrq_u32(vshrq_n_u32((x), (n)), vshlq_n_u32((x), 32 - (n)))
#define SHR4(x, n)  vshrq_n_u32((x), (n))
#define BSIG0_4(x)  veorq_u32(veorq_u32(ROTR4(x, 2),  ROTR4(x, 13)), ROTR4(x, 22))
#define BSIG1_4(x)  veorq_u32(veorq_u32(ROTR4(x, 6),  ROTR4(x, 11)), ROTR4(x, 25))
#define SSIG0_4(x)  veorq_u32(veorq_u32(ROTR4(x, 7),  ROTR4(x, 18)), SHR4(x, 3))
#define SSIG1_4(x)  veorq_u32(veorq_u32(ROTR4(x, 17), ROTR4(x, 19)), SHR4(x, 10))

static inline uint32x4_t Ch4(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vbslq_u32(x, y, z);                       // (x&y)^(~x&z)
}
static inline uint32x4_t Maj4(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(vandq_u32(y, z), vandq_u32(x, veorq_u32(y, z)));
}

static void block4(uint32x4_t st[8],
                   const uint8_t *p0, const uint8_t *p1,
                   const uint8_t *p2, const uint8_t *p3) {
    uint32x4_t w[64];
    for (int i = 0; i < 16; i++) {
        uint32_t t[4] = { be32(p0 + 4 * i), be32(p1 + 4 * i),
                          be32(p2 + 4 * i), be32(p3 + 4 * i) };
        w[i] = vld1q_u32(t);
    }
    for (int i = 16; i < 64; i++)
        w[i] = vaddq_u32(vaddq_u32(SSIG1_4(w[i - 2]), w[i - 7]),
                         vaddq_u32(SSIG0_4(w[i - 15]), w[i - 16]));

    uint32x4_t a = st[0], b = st[1], c = st[2], d = st[3];
    uint32x4_t e = st[4], f = st[5], g = st[6], h = st[7];

    for (int i = 0; i < 64; i++) {
        uint32x4_t kw = vaddq_u32(vdupq_n_u32(K[i]), w[i]);
        uint32x4_t t1 = vaddq_u32(vaddq_u32(h, BSIG1_4(e)),
                                  vaddq_u32(Ch4(e, f, g), kw));
        uint32x4_t t2 = vaddq_u32(BSIG0_4(a), Maj4(a, b, c));
        h = g; g = f; f = e; e = vaddq_u32(d, t1);
        d = c; c = b; b = a; a = vaddq_u32(t1, t2);
    }

    st[0] = vaddq_u32(st[0], a); st[1] = vaddq_u32(st[1], b);
    st[2] = vaddq_u32(st[2], c); st[3] = vaddq_u32(st[3], d);
    st[4] = vaddq_u32(st[4], e); st[5] = vaddq_u32(st[5], f);
    st[6] = vaddq_u32(st[6], g); st[7] = vaddq_u32(st[7], h);
}

void sha256_x4_neon(const uint8_t *const msg[4], size_t len, uint8_t out[][32]) {
    uint32x4_t st[8];
    for (int i = 0; i < 8; i++) st[i] = vdupq_n_u32(IV[i]);

    size_t full = len / 64;
    for (size_t b = 0; b < full; b++)
        block4(st, msg[0] + b * 64, msg[1] + b * 64,
                   msg[2] + b * 64, msg[3] + b * 64);

    size_t rem = len - full * 64;
    uint8_t tail[4][128];
    int tblocks = 0;
    for (int l = 0; l < 4; l++)
        tblocks = build_tail(tail[l], msg[l] + full * 64, rem, len);
    for (int tb = 0; tb < tblocks; tb++)
        block4(st, tail[0] + tb * 64, tail[1] + tb * 64,
                   tail[2] + tb * 64, tail[3] + tb * 64);

    for (int i = 0; i < 8; i++) {
        uint32_t wd[4];
        vst1q_u32(wd, st[i]);
        for (int l = 0; l < 4; l++) {
            out[l][4 * i + 0] = (uint8_t)(wd[l] >> 24);
            out[l][4 * i + 1] = (uint8_t)(wd[l] >> 16);
            out[l][4 * i + 2] = (uint8_t)(wd[l] >> 8);
            out[l][4 * i + 3] = (uint8_t)(wd[l]);
        }
    }
}

/*
 * One 4-lane chain leaves the NEON units mostly idle: the 64 rounds are a
 * serial dependency chain and each op has multi-cycle latency. Running two
 * independent 4-lane groups side by side fills those idle issue slots, so
 * throughput scales well past x4 on wide cores.
 */

static inline uint32x4_t gather_w(const uint8_t *const p[4], int i) {
    uint32_t t[4] = { be32(p[0] + 4 * i), be32(p[1] + 4 * i),
                      be32(p[2] + 4 * i), be32(p[3] + 4 * i) };
    return vld1q_u32(t);
}

#define RND8_SCHED(dst, s2, s7, s15, s16)                                  \
    dst = vaddq_u32(vaddq_u32(SSIG1_4(s2), s7), vaddq_u32(SSIG0_4(s15), s16))

static void block8(uint32x4_t A[8], uint32x4_t B[8],
                   const uint8_t *const qa[4], const uint8_t *const qb[4]) {
    uint32x4_t wa[64], wb[64];
    for (int i = 0; i < 16; i++) { wa[i] = gather_w(qa, i); wb[i] = gather_w(qb, i); }
    for (int i = 16; i < 64; i++) {
        RND8_SCHED(wa[i], wa[i - 2], wa[i - 7], wa[i - 15], wa[i - 16]);
        RND8_SCHED(wb[i], wb[i - 2], wb[i - 7], wb[i - 15], wb[i - 16]);
    }

    uint32x4_t a0 = A[0], b0 = A[1], c0 = A[2], d0 = A[3], e0 = A[4], f0 = A[5], g0 = A[6], h0 = A[7];
    uint32x4_t a1 = B[0], b1 = B[1], c1 = B[2], d1 = B[3], e1 = B[4], f1 = B[5], g1 = B[6], h1 = B[7];

    for (int i = 0; i < 64; i++) {
        uint32x4_t kv  = vdupq_n_u32(K[i]);
        // Two independent groups interleaved for instruction-level parallelism.
        uint32x4_t T1a = vaddq_u32(vaddq_u32(h0, BSIG1_4(e0)), vaddq_u32(Ch4(e0, f0, g0), vaddq_u32(kv, wa[i])));
        uint32x4_t T1b = vaddq_u32(vaddq_u32(h1, BSIG1_4(e1)), vaddq_u32(Ch4(e1, f1, g1), vaddq_u32(kv, wb[i])));
        uint32x4_t T2a = vaddq_u32(BSIG0_4(a0), Maj4(a0, b0, c0));
        uint32x4_t T2b = vaddq_u32(BSIG0_4(a1), Maj4(a1, b1, c1));
        h0 = g0; g0 = f0; f0 = e0; e0 = vaddq_u32(d0, T1a); d0 = c0; c0 = b0; b0 = a0; a0 = vaddq_u32(T1a, T2a);
        h1 = g1; g1 = f1; f1 = e1; e1 = vaddq_u32(d1, T1b); d1 = c1; c1 = b1; b1 = a1; a1 = vaddq_u32(T1b, T2b);
    }

    A[0] = vaddq_u32(A[0], a0); A[1] = vaddq_u32(A[1], b0); A[2] = vaddq_u32(A[2], c0); A[3] = vaddq_u32(A[3], d0);
    A[4] = vaddq_u32(A[4], e0); A[5] = vaddq_u32(A[5], f0); A[6] = vaddq_u32(A[6], g0); A[7] = vaddq_u32(A[7], h0);
    B[0] = vaddq_u32(B[0], a1); B[1] = vaddq_u32(B[1], b1); B[2] = vaddq_u32(B[2], c1); B[3] = vaddq_u32(B[3], d1);
    B[4] = vaddq_u32(B[4], e1); B[5] = vaddq_u32(B[5], f1); B[6] = vaddq_u32(B[6], g1); B[7] = vaddq_u32(B[7], h1);
}

static void store_digests(const uint32x4_t st[8], uint8_t out[][32], int base) {
    for (int i = 0; i < 8; i++) {
        uint32_t wd[4];
        vst1q_u32(wd, st[i]);
        for (int l = 0; l < 4; l++) {
            out[base + l][4 * i + 0] = (uint8_t)(wd[l] >> 24);
            out[base + l][4 * i + 1] = (uint8_t)(wd[l] >> 16);
            out[base + l][4 * i + 2] = (uint8_t)(wd[l] >> 8);
            out[base + l][4 * i + 3] = (uint8_t)(wd[l]);
        }
    }
}

void sha256_x8_neon(const uint8_t *const msg[8], size_t len, uint8_t out[][32]) {
    uint32x4_t A[8], B[8];
    for (int i = 0; i < 8; i++) { A[i] = vdupq_n_u32(IV[i]); B[i] = vdupq_n_u32(IV[i]); }

    size_t full = len / 64;
    for (size_t b = 0; b < full; b++) {
        const uint8_t *qa[4] = { msg[0] + b * 64, msg[1] + b * 64, msg[2] + b * 64, msg[3] + b * 64 };
        const uint8_t *qb[4] = { msg[4] + b * 64, msg[5] + b * 64, msg[6] + b * 64, msg[7] + b * 64 };
        block8(A, B, qa, qb);
    }

    size_t rem = len - full * 64;
    uint8_t tail[8][128];
    int tblocks = 0;
    for (int l = 0; l < 8; l++)
        tblocks = build_tail(tail[l], msg[l] + full * 64, rem, len);
    for (int tb = 0; tb < tblocks; tb++) {
        const uint8_t *qa[4] = { tail[0] + tb * 64, tail[1] + tb * 64, tail[2] + tb * 64, tail[3] + tb * 64 };
        const uint8_t *qb[4] = { tail[4] + tb * 64, tail[5] + tb * 64, tail[6] + tb * 64, tail[7] + tb * 64 };
        block8(A, B, qa, qb);
    }

    store_digests(A, out, 0);
    store_digests(B, out, 4);
}

// 2-way interleaved ARMv8 Crypto Extension
#ifdef HAVE_ARMV8_SHA2

/* Each statement of the single-stream round sequence is wrapped in a 2-trip
 * loop so the compiler emits lane 0 then lane 1 back to back; the two streams
 * fill each other's SHA256H latency. */
#define FL(stmt) do { for (int l = 0; l < 2; l++) { stmt; } } while (0)

static void block_ce2(uint32x4_t S0[2], uint32x4_t S1[2],
                      const uint8_t *p0, const uint8_t *p1) {
    const uint8_t *P[2] = { p0, p1 };
    uint32x4_t MSG0[2], MSG1[2], MSG2[2], MSG3[2];
    uint32x4_t T0[2], T1[2], T2[2], ABEF[2], CDGH[2];

    FL(ABEF[l] = S0[l]);
    FL(CDGH[l] = S1[l]);

    FL(MSG0[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(P[l] +  0))));
    FL(MSG1[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(P[l] + 16))));
    FL(MSG2[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(P[l] + 32))));
    FL(MSG3[l] = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(P[l] + 48))));

    FL(T0[l] = vaddq_u32(MSG0[l], vld1q_u32(&K[0x00])));

    // Rounds 0-3
    FL(MSG0[l] = vsha256su0q_u32(MSG0[l], MSG1[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG1[l], vld1q_u32(&K[0x04])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG0[l] = vsha256su1q_u32(MSG0[l], MSG2[l], MSG3[l]));

    // Rounds 4-7
    FL(MSG1[l] = vsha256su0q_u32(MSG1[l], MSG2[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG2[l], vld1q_u32(&K[0x08])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG1[l] = vsha256su1q_u32(MSG1[l], MSG3[l], MSG0[l]));

    // Rounds 8-11
    FL(MSG2[l] = vsha256su0q_u32(MSG2[l], MSG3[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG3[l], vld1q_u32(&K[0x0c])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG2[l] = vsha256su1q_u32(MSG2[l], MSG0[l], MSG1[l]));

    // Rounds 12-15
    FL(MSG3[l] = vsha256su0q_u32(MSG3[l], MSG0[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG0[l], vld1q_u32(&K[0x10])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG3[l] = vsha256su1q_u32(MSG3[l], MSG1[l], MSG2[l]));

    // Rounds 16-19
    FL(MSG0[l] = vsha256su0q_u32(MSG0[l], MSG1[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG1[l], vld1q_u32(&K[0x14])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG0[l] = vsha256su1q_u32(MSG0[l], MSG2[l], MSG3[l]));

    // Rounds 20-23
    FL(MSG1[l] = vsha256su0q_u32(MSG1[l], MSG2[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG2[l], vld1q_u32(&K[0x18])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG1[l] = vsha256su1q_u32(MSG1[l], MSG3[l], MSG0[l]));

    // Rounds 24-27
    FL(MSG2[l] = vsha256su0q_u32(MSG2[l], MSG3[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG3[l], vld1q_u32(&K[0x1c])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG2[l] = vsha256su1q_u32(MSG2[l], MSG0[l], MSG1[l]));

    // Rounds 28-31
    FL(MSG3[l] = vsha256su0q_u32(MSG3[l], MSG0[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG0[l], vld1q_u32(&K[0x20])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG3[l] = vsha256su1q_u32(MSG3[l], MSG1[l], MSG2[l]));

    // Rounds 32-35
    FL(MSG0[l] = vsha256su0q_u32(MSG0[l], MSG1[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG1[l], vld1q_u32(&K[0x24])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG0[l] = vsha256su1q_u32(MSG0[l], MSG2[l], MSG3[l]));

    // Rounds 36-39
    FL(MSG1[l] = vsha256su0q_u32(MSG1[l], MSG2[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG2[l], vld1q_u32(&K[0x28])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG1[l] = vsha256su1q_u32(MSG1[l], MSG3[l], MSG0[l]));

    // Rounds 40-43
    FL(MSG2[l] = vsha256su0q_u32(MSG2[l], MSG3[l]));
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG3[l], vld1q_u32(&K[0x2c])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));
    FL(MSG2[l] = vsha256su1q_u32(MSG2[l], MSG0[l], MSG1[l]));

    // Rounds 44-47
    FL(MSG3[l] = vsha256su0q_u32(MSG3[l], MSG0[l]));
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG0[l], vld1q_u32(&K[0x30])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));
    FL(MSG3[l] = vsha256su1q_u32(MSG3[l], MSG1[l], MSG2[l]));

    // Rounds 48-51
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG1[l], vld1q_u32(&K[0x34])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));

    // Rounds 52-55
    FL(T2[l] = S0[l]);
    FL(T0[l] = vaddq_u32(MSG2[l], vld1q_u32(&K[0x38])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));

    // Rounds 56-59
    FL(T2[l] = S0[l]);
    FL(T1[l] = vaddq_u32(MSG3[l], vld1q_u32(&K[0x3c])));
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T0[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T0[l]));

    // Rounds 60-63
    FL(T2[l] = S0[l]);
    FL(S0[l] = vsha256hq_u32(S0[l], S1[l], T1[l]));
    FL(S1[l] = vsha256h2q_u32(S1[l], T2[l], T1[l]));

    FL(S0[l] = vaddq_u32(S0[l], ABEF[l]));
    FL(S1[l] = vaddq_u32(S1[l], CDGH[l]));
}

void sha256_x2_ce(const uint8_t *const msg[2], size_t len, uint8_t out[][32]) {
    uint32x4_t S0[2], S1[2];
    for (int l = 0; l < 2; l++) {
        S0[l] = vld1q_u32(&IV[0]);
        S1[l] = vld1q_u32(&IV[4]);
    }

    size_t full = len / 64;
    for (size_t b = 0; b < full; b++)
        block_ce2(S0, S1, msg[0] + b * 64, msg[1] + b * 64);

    size_t rem = len - full * 64;
    uint8_t tail[2][128];
    int tblocks = 0;
    for (int l = 0; l < 2; l++)
        tblocks = build_tail(tail[l], msg[l] + full * 64, rem, len);
    for (int tb = 0; tb < tblocks; tb++)
        block_ce2(S0, S1, tail[0] + tb * 64, tail[1] + tb * 64);

    for (int l = 0; l < 2; l++) {
        uint32_t wd[8];
        vst1q_u32(&wd[0], S0[l]);
        vst1q_u32(&wd[4], S1[l]);
        for (int i = 0; i < 8; i++) {
            out[l][4 * i + 0] = (uint8_t)(wd[i] >> 24);
            out[l][4 * i + 1] = (uint8_t)(wd[i] >> 16);
            out[l][4 * i + 2] = (uint8_t)(wd[i] >> 8);
            out[l][4 * i + 3] = (uint8_t)(wd[i]);
        }
    }
}

#endif // HAVE_ARMV8_SHA2
#endif // __aarch64__
