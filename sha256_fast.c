/*
 * Optimized *software* SHA-256 compression (no crypto extension).
 *
 * Two improvements over the naive scalar backend:
 *   1. Rolling 16-word message schedule instead of a 64-word array, so the
 *      working set stays in registers rather than spilling to the stack.
 *   2. Fully unrolled rounds with register renaming: the a..h working
 *      variables are rotated by macro argument order instead of being
 *      physically copied every round (`h=g; g=f; ...`), removing 6 moves
 *      per round * 64 rounds per block.
 *
 * This is the portable technique used by OpenSSL's C path. It matters most
 * on cores that lack the ARMv8 crypto extension (e.g. the Pi 4 Cortex-A72).
 */
#include "sha256.h"

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

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define Ch(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x) (ROTR(x, 2)  ^ ROTR(x, 13) ^ ROTR(x, 22))
#define BSIG1(x) (ROTR(x, 6)  ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SSIG0(x) (ROTR(x, 7)  ^ ROTR(x, 18) ^ ((x) >> 3))
#define SSIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

// One round: writes new 'a' into h's slot and new 'e' into d's slot
#define RND(a, b, c, d, e, f, g, h, Wv, Kv)          \
    do {                                             \
        uint32_t T1 = (h) + BSIG1(e) + Ch(e, f, g) + (Kv) + (Wv); \
        uint32_t T2 = BSIG0(a) + Maj(a, b, c);       \
        (d) += T1;                                   \
        (h)  = T1 + T2;                              \
    } while (0)

// The 8 register-renaming rotations; 8 rounds restore the original order
#define P0(W, Kv) RND(a, b, c, d, e, f, g, h, W, Kv)
#define P1(W, Kv) RND(h, a, b, c, d, e, f, g, W, Kv)
#define P2(W, Kv) RND(g, h, a, b, c, d, e, f, W, Kv)
#define P3(W, Kv) RND(f, g, h, a, b, c, d, e, W, Kv)
#define P4(W, Kv) RND(e, f, g, h, a, b, c, d, W, Kv)
#define P5(W, Kv) RND(d, e, f, g, h, a, b, c, W, Kv)
#define P6(W, Kv) RND(c, d, e, f, g, h, a, b, W, Kv)
#define P7(W, Kv) RND(b, c, d, e, f, g, h, a, W, Kv)

// Rolling schedule update for round t>=16; evaluates to the new W[t]
#define WUP(t) (W[(t) & 15] += SSIG1(W[((t) + 14) & 15]) + W[((t) + 9) & 15] + SSIG0(W[((t) + 1) & 15]))

void sha256_compress_fast(uint32_t state[8], const uint8_t *data, size_t nblocks) {
    while (nblocks--) {
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        uint32_t W[16];

        for (int i = 0; i < 16; i++)
            W[i] = ((uint32_t)data[4 * i + 0] << 24) |
                   ((uint32_t)data[4 * i + 1] << 16) |
                   ((uint32_t)data[4 * i + 2] <<  8) |
                   ((uint32_t)data[4 * i + 3]);

        // Rounds 0-15: schedule words preloaded
        P0(W[0],  K[0]);  P1(W[1],  K[1]);  P2(W[2],  K[2]);  P3(W[3],  K[3]);
        P4(W[4],  K[4]);  P5(W[5],  K[5]);  P6(W[6],  K[6]);  P7(W[7],  K[7]);
        P0(W[8],  K[8]);  P1(W[9],  K[9]);  P2(W[10], K[10]); P3(W[11], K[11]);
        P4(W[12], K[12]); P5(W[13], K[13]); P6(W[14], K[14]); P7(W[15], K[15]);

        // Rounds 16-63: schedule computed on the fly in the 16-word window
        P0(WUP(16), K[16]); P1(WUP(17), K[17]); P2(WUP(18), K[18]); P3(WUP(19), K[19]);
        P4(WUP(20), K[20]); P5(WUP(21), K[21]); P6(WUP(22), K[22]); P7(WUP(23), K[23]);
        P0(WUP(24), K[24]); P1(WUP(25), K[25]); P2(WUP(26), K[26]); P3(WUP(27), K[27]);
        P4(WUP(28), K[28]); P5(WUP(29), K[29]); P6(WUP(30), K[30]); P7(WUP(31), K[31]);
        P0(WUP(32), K[32]); P1(WUP(33), K[33]); P2(WUP(34), K[34]); P3(WUP(35), K[35]);
        P4(WUP(36), K[36]); P5(WUP(37), K[37]); P6(WUP(38), K[38]); P7(WUP(39), K[39]);
        P0(WUP(40), K[40]); P1(WUP(41), K[41]); P2(WUP(42), K[42]); P3(WUP(43), K[43]);
        P4(WUP(44), K[44]); P5(WUP(45), K[45]); P6(WUP(46), K[46]); P7(WUP(47), K[47]);
        P0(WUP(48), K[48]); P1(WUP(49), K[49]); P2(WUP(50), K[50]); P3(WUP(51), K[51]);
        P4(WUP(52), K[52]); P5(WUP(53), K[53]); P6(WUP(54), K[54]); P7(WUP(55), K[55]);
        P0(WUP(56), K[56]); P1(WUP(57), K[57]); P2(WUP(58), K[58]); P3(WUP(59), K[59]);
        P4(WUP(60), K[60]); P5(WUP(61), K[61]); P6(WUP(62), K[62]); P7(WUP(63), K[63]);

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;

        data += SHA256_BLOCK_SIZE;
    }
}
