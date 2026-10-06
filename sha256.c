/*
 * SHA-256 streaming front-end (init/update/final) shared by all backends.
 * Only the block compression differs per backend; padding and length
 * encoding live here once.
 */
#include "sha256.h"
#include <string.h>

static const uint32_t IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

void sha256_init_backend(sha256_ctx *ctx, sha256_compress_fn fn) {
    memcpy(ctx->state, IV, sizeof IV);
    ctx->total    = 0;
    ctx->buflen   = 0;
    ctx->compress = fn;
}

void sha256_init(sha256_ctx *ctx) {
    sha256_init_backend(ctx, sha256_best_compress(NULL));
}

void sha256_update(sha256_ctx *ctx, const void *data_, size_t len) {
    const uint8_t *data = (const uint8_t *)data_;
    ctx->total += len;

    // Top up a partial block first
    if (ctx->buflen) {
        size_t need = SHA256_BLOCK_SIZE - ctx->buflen;
        size_t take = len < need ? len : need;
        memcpy(ctx->buf + ctx->buflen, data, take);
        ctx->buflen += take;
        data += take;
        len  -= take;
        if (ctx->buflen == SHA256_BLOCK_SIZE) {
            ctx->compress(ctx->state, ctx->buf, 1);
            ctx->buflen = 0;
        }
    }

    // Bulk: hand whole blocks straight to the backend
    if (len >= SHA256_BLOCK_SIZE) {
        size_t nblocks = len / SHA256_BLOCK_SIZE;
        ctx->compress(ctx->state, data, nblocks);
        size_t consumed = nblocks * SHA256_BLOCK_SIZE;
        data += consumed;
        len  -= consumed;
    }

    // Stash the remainder
    if (len) {
        memcpy(ctx->buf, data, len);
        ctx->buflen = len;
    }
}

void sha256_final(sha256_ctx *ctx, uint8_t out[SHA256_DIGEST_SIZE]) {
    uint64_t bits = ctx->total * 8;

    // Append the mandatory 0x80 byte
    ctx->buf[ctx->buflen++] = 0x80;

    // If no room for the 8-byte length, flush this block first
    if (ctx->buflen > 56) {
        memset(ctx->buf + ctx->buflen, 0, SHA256_BLOCK_SIZE - ctx->buflen);
        ctx->compress(ctx->state, ctx->buf, 1);
        ctx->buflen = 0;
    }
    memset(ctx->buf + ctx->buflen, 0, 56 - ctx->buflen);

    // 64-bit big-endian bit length in the final 8 bytes
    for (int i = 0; i < 8; i++)
        ctx->buf[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
    ctx->compress(ctx->state, ctx->buf, 1);

    // Emit big-endian state words
    for (int i = 0; i < 8; i++) {
        out[4 * i + 0] = (uint8_t)(ctx->state[i] >> 24);
        out[4 * i + 1] = (uint8_t)(ctx->state[i] >> 16);
        out[4 * i + 2] = (uint8_t)(ctx->state[i] >> 8);
        out[4 * i + 3] = (uint8_t)(ctx->state[i]);
    }
}

void sha256_oneshot(sha256_compress_fn fn, const void *data, size_t len,
                    uint8_t out[SHA256_DIGEST_SIZE]) {
    sha256_ctx c;
    sha256_init_backend(&c, fn);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}

sha256_compress_fn sha256_best_compress(const char **name_out) {
#ifdef HAVE_ARMV8_SHA2
    if (sha256_armv8_available()) {
        if (name_out) *name_out = "armv8-ce";
        return sha256_compress_armv8;
    }
#endif
    if (name_out) *name_out = "scalar";
    return sha256_compress_scalar;
}
