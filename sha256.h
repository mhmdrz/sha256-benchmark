#ifndef SHA256_H
#define SHA256_H

#include <stdint.h>
#include <stddef.h>

#define SHA256_DIGEST_SIZE 32
#define SHA256_BLOCK_SIZE  64

/*
 * A compression function processes `nblocks` consecutive 64-byte blocks,
 * folding them into the 8-word state. All backends share this signature so
 * the high-level streaming API is backend-agnostic.
 */
typedef void (*sha256_compress_fn)(uint32_t state[8],
                                   const uint8_t *data,
                                   size_t nblocks);

typedef struct {
    uint32_t           state[8];
    uint64_t           total;                  // total message bytes seen
    uint8_t            buf[SHA256_BLOCK_SIZE];  // partial-block buffer
    size_t             buflen;
    sha256_compress_fn compress;
} sha256_ctx;

// Compression backends

// Portable C reference. Always available on every target
void sha256_compress_scalar(uint32_t state[8], const uint8_t *data, size_t nblocks);

// Optimized portable software path (unrolled + rolling schedule)
void sha256_compress_fast(uint32_t state[8], const uint8_t *data, size_t nblocks);

#ifdef HAVE_ARMV8_SHA2
// ARMv8-A Crypto Extension (vsha256h/vsha256su) (aarch64 only)
void sha256_compress_armv8(uint32_t state[8], const uint8_t *data, size_t nblocks);
// Runtime CPU probe: 1 if this core actually has the SHA2 extension
int  sha256_armv8_available(void);
#endif

// Multi-buffer one-shot hashing
// Hash N independent, EQUAL-length messages in parallel; out[i] receives
// the digest of msg[i] (aarch64 only)
#if defined(__aarch64__)
void sha256_x4_neon(const uint8_t *const msg[4], size_t len, uint8_t out[][32]);
void sha256_x8_neon(const uint8_t *const msg[8], size_t len, uint8_t out[][32]);
#endif
#ifdef HAVE_ARMV8_SHA2
void sha256_x2_ce(const uint8_t *const msg[2], size_t len, uint8_t out[][32]);
#endif

// High-level streaming API

void sha256_init_backend(sha256_ctx *ctx, sha256_compress_fn fn);
void sha256_init(sha256_ctx *ctx);  // uses the best backend available
void sha256_update(sha256_ctx *ctx, const void *data, size_t len);
void sha256_final(sha256_ctx *ctx, uint8_t out[SHA256_DIGEST_SIZE]);

// One-shot hash with an explicit backend
void sha256_oneshot(sha256_compress_fn fn, const void *data, size_t len,
                    uint8_t out[SHA256_DIGEST_SIZE]);

/*
 * Return the fastest compression backend available on this CPU at runtime,
 * and (optionally) its short name. This is the "default algo" selector.
 */
sha256_compress_fn sha256_best_compress(const char **name_out);

#endif // SHA256_H
