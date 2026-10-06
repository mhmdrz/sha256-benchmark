/*
 * SHA-256 arm64 backend benchmark.
 *
 * Verifies each available backend against NIST test vectors, then measures
 * throughput across a range of input sizes. Reports MiB/s, ns/byte, and
 * (if --mhz is given) cycles/byte, which is the comparable crypto metric.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "sha256.h"

// Defeats dead-store elimination of the digest.
static volatile uint32_t g_sink;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    const char        *name;
    sha256_compress_fn fn;
} backend_t;

static void to_hex(const uint8_t *d, char *out) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2 * i]     = h[d[i] >> 4];
        out[2 * i + 1] = h[d[i] & 0xf];
    }
    out[64] = '\0';
}

static int check(backend_t b, const uint8_t *msg, size_t len, const char *expect) {
    uint8_t d[32];
    char got[65];
    sha256_oneshot(b.fn, msg, len, d);
    to_hex(d, got);
    if (strcmp(got, expect) != 0) {
        fprintf(stderr, "  [FAIL] %-9s got %s\n              want %s\n", b.name, got, expect);
        return 0;
    }
    return 1;
}

// Correctness gate: empty, "abc", and the one-million-'a' NIST vector
static int selftest(backend_t b) {
    int ok = 1;
    ok &= check(b, (const uint8_t *)"", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ok &= check(b, (const uint8_t *)"abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    size_t n = 1000000;
    uint8_t *big = malloc(n);
    memset(big, 'a', n);
    ok &= check(b, big, n, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    free(big);
    printf("  %-9s %s\n", b.name, ok ? "ok" : "FAILED");
    return ok;
}

// Best bytes/sec over `trials` runs, each at least `budget` seconds
static double bench_size(backend_t b, const uint8_t *buf, size_t size, double budget, int trials) {
    uint8_t d[32];
    // Calibrate iteration count so a trial lasts ~budget seconds
    size_t iters = 1;
    for (;;) {
        double t0 = now_sec();
        for (size_t i = 0; i < iters; i++)
            sha256_oneshot(b.fn, buf, size, d);
        double dt = now_sec() - t0;
        g_sink += d[0];
        if (dt >= budget || iters >= ((size_t)1 << 40)) break;
        if (dt < budget / 8.0) iters *= 8; else iters *= 2;
    }

    double best = 0.0;
    for (int t = 0; t < trials; t++) {
        double t0 = now_sec();
        for (size_t i = 0; i < iters; i++)
            sha256_oneshot(b.fn, buf, size, d);
        double dt = now_sec() - t0;
        g_sink += d[0];
        double bps = (double)iters * (double)size / dt;
        if (bps > best) best = bps;
    }
    return best;
}

#if defined(__aarch64__)
// Confirm a multi-buffer hash matches the single-stream result per lane
static int mb_check(const char *name, const uint8_t *buf, size_t len, int lanes, void (*fn)(const uint8_t *const *, size_t, uint8_t (*)[32])) {
    const uint8_t *msg[8];
    for (int l = 0; l < lanes; l++) msg[l] = buf;
    uint8_t mb_out[8][32], ref[32];
    sha256_oneshot(sha256_compress_scalar, buf, len, ref);
    fn(msg, len, mb_out);
    for (int l = 0; l < lanes; l++) {
        if (memcmp(mb_out[l], ref, 32) != 0) {
            fprintf(stderr, "  [FAIL] %s lane %d\n", name, l);
            return 0;
        }
    }
    printf("  %-9s ok\n", name);
    return 1;
}

static void run_x2(const uint8_t *const *m, size_t len, uint8_t (*o)[32]) {
#ifdef HAVE_ARMV8_SHA2
    sha256_x2_ce(m, len, o);
#else
    (void)m; (void)len; (void)o;
#endif
}
static void run_x4(const uint8_t *const *m, size_t len, uint8_t (*o)[32]) {
    sha256_x4_neon(m, len, o);
}
static void run_x8(const uint8_t *const *m, size_t len, uint8_t (*o)[32]) {
    sha256_x8_neon(m, len, o);
}

// Best aggregate bytes/sec over `trials`, hashing `lanes` buffers per call
static double bench_mb(void (*fn)(const uint8_t *const *, size_t, uint8_t (*)[32]), const uint8_t *buf, size_t size, int lanes, double budget, int trials) {
    const uint8_t *msg[8];
    for (int l = 0; l < lanes; l++) msg[l] = buf;
    uint8_t out[8][32];
    size_t iters = 1;
    for (;;) {
        double t0 = now_sec();
        for (size_t i = 0; i < iters; i++) fn(msg, size, out);
        double dt = now_sec() - t0;
        g_sink += out[0][0];
        if (dt >= budget || iters >= ((size_t)1 << 40)) break;
        iters *= (dt < budget / 8.0) ? 8 : 2;
    }
    double best = 0.0;
    for (int t = 0; t < trials; t++) {
        double t0 = now_sec();
        for (size_t i = 0; i < iters; i++) fn(msg, size, out);
        double dt = now_sec() - t0;
        g_sink += out[0][0];
        double bps = (double)iters * (double)lanes * (double)size / dt;
        if (bps > best) best = bps;
    }
    return best;
}

// Single-stream aggregate for comparison.
static double bench_single(sha256_compress_fn fn, const uint8_t *buf, size_t size,
                           double budget, int trials) {
    backend_t b = { "", fn };
    return bench_size(b, buf, size, budget, trials);
}

static int run_parallel(double mhz, double budget, int trials,
                        const size_t *sizes, int nsizes) {
    printf("multi-buffer self-test (vs scalar reference):\n");
    uint8_t *tv = malloc(1000000);
    memset(tv, 'a', 1000000);
    int ok = 1;
#ifdef HAVE_ARMV8_SHA2
    ok &= mb_check("x2-ce",   (const uint8_t *)"abc", 3, 2, run_x2);
    ok &= mb_check("x2-ce",   tv, 1000000,             2, run_x2);
#endif
    ok &= mb_check("x4-neon", (const uint8_t *)"abc", 3, 4, run_x4);
    ok &= mb_check("x4-neon", tv, 1000000,             4, run_x4);
    ok &= mb_check("x8-neon", (const uint8_t *)"abc", 3, 8, run_x8);
    ok &= mb_check("x8-neon", tv, 1000000,             8, run_x8);
    free(tv);
    if (!ok) { fprintf(stderr, "\nmulti-buffer self-test failed, aborting\n"); return 1; }

    size_t maxsz = 0;
    for (int i = 0; i < nsizes; i++) if (sizes[i] > maxsz) maxsz = sizes[i];
    uint8_t *buf = malloc(maxsz ? maxsz : 1);
    for (size_t i = 0; i < maxsz; i++) buf[i] = (uint8_t)(i * 1103515245u + 12345u);

    printf("\nparallel throughput (aggregate across lanes, best of %d):\n\n", trials);
    printf("  %-10s %6s %10s %10s %10s", "method", "lanes", "size", "MiB/s", "ns/byte");
    if (mhz > 0) printf(" %10s", "cyc/byte");
    printf("\n  ---------- ------ ---------- ---------- ----------");
    if (mhz > 0) printf(" ----------");
    printf("\n");

    struct { const char *name; int lanes; int kind; sha256_compress_fn sfn;
             void (*mfn)(const uint8_t *const *, size_t, uint8_t (*)[32]); } rows[] = {
        { "fast",    1, 0, sha256_compress_fast, NULL },
#ifdef HAVE_ARMV8_SHA2
        { "armv8-ce",1, 0, sha256_compress_armv8, NULL },
        { "x2-ce",   2, 1, NULL, run_x2 },
#endif
        { "x4-neon", 4, 1, NULL, run_x4 },
        { "x8-neon", 8, 1, NULL, run_x8 },
    };
    int nrows = (int)(sizeof rows / sizeof rows[0]);

    for (int i = 0; i < nsizes; i++) {
        for (int r = 0; r < nrows; r++) {
            double bps = rows[r].kind == 0
                ? bench_single(rows[r].sfn, buf, sizes[i], budget, trials)
                : bench_mb(rows[r].mfn, buf, sizes[i], rows[r].lanes, budget, trials);
            printf("  %-10s %6d %10zu %10.1f %10.3f",
                   rows[r].name, rows[r].lanes, sizes[i],
                   bps / (1024.0 * 1024.0), 1e9 / bps);
            if (mhz > 0) printf(" %10.3f", (mhz * 1e6) / bps);
            printf("\n");
        }
        if (i + 1 < nsizes) printf("\n");
    }
    free(buf);
    return 0;
}
#endif // __aarch64__

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  --mhz N        CPU clock in MHz, enables cycles/byte column\n"
        "  --sizes a,b,c  input sizes in bytes (default 64,256,1024,8192,65536,1048576)\n"
        "  --seconds S    measurement budget per size (default 0.5)\n"
        "  --trials N     trials per size, best is reported (default 5)\n"
        "  --backend B    scalar | fast | armv8 | all (default all available)\n"
        "  --parallel     multi-buffer mode: compare x2-ce/x4-neon aggregate throughput\n",
        prog);
}

int main(int argc, char **argv) {
    double mhz        = 0.0;
    double budget     = 0.5;
    int    trials     = 5;
    const char *pick  = "all";
    int    parallel   = 0;
    size_t default_sizes[] = {64, 256, 1024, 8192, 65536, 1048576};
    size_t *sizes = default_sizes;
    int     nsizes = (int)(sizeof default_sizes / sizeof default_sizes[0]);
    size_t  parsed[64];

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mhz") && i + 1 < argc) {
            mhz = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            budget = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--backend") && i + 1 < argc) {
            pick = argv[++i];
        } else if (!strcmp(argv[i], "--parallel")) {
            parallel = 1;
        } else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) {
            nsizes = 0;
            char *s = argv[++i], *tok = strtok(s, ",");
            while (tok && nsizes < 64) { parsed[nsizes++] = strtoull(tok, NULL, 10); tok = strtok(NULL, ","); }
            sizes = parsed;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (parallel) {
#if defined(__aarch64__)
        return run_parallel(mhz, budget, trials, sizes, nsizes);
#else
        fprintf(stderr, "--parallel requires aarch64\n");
        return 1;
#endif
    }

    // Assemble the backend list
    backend_t all[3];
    int nb = 0;
    int want_scalar = !strcmp(pick, "all") || !strcmp(pick, "scalar");
    int want_fast   = !strcmp(pick, "all") || !strcmp(pick, "fast");
    int want_armv8  = !strcmp(pick, "all") || !strcmp(pick, "armv8");

    if (want_scalar)
        all[nb++] = (backend_t){"scalar", sha256_compress_scalar};
    if (want_fast)
        all[nb++] = (backend_t){"fast", sha256_compress_fast};
#ifdef HAVE_ARMV8_SHA2
    if (want_armv8 && sha256_armv8_available())
        all[nb++] = (backend_t){"armv8-ce", sha256_compress_armv8};
    else if (want_armv8 && strcmp(pick, "all"))
        fprintf(stderr, "note: armv8 crypto extension not available on this CPU\n");
#else
    if (want_armv8 && strcmp(pick, "all"))
        fprintf(stderr, "note: binary was not built with the armv8 backend\n");
#endif
    if (nb == 0) { fprintf(stderr, "no backends selected\n"); return 1; }

    const char *best_name = NULL;
    sha256_best_compress(&best_name);
    printf("default (auto-selected) backend: %s\n\n", best_name);

    printf("self-test (NIST vectors):\n");
    int ok = 1;
    for (int i = 0; i < nb; i++) ok &= selftest(all[i]);
    if (!ok) { fprintf(stderr, "\nself-test failed, aborting\n"); return 1; }

    // Allocate the largest buffer once.
    size_t maxsz = 0;
    for (int i = 0; i < nsizes; i++) if (sizes[i] > maxsz) maxsz = sizes[i];
    uint8_t *buf = malloc(maxsz ? maxsz : 1);
    for (size_t i = 0; i < maxsz; i++) buf[i] = (uint8_t)(i * 1103515245u + 12345u);

    printf("\nthroughput (best of %d, ~%.2gs each):\n\n", trials, budget);
    printf("  %-10s %10s %10s %10s", "backend", "size", "MiB/s", "ns/byte");
    if (mhz > 0) printf(" %10s", "cyc/byte");
    printf("\n");
    printf("  ---------- ---------- ---------- ----------");
    if (mhz > 0) printf(" ----------");
    printf("\n");

    for (int i = 0; i < nsizes; i++) {
        for (int j = 0; j < nb; j++) {
            double bps = bench_size(all[j], buf, sizes[i], budget, trials);
            double mibs = bps / (1024.0 * 1024.0);
            double nspb = 1e9 / bps;
            printf("  %-10s %10zu %10.1f %10.3f", all[j].name, sizes[i], mibs, nspb);
            if (mhz > 0) printf(" %10.3f", (mhz * 1e6) / bps);
            printf("\n");
        }
        if (nb > 1 && i + 1 < nsizes) printf("\n");
    }

    free(buf);
    return 0;
}
