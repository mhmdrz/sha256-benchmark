# sha256-benchmark

Benchmarks SHA-256 compression backends on arm64 to find the fastest option
for a given CPU.

Three backends:

- **`scalar`** — portable C reference. The default algorithm; builds and runs
  anywhere. This is the baseline.
- **`fast`** — optimized portable software: fully unrolled rounds with register
  renaming (no per-round variable shuffling) and a rolling 16-word message
  schedule (no 64-word stack array). ~1.6× the scalar baseline, still pure C.
  This is the best path on cores *without* the crypto extension (e.g. Pi 4).
- **`armv8-ce`** — ARMv8-A Crypto Extension (`SHA256H`/`SHA256H2`/`SHA256SU0`/
  `SHA256SU1` via NEON intrinsics). Typically **5–10× faster**. Built on
  aarch64 and selected automatically at runtime *if the CPU implements it*.

At startup the tool verifies every backend against the NIST test vectors
(empty, `"abc"`, and 1,000,000 × `'a'`) and aborts if any fails.

## Multi-buffer (parallel) hashing — `--parallel`

A single SHA-256 stream is a serial dependency chain, so one hash can't be
sped up much past the single-block cost. Throughput workloads (hashing many
files/chunks) go faster by hashing **independent messages in parallel**:

- **`x4-neon`** — 4-way pure-software SIMD; each NEON lane carries one message.
  No crypto extension needed, so this is the fast path on cores like the Pi 4.
  ~1.4× the best single-stream *software* path.
- **`x8-neon`** — two interleaved 4-lane groups (8 messages at once). A single
  4-lane chain leaves the NEON units idle waiting on the per-round dependency
  chain; a second independent group fills those slots. ~2.4× single-stream
  software, i.e. ~4× the naive scalar backend — still pure C+NEON.
- **`x2-ce`** — 2-way interleave of the crypto instructions, to hide `SHA256H`
  latency. **Hardware-dependent:** wins on latency-bound cores (e.g. Pi 5 A76,
  typically 1.3–1.7×) but can *lose* on cores already throughput-bound on the
  crypto unit (measured ~12% slower on Apple Silicon). Benchmark your own chip.

```sh
./bench --parallel --mhz 2400
```

Both one-shot helpers assume the N input messages are **equal length** (the
common chunked-hashing case) so padding is shared across lanes. Each lane's
digest is verified against the scalar reference before timing.

## Which backend you get

`sha256_best_compress()` picks `armv8-ce` when the CPU advertises the SHA2
feature (Linux `HWCAP_SHA2`; always present on Apple Silicon), otherwise falls
back to `scalar`. Relevant for Raspberry Pi:

| Board | Core | Crypto ext? |
|---|---|---|
| Pi 3 / 3B+ / Zero 2 W | Cortex-A53 | ✅ yes |
| Pi 4 | Cortex-A72 | ❌ no — falls back to scalar |
| Pi 5 | Cortex-A76 | ✅ yes |

Check your CPU directly: `grep -o sha2 /proc/cpuinfo | head -1`.

## Build & run

```sh
make
./bench
```

### Options

```
--mhz N        CPU clock in MHz; adds a cycles/byte column (the comparable metric)
--sizes a,b,c  input sizes in bytes (default 64,256,1024,8192,65536,1048576)
--seconds S    measurement budget per size (default 0.5)
--trials N     trials per size; best is reported (default 5)
--backend B    scalar | fast | armv8 | all (default all available)
```

Example:

```sh
./bench --mhz 2400 --sizes 1024,1048576
```

## Getting trustworthy numbers on a Pi

Sustained hashing throttles a Pi, which wrecks benchmarks. Before measuring:

```sh
# pin the clock high
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# pin the benchmark to one core (and watch for throttling)
taskset -c 3 ./bench
vcgencmd get_throttled    # non-zero => you were throttled; add cooling
```

Pass the measured core clock to `--mhz` to read cycles/byte; the crypto
extension lands around ~1.8–2.0 cpb, the scalar path around ~18–20 cpb.

## Layout

| File | Purpose |
|---|---|
| `sha256.h` / `sha256.c` | streaming init/update/final + backend selection |
| `sha256_scalar.c` | portable scalar compression (default algo) |
| `sha256_fast.c` | optimized portable software compression |
| `sha256_mb.c` | multi-buffer parallel hashing (x4-neon, x2-ce) |
| `sha256_armv8.c` | ARMv8 crypto-extension compression + CPU probe |
| `bench.c` | self-test, timing harness, CLI |
