# DOT4 Flash Attention & Packed16 K Cache

## Part 1: Architecture & Performance

---

## What This Is

A native RDNA3 DOT4 (`sudot4`) FlashAttention kernel for llama.cpp HIP. Compresses the QK dot product from 256 scalar multiplies to 64 packed `v_dot4_i32_iu8` instructions, running inside a tiled online-softmax loop. Two modes:

| Mode | K storage | PPL | VRAM at c512 |
|------|-----------|-----|-------------|
| **f16 K (default)** | Standard f16 KV cache | **1.0128** (lossless) | 40.50 MiB |
| **packed16-only K** | I32 payload + F16 scales | **1.0342** (+2.1%) | **24.50 MiB** (−40%) |

At 64k context, packed16-only saves ~960 MB vs f16 K cache.

**One sentence**: Same PPL (or near-same with VRAM savings), INT8 math via RDNA3 tensor instructions.

---

## Motivation

llama.cpp's ROCm FlashAttention currently has two paths for prefill:

| Route | Math | KV Cache | PPL cost | VRAM cost |
|---|---|---|---|---|
| VEC / tile (quantized-KV) | FP16 vec/tile | q8_0 / q4_0 / TBQ4 | Lossy (quantization) | Low |
| MMA-f16 (WMMA temp) | FP16 WMMA | f16 temp buffer | Lossless | High (allocates temp) |

This is the standard tradeoff: choose between PPL fidelity and VRAM. Our kernel breaks the tradeoff in two ways:

1. **f16 K mode**: Uses INT8 `sudot4` math on the **default f16 KV cache**, delivering FP16-equivalent PPL (1.0128) without allocating a temporary FP16 buffer.
2. **packed16-only K mode**: Eliminates f16 K cache entirely — K is stored as compact I32 payload + F16 scales. 40% VRAM reduction with only 2.1% PPL cost from INT8 quantization. No f16 K allocation, no temp buffer, no format lock-in.

RDNA3 GPUs have `sudot4` (4× int8 dot product per cycle) but weak FP16 tensor support — WMMA requires strict 16×16×16 register layouts. Existing attention paths either ignore `sudot4` entirely or try to use it through WMMA's constrained tile contract. Our kernel calls `sudot4` directly in a scalar loop, avoiding WMMA's register pressure and layout constraints while still getting 4× the arithmetic density of scalar FP16.

The FlashAttention-2 paper (Dao et al.) established the tiled online-softmax design with large matmul tiles (M=128, N=128) sized for tensor cores. Our tiles are BM=8, BN=8 — sized for the scalar DOT4 body, not tensor core geometry. Making them larger just launches more scalar loops; it doesn't change compute density.

Attention is 0.9% of total GPU time on Qwen3.6-27B (rocprof). The primary value is not throughput — it's correctness-safe acceleration plus the VRAM savings from packed16 K cache.

---

---

## llama.cpp Context

Our kernel coexists with the existing attention routes in the same repository:

| Route | Lines | Math | KV Cache | Use Case |
|---|---|---|---|---|
| `fattn-vec.cuh` (VEC) | 1040 | FP16 vec | Quantized (q8_0/q4_0/TBQ4) | Decode (1–2 tokens) |
| `fattn-tile.cuh` (tile) | 1309 | FP16 tile | Quantized | Prefill (GQA, split-KV) |
| `fattn-mma-f16.cuh` (MMA) | 2163 | FP16 WMMA | FP16 temp buffer | Prefill (batch > threshold) |
| **Our DOT4 FA** | **3793** | **INT8 sudot4** | **f16 or packed16 I32** | **Prefill + decode** |

Every existing route either uses a quantized KV cache (VEC, tile — lossy, format lock-in) or an FP16 temporary buffer (MMA — high VRAM, lossless). Our kernel is the first that delivers FP16-equivalent PPL on f16 KV cache without a temporary buffer, AND supports packed16-only K storage for VRAM savings.

The 3793-line count includes Q quantization, K packing, 7 experimental variants, split-K combine, timing infrastructure, the tensor registry, and the packed16 cache management layer. The kernel body is ~190 lines. Existing routes spread equivalent infrastructure across separate headers and dispatch logic.

The CK-Tile FlashAttention (AMD's official ROCm reference) uses template-based tile composition with large FP16 tiles. We bypass it entirely — raw HIP with manual LDS lets us call `sudot4` directly and manage the packed16 layout without fitting into a GEMM-centric tile contract.

## Core Design

Three components that together unlock INT8 FlashAttention on consumer RDNA3:

| Component | What | Why it matters alone | Why it matters together |
|---|---|---|---|
| **Packed16 K cache** | f16 K → I32 payload + F16 scales, 16B-friendly layout | 2× KQ throughput vs q8_0 blocks | Separates K layout from K format — kernel gets DOT4-ready data without KV cache format change |
| **I8 WMMA-sized 16×16 tile** | QK computed as macro tile with `sudot4` inner product | 4× arithmetic density vs FP16, no WMMA register constraints | Tile sized for INT8 work decomposition, not FP16 tensor core geometry |
| **Raw HIP kernel** | `__global__` function, manual LDS, no CK-Tile/rocWMMA | Full control over `sudot4` calls and packed16 layout | ~190-line kernel body — no library overhead, no GEMM-centric tile contract |

Individually, none of these are novel. Packed INT8 layouts exist in Vulkan shaders. WMMA-sized tiles are standard FA2. Raw HIP kernels are how all of llama.cpp works. The combination is what makes it work.

---

## System Context

| Parameter | Value |
|-----------|-------|
| GPU | RX 7900 XTX, gfx1100, 24.6 GB VRAM, wave32 |
| ROCm | 7.2.3, hipcc, hipBLAS |
| Model (testing) | Qwen3.6-27B Q4_K_M, 48 layers, GQA=6, 2560 hidden |
| Build | `build-rocm-rdna3-fa`, `GGML_HIPBLAS=ON`, `GGML_HIP_ROCWMMA_FATTN=ON` |
| Arch constraint | RDNA3/4 only (`sudot4` intrinsic) |

---

## Architecture

### Dataflow (f16 K mode)
```
┌─────────────────────────────────────────────────────────┐
│                    KV Cache (f16 K + f16 V)              │
│                    No format change required             │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
     ┌───────▼────────┐          ┌──────▼──────┐
     │  cpy_k()        │          │  f16 V      │
     │  f16→packed16   │          │  direct load │
     │  quant kernel   │          │  (no dequant)│
     └───────┬────────┘          └──────┬──────┘
             │                          │
     ┌───────▼──────────────────────────▼─────────┐
     │         DOT4 FlashAttention Kernel           │
     │  BM=8, BN=8/16, D=256, sudot4 KQ + PV       │
     └──────────────────────────────────────────────┘
```

### Dataflow (packed16-only K mode)
```
┌─────────────────────────────────────────────────────────┐
│           KV Cache: I32 payload + F16 scales              │
│           No f16 K allocation.  Compact per-head layout. │
└────────────┬──────────────────────────┬─────────────────┘
             │                          │
     ┌───────▼────────┐          ┌──────▼──────┐
     │  cpy_k()        │          │  f16 V      │
     │  f32→int8 quant │          │  direct load │
     │  (indexed kernel)│          │  (no dequant)│
     │  via k_idxs     │          │              │
     └───────┬────────┘          └──────┬──────┘
             │                          │
     ┌───────▼──────────────────────────▼─────────┐
     │         DOT4 FlashAttention Kernel           │
     │  reads k_payload/k_scales from registry      │
     │  fixed kv_size head stride for persistent K  │
     │  BM=8, BN=8/16, D=256, sudot4 KQ + PV       │
     └──────────────────────────────────────────────┘
```

### Packed16-Only K Details

| Component | Detail |
|-----------|--------|
| Payload tensor | `GGML_TYPE_I32`, shape `[D/4, kv_size × n_head_kv, n_stream]` |
| Scales tensor | `GGML_TYPE_F16`, shape `[D/32, kv_size × n_head_kv, n_stream]` |
| Per-head row width | 64 I32 values (= 256 int8 values packed 4-per-I32) |
| Row layout | Head-major: rows 0..511=head0, 512..1023=head1, etc. |
| Pack kernel | Indexed: uses `k_idxs` to write absolute cache slots |
| Head stride | Fixed `kv_size` (not active `nk`) for persistent cache |
| Qwen3.6-27B D=256 | Payload [64, 2048], Scales [8, 2048] |
| VRAM at c512 | K: 8.50 MiB, V: 16.00 MiB, Total: 24.50 MiB |
| Env gate | `GGML_CUDA_ROCM_Q8K_DOT4_PACKED16_K_CACHE=1` |
| MSE quantization | 2-pass: amax initial scale → warp reduction → optimal L2 scale |

### Route Contract (updated)

- **Q**: F32, D=256
- **K**: Q8_0, F16, or **I32 packed16**
- **V**: Q4_0, F16
- **I32 K**: requires F16 V, DOT4 route enabled, per-head `K->ne[0] = D/4`
- **GQA**: required (Q heads % K heads == 0)
- **No**: sinks, max_bias, logit_softcap
- **Mask**: F16 only, correct shape
- **Gating**: All paths behind `GGML_CUDA_ROCM_EXPERIMENTAL_UNSAFE=1` + route-specific flags. No default route change.

---

## Key Performance Numbers

### PPL Correctness (Qwen3.6-27B Q4_K_M, gfx1100)

| Mode | K storage | PPL | vs baseline |
|------|-----------|-----|-------------|
| Baseline f16 FA | f16 | **1.0128** | — |
| DOT4 FA (f16 K) | f16 + DOT4 math | **1.0128** | identical |
| **DOT4 FA (packed16-only K)** | **I32 int8** | **1.0342** | **+2.1%** |

```
f16 K:    815-830 t/s, PPL 1.0128
packed16: 820-850 t/s, PPL 1.0342, KV cache 24.50 MiB
```

### Throughput (pp1536, gfx1100)

| Metric | Value |
|--------|-------|
| Prefill | 820–850 t/s |
| Decode (tg16) | 22–23 t/s |
| DOT4 KQ ceiling (standalone microbench) | ~874 t/s |

### Packed16 VRAM Savings

| Context | f16 K cache | packed16 K cache | Savings |
|---------|------------|-----------------|---------|
| c512 | 34.00 MiB | 8.50 MiB | **−75%** |
| c64k (projected) | ~4.0 GiB | ~1.0 GiB | **−75%** |

### KQ Microbench (standalone, nq=256, nk=1024, D=256)

| Layout | ms | DOT4 GOP/s | vs q8_0 baseline |
|--------|----:|-----------:|------------------:|
| q8_0 block (34B) | 0.0385 | 435.8 | 1.00× |
| **packed16 payload + q8 scales** | **0.0193** | **870.9** | **2.00×** |
| packed16 + one row scale | 0.0104 | 1606.4 | 3.69× |

Packed16 layout alone delivers 2× the KQ throughput of native q8_0 blocks.

![KQ Microbench bar chart](graphs/kq-microbench.svg)

### Tile16 Scale-Ceiling Probe

| nq | nk | q8block ms | packed16 ms | scale-once ms | packed speedup |
|----:|----:|-----------:|------------:|--------------:|---------------:|
| 256 | 1024 | 0.0316 | 0.0181 | 0.0173 | 1.75× |
| 512 | 2048 | 0.1361 | 0.0672 | 0.0570 | 2.02× |
| 1024 | 4096 | 0.3083 | 0.1080 | 0.1011 | 2.85× |

### blockFA Route A/B — Skeleton (May 26, pre-rebuild)

| Prompt | Default tok/s | blockFA tok/s | Delta |
|--------|--------------:|--------------:|------:|
| **pp512** | 902.2 | 842.4 | **-6.6%** |
| **pp1024** | 772.2 | 816.8 | **+5.8%** |
| **pp2048** | 925.5 | 723.3 | **-21.9%** |

### Design Space Exploration (each rejection informed the final architecture)

| Experiment | Result | What it taught us |
|-----------|--------|-------------------|
| BN16 wider K tiles | No gain | Per-row V dequant cost scales linearly with BN |
| BM16 more Q rows | No gain | Launch overhead already negligible at BM=8 |
| K LDS staging | **-0.5–1.0%** | LDS traffic exceeds reuse benefit |
| Cooperative warp2 KQ | **Slower** | Shuffle reduction overhead > lane parallelism |
| Register 2×2 micro-tile | **1.2–7.4× slower** | Register pressure kills occupancy |
| LDS tile cache (16×16 QK) | **4.5–5.9× slower** | Direct global/cache path beats explicit staging |
| Fused V dequant | No gain | QK dot fully occupies SIMD units |

---

## Rectangular History Fix (recthist v2)

The pp2048 cliff (-21.9%) is caused by the kernel treating all 2048 keys as causal, when only the tail (k ≥ q_offset) needs causal masking.

```
Prefix k ∈ [0, q_offset):  full non-causal, exact tiles, no lane masking
Tail   k ∈ [q_offset, nk): causal check per (q,k)
Merge: single online softmax state across both regions
```

| Kernel | Total ms | Prefix ms | Tail ms | Merge ms | Rel RMS |
|--------|---------:|----------:|--------:|---------:|--------:|
| **recthist_v2_prefix_exact** | **1.400** | 0.930 | 0.650 | 0.017 | 8.2e-07 |
| recthist_v2 | 1.418 | 0.936 | 0.615 | 0.018 | 8.2e-07 |

**Correctness**: 8.2e-07 relative RMS vs CPU reference. **191.7 DOT4 GOP/s**, **766.9 PV GOP/s**.

---

## The Matmul Ceiling

![GPU time pie chart](graphs/gpu-time-pie.svg)

No amount of attention optimization moves throughput past ~870 t/s. The DOT4 FA kernel is already at 820–850 t/s — within 2–5% of the ceiling. Further attention work is about **VRAM savings** (packed16 K cache swap), not tokens per second.

---

## Open Work

1. **Splice recthist v2 into route**: Replace scalar blockFA body with prefix/tail split. Target: eliminate -21.9% pp2048 cliff.

2. **Wire into fattn.cu dispatch**: Add DOT4 FA to normal route selection (not just `FA_ROUTE_REQUIRE` override).

3. **Mixed-precision K cache**: f16 K window for recent tokens + packed16 for older tokens. Combines baseline PPL with VRAM savings.

4. **Production A/B**: Disciplined r≥5 A/B against stable default on pp512/1024/2048/4096.

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
- Dao, "FlashAttention-2: Better Attention with Better Parallelism" (2023)
- AMD, "Implementing FlashAttention with CK-Tile" (ROCm blog)
- `docs/rocm-tbq4-paths/09-q8q4-dot4-packed-k-direction.md`
- `docs/rocm-tbq4-paths/23-q8k-dot4-fa-root-cause-and-next-plan.md`
- `docs/rocm-tbq4-paths/24-wmma-sized-dot4-tile-plan.md`
- `.harness/research/packed16-only-k-implementation-plan.md`
- `.harness/research/packed16-only-k-crash-fix-instructions-20260527.md`
- `.harness/research/packed16-only-k-nan-fix-instructions-20260527.md`
- `.harness/research/packed16-only-k-final-nan-fix-instructions-20260527.md`
