// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MonoKernel phase 2a (gfx1030): the whole Qwen4Exp (Flash-Next) FFN block of
// one decoder layer for a pure decode step (M <= 24 rows) in ONE persistent
// launch. It replaces the ~21 (M <= 8) / ~28 (M > 8) kernels between the
// attention all-reduce and the FFN all-reduce; both all-reduces stay outside.
//
//   P1  HC combine (residual + attn_out * 2 sigmoid(inj/HC)) + grouped Gemma
//       RMSNorm -> res_out, xn                      (zero the fp32 scratch)
//   P2  HC down+inject GEMV  dai = xn . Wd^T        (K split, fp32 atomics)
//   P3  HC up GEMV + sigmoid + gated mean -> block_input; inj_out
//   P4  router GEMV (fp32 logits), shared-expert gate_up + silu, expert gate
//   P5  fp32 softmax top-k (every WG, redundant), local expert lists;
//       routed gate_up + SwiGLU per (expert, 16 columns); shared-expert down
//   P6  routed down per (expert, 32 columns), * top-k weight, fp32 atomics
//   P7  out = fp16(shared + routed)                 (rank-local partial)
//
// Phases are separated by a grid barrier on a sense counter in global memory
// (agent-scope atomics, bounded wait, no grid.sync). A timed-out wait writes a
// sticky {code, phase, wg} record and every WG leaves; the record stays set
// until the host clears it, and a launch that starts with it set exits at
// once (fail-stop: the caller must check mono_ffn_rdna2_error()).
//
// Plain launch, one stream, capturable. The grid must be co-resident: the
// default is one WG per CU (P0 probe: 72 WGs x 256 threads x 64 KiB LDS are
// co-resident on a V620 in WGP and CU mode); this kernel uses < 32 KiB.
//
// Weight layouts are the ones the current decode path reads:
//   Wd [ND, HC*H], Wu [HC*H, R], router [E, H], se_w1 [2*Ise, H] (gate rows
//   then up rows), se_w2 [H, Ise], se_wg [H]  -- fp16, row-major
//   w13 [El, H/8, 2I] / w2 [El, I/8, H] uint32 (exllama-shuffled int4),
//   s13 / s2 [El, groups, N] fp16, z13 / z2 [El, groups, N/8] packed zeros
//   (moe_gptq_gemm_rdna2's layout, symmetric).

#include <algorithm>
#include <cstdint>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q_gemm_rdna2_common.cuh"
#include "qdq_4_rdna2.cuh"

namespace mono_ffn {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kMaxM = 24;
constexpr int kTopK = 10;
constexpr int kMaxPairs = kMaxM * kTopK;
constexpr int kLocalE = 128;
constexpr int kHC = 4;
constexpr int kRowChunk = 8;

struct Ctrl {
  unsigned count;  // barrier arrivals
  unsigned gen;    // barrier generation
  int err_code;    // sticky: 0 ok, 1 barrier timeout
  int err_phase;
  int err_wg;
  int pad[3];
  unsigned long long ts[8];  // WG0 wall clock at entry and after each phase
  unsigned long long wg_done[7][256];  // per-WG end of phase work (debug)
};

struct Params {
  // activations
  const half* res;    // [M, HC*H]
  const half* blk;    // [M, H]   attention output (after its all-reduce)
  const half* inj;    // [M, HC]  pending injection logits
  half* res_out;      // [M, HC*H]
  half* inj_out;      // [M, HC]
  half* out;          // [M, H]   rank-local FFN partial (pre all-reduce)
  // weights
  const half* norm_w;  // [HC*H]
  const half* wd;      // [ND, HC*H]
  const half* wu;      // [HC*H, R]
  const half* wr;      // [E, H] router
  const half* se_w1;   // [2*Ise, H]
  const half* se_w2;   // [H, Ise]
  const half* se_wg;   // [H]
  const uint32_t* w13;
  const half* s13;
  const uint32_t* z13;
  const uint32_t* w2;
  const half* s2;
  const uint32_t* z2;
  // workspace
  Ctrl* ctrl;
  float* dai;       // [M, ND]
  float* acc;       // [M, H]
  half* xn;         // [M, HC*H]
  half* bin;        // [M, H]
  float* logits;    // [M, E]
  float* se_g;      // [M, 2*Ise] shared-expert gate|up pre-activation
  float* se_gate;   // [M] expert-gate logit
  half* act;        // [M*topk, I]
  // shapes
  int M, H, R, ND, E, Ise, I, group, expert_base, zero_offset;
  float eps;
  int prefetch;  // 1: waves 1..7 touch the next phase's weights while waiting
  unsigned long long timeout_ticks;
};

__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float sigmoid_f(float v) { return 1.f / (1.f + __expf(-v)); }

template <int N>
__device__ __forceinline__ void wave_sum(float (&a)[N]) {
#pragma unroll
  for (int m = 0; m < N; m++) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) a[m] += __shfl_xor(a[m], off);
  }
}

// ----------------------------------------------------------------- barrier
__device__ __forceinline__ bool err_set(Ctrl* c) {
  return __hip_atomic_load(&c->err_code, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT) != 0;
}

// Returns false if this WG must leave (timeout here or elsewhere).
struct NoPrefetch {
  __device__ void operator()(int, uint32_t&, const volatile int*) const {}
};

// Returns false if this WG must leave (timeout here or elsewhere). Wave 0
// arrives and polls; waves 1..7 meanwhile run `prefetch(t, sink)` (t =
// thread index among them), which touches the weights of this WG's tasks in
// the next phase so they come from L2 / Infinity Cache instead of DRAM.
template <typename F = NoPrefetch>
__device__ bool grid_barrier(const Params& p, int phase, int* s_flag,
                             F prefetch = F()) {
  __shared__ volatile int s_rel;  // set once the barrier has released
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  if (threadIdx.x == 0) s_rel = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      if (blockIdx.x < 256) p.ctrl->wg_done[phase - 1][blockIdx.x] = wall_clock64();
      Ctrl* c = p.ctrl;
      const unsigned G = gridDim.x;
      const unsigned g = __hip_atomic_load(&c->gen, __ATOMIC_ACQUIRE,
                                           __HIP_MEMORY_SCOPE_AGENT);
      const unsigned arrived = __hip_atomic_fetch_add(
          &c->count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      int ok = 1;
      if (arrived == G - 1) {
        __hip_atomic_store(&c->count, 0u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_fetch_add(&c->gen, 1u, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else {
        const unsigned long long t0 = wall_clock64();
        // Relaxed polling: an acquire load per poll would invalidate the
        // GL0/GL1 caches of WGs that are still working; the acquire fence
        // after the barrier orders the data.
        while (__hip_atomic_load(&c->gen, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT) == g) {
          if (err_set(c)) { ok = 0; break; }
          if (wall_clock64() - t0 > p.timeout_ticks) {
            int zero = 0;
            if (__hip_atomic_compare_exchange_strong(
                    &c->err_code, &zero, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                    __HIP_MEMORY_SCOPE_AGENT)) {
              c->err_phase = phase;
              c->err_wg = blockIdx.x;
            }
            ok = 0;
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
      if (ok && err_set(c)) ok = 0;
      *s_flag = ok;
      s_rel = 1;
    }
  } else {
    uint32_t sink = 0;
    if (p.prefetch) prefetch(threadIdx.x - 32, sink, &s_rel);
    if (sink == 0x9e3779b9u && p.M < 0) p.ctrl->pad[1] = sink;  // keep the loads
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return *s_flag != 0;
}

// Touch [base, base + n16 * 16) in 16-byte units, strided over 224 threads.
__device__ __forceinline__ void touch16(const void* base, int n16, int t,
                                        uint32_t& sink, const volatile int* rel) {
  const uint4* q = reinterpret_cast<const uint4*>(base);
  for (int i = t; i < n16; i += 224) {
    if (*rel) return;  // released: stop, the next phase is on the clock
    const uint4 v = q[i];
    sink ^= v.x ^ v.w;
  }
}

// ------------------------------------------- 4 fp16 weight rows x 8 act rows
// One wave, lanes striding K in 8-element (16 B) steps over [lo, hi). Each x
// load is shared by the 4 weight rows; acc[r][m] for x rows m0 .. m0+7.
__device__ __forceinline__ void rows4_dot(const half* const (&w)[4], int nw,
                                          const half* __restrict__ x, int ldx,
                                          int m0, int M, int lo, int hi,
                                          int lane, float (&acc)[4][8]) {
  for (int i0 = lo + lane; i0 < hi; i0 += 64) {
    uint4 wq[2][4];
#pragma unroll
    for (int u = 0; u < 2; u++)
#pragma unroll
      for (int r = 0; r < 4; r++)
        wq[u][r] = (r < nw && i0 + 32 * u < hi)
                       ? reinterpret_cast<const uint4*>(w[r])[i0 + 32 * u]
                       : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int i = i0 + 32 * u;
      if (i >= hi) break;
#pragma unroll
      for (int m = 0; m < 8; m++) {
        if (m0 + m < M) {
          const uint4 xq =
              reinterpret_cast<const uint4*>(x + (size_t)(m0 + m) * ldx)[i];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const half2* wh = reinterpret_cast<const half2*>(&wq[u][r]);
            float a = acc[r][m];
            a = __builtin_amdgcn_fdot2(wh[0], xh[0], a, false);
            a = __builtin_amdgcn_fdot2(wh[1], xh[1], a, false);
            a = __builtin_amdgcn_fdot2(wh[2], xh[2], a, false);
            a = __builtin_amdgcn_fdot2(wh[3], xh[3], a, false);
            acc[r][m] = a;
          }
        }
      }
    }
  }
}

// Sum 32 per-lane values over the wave in 31 shuffles; lane L returns the
// total of value L (here acc[L / 8][L % 8]).
__device__ __forceinline__ float wave_transpose_sum(float (&acc)[4][8], int lane) {
  float v[32];
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int m = 0; m < 8; m++) v[r * 8 + m] = acc[r][m];
#pragma unroll
  for (int half_n = 16; half_n >= 1; half_n >>= 1) {
    const bool up = (lane & half_n) != 0;
#pragma unroll
    for (int i = 0; i < half_n; i++) {
      const float send = up ? v[i] : v[i + half_n];
      const float keep = up ? v[i + half_n] : v[i];
      v[i] = keep + __shfl_xor(send, half_n);
    }
  }
  return v[0];
}

__device__ __forceinline__ void zero48(float (&acc)[4][8]) {
#pragma unroll
  for (int r = 0; r < 4; r++)
#pragma unroll
    for (int m = 0; m < 8; m++) acc[r][m] = 0.f;
}

// Partial int4 GEMM of one wave: 4 consecutive columns per lane at `col`,
// k8 rows [k8_lo, k8_hi) of a [K/8, N] packed matrix, against nr activation
// rows (x row r at xrow[r], 8 halfs per k8). Adds into acc[r][4].
template <int NR>
__device__ __forceinline__ void q4_cols4(const uint32_t* __restrict__ wq,
                                         const half* __restrict__ sc,
                                         const uint32_t* __restrict__ zq, int N,
                                         int col, int group8, int zoff,
                                         int k8_lo, int k8_hi,
                                         const half* const (&xrow)[NR], int nr,
                                         float (&acc)[NR][4]) {
  if (k8_lo >= k8_hi) return;
  // The slice spans at most 4 quant groups; fetch their zeros/scales up front
  // so the group switch never waits on memory.
  constexpr int kMaxG = 4;
  const int g_lo = k8_lo / group8;
  uint32_t zw[kMaxG];
  uint2 sw[kMaxG];
#pragma unroll
  for (int q = 0; q < kMaxG; q++) {
    const int g = min(g_lo + q, (k8_hi - 1) / group8);
    zw[q] = zq[(size_t)g * (N / 8) + col / 8];
    sw[q] = *reinterpret_cast<const uint2*>(sc + (size_t)g * N + col);
  }
  half2 zz[4][2], yy[4][2], s2[4];
  int cur = -1;
  constexpr int U = 8;
  for (int k0 = k8_lo; k0 < k8_hi; k0 += U) {
    uint4 w[U];
#pragma unroll
    for (int u = 0; u < U; u++)
      w[u] = (k0 + u < k8_hi)
                 ? *reinterpret_cast<const uint4*>(wq + (size_t)(k0 + u) * N + col)
                 : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int k8 = k0 + u;
      if (k8 >= k8_hi) break;
      const int q = k8 / group8 - g_lo;
      if (q != cur) {
        cur = q;
        uint32_t zsel = zw[0];
        uint2 ssel = sw[0];
#pragma unroll
        for (int t = 1; t < kMaxG; t++)
          if (q == t) { zsel = zw[t]; ssel = sw[t]; }
        const uint32_t d = zsel >> ((col & 7) * 4);
        const half* sh = reinterpret_cast<const half*>(&ssel);
#pragma unroll
        for (int c = 0; c < 4; c++) {
          vllm::gptq_rdna2::prep_zero_scale_fp16(((d >> (4 * c)) & 0xF) + zoff,
                                                 __float2half_rn(1.0f), zz[c],
                                                 yy[c]);
          s2[c] = __half2half2(sh[c]);
        }
      }
      half2 dq[4][4];
      const uint32_t* wu = reinterpret_cast<const uint32_t*>(&w[u]);
#pragma unroll
      for (int c = 0; c < 4; c++) {
        vllm::gptq_rdna2::dequant_4bit_8_fp16(wu[c], dq[c], zz[c], yy[c]);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) dq[c][jj] = __hmul2(dq[c][jj], s2[c]);
      }
#pragma unroll
      for (int r = 0; r < NR; r++) {
        if (r < nr) {
          const uint4 xq = reinterpret_cast<const uint4*>(xrow[r])[k8];
          const half2* xh = reinterpret_cast<const half2*>(&xq);
#pragma unroll
          for (int c = 0; c < 4; c++) {
            float v = acc[r][c];
            v = __builtin_amdgcn_fdot2(dq[c][0], xh[0], v, false);
            v = __builtin_amdgcn_fdot2(dq[c][1], xh[1], v, false);
            v = __builtin_amdgcn_fdot2(dq[c][2], xh[2], v, false);
            v = __builtin_amdgcn_fdot2(dq[c][3], xh[3], v, false);
            acc[r][c] = v;
          }
        }
      }
    }
  }
}

// int4 column dequant constants for one (expert, group, column).
__device__ __forceinline__ void q4_prep(const uint32_t* z, const half* s,
                                        int group, int N, int col, int zoff,
                                        half2 (&zz)[2], half2 (&yy)[2],
                                        half2& sc) {
  const uint32_t zw = z[(size_t)group * (N / 8) + col / 8];
  const uint32_t zero = (zw >> ((col & 7) * 4)) & 0xF;
  vllm::gptq_rdna2::prep_zero_scale_fp16(zero + zoff, __float2half_rn(1.0f),
                                         zz, yy);
  sc = __half2half2(s[(size_t)group * N + col]);
}

// ------------------------------------------------------------------ kernel
#ifndef MONO_FFN_MIN_BLOCKS
#define MONO_FFN_MIN_BLOCKS 1
#endif
// 1: one WG (8 waves) per CU; 2: two WGs per CU (<= 128 VGPRs).
__global__ void __launch_bounds__(kThreads)
    __attribute__((amdgpu_waves_per_eu(4 * MONO_FFN_MIN_BLOCKS, 16)))
    mono_ffn_kernel(const Params p) {
  __shared__ int s_flag;
  __shared__ float s_red[kWaves];
  __shared__ half s_lora[kMaxM * 320];  // P3 lora; reused as the P5 SE act
  __shared__ int s_tid[kMaxPairs];
  __shared__ float s_tw[kMaxPairs];
  __shared__ int s_used[kLocalE];          // used-expert slot -> local expert
  __shared__ int s_ucnt[kLocalE];          // rows per used expert
  __shared__ int s_slot[kLocalE];          // local expert -> used slot (-1)
  __shared__ short s_urow[kLocalE * kMaxM];  // pair ids per used expert
  __shared__ int s_nused;
  __shared__ float s_part[kRowChunk * 128];  // per-WG partial sums (P5b/P6)
  half* const s_seact = s_lora;  // shared-expert activation (P5), M x Ise

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int G = gridDim.x, wg = blockIdx.x;
  const int gw = wg * kWaves + wave, GW = G * kWaves;
  const int M = p.M, H = p.H, HCH = kHC * p.H, R = p.R, ND = p.ND;

  if (err_set(p.ctrl)) return;  // fail-stop: a previous launch timed out
  if (wg == 0 && tid == 0) {
    p.ctrl->ts[0] = wall_clock64();
    p.ctrl->pad[0] = G;  // grid actually launched
  }

  // ---------------- P1: combine + grouped RMSNorm; zero scratch
  for (int i = wg * kThreads + tid; i < M * ND; i += G * kThreads) p.dai[i] = 0.f;
  for (int i = wg * kThreads + tid; i < M * H; i += G * kThreads) p.acc[i] = 0.f;
  for (int i = wg * kThreads + tid; i < M * p.E; i += G * kThreads) p.logits[i] = 0.f;
  for (int i = wg * kThreads + tid; i < M * 2 * p.Ise; i += G * kThreads) p.se_g[i] = 0.f;
  for (int i = wg * kThreads + tid; i < M; i += G * kThreads) p.se_gate[i] = 0.f;
  for (int item = wg; item < M * kHC; item += G) {
    const int m = item / kHC, c = item % kHC;
    const float injc = 2.f * sigmoid_f(__half2float(p.inj[m * kHC + c]) / kHC);
    const half* rr = p.res + (size_t)m * HCH + c * H;
    const half* bb = p.blk + (size_t)m * H;
    half* ro = p.res_out + (size_t)m * HCH + c * H;
    float ss = 0.f;
    for (int h = tid; h < H; h += kThreads) {
      const half o = __float2half(__half2float(rr[h]) + __half2float(bb[h]) * injc);
      ro[h] = o;
      const float of = __half2float(o);
      ss += of * of;
    }
    for (int off = 16; off >= 1; off >>= 1) ss += __shfl_xor(ss, off);
    if (lane == 0) s_red[wave] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < kWaves; w++) tot += s_red[w];
    const float rrms = rsqrtf(tot / H + p.eps);
    const half* nw = p.norm_w + c * H;
    half* xo = p.xn + (size_t)m * HCH + c * H;
    for (int h = tid; h < H; h += kThreads) {
      float y = __half2float(ro[h]) * rrms;
      y += y * __half2float(nw[h]);
      xo[h] = __float2half(y);
    }
    __syncthreads();
  }
  {
    const int K8 = HCH / 8, NG = (ND + 3) / 4;
    const int ksplit = max(1, (2 * GW + NG - 1) / NG);
    const int chunk = (K8 + ksplit - 1) / ksplit;
    auto pf = [&](int tt, uint32_t& sink, const volatile int* rel) {
      for (int w8 = 0; w8 < kWaves; w8++)
        for (int t = wg * kWaves + w8; t < NG * ksplit; t += GW) {
          const int g = t % NG, sidx = t / NG;
          const int lo = sidx * chunk, hi = min(K8, lo + chunk);
          for (int r = 0; r < min(4, ND - 4 * g); r++)
            touch16(p.wd + (size_t)(4 * g + r) * HCH + lo * 8, hi - lo, tt, sink, rel);
        }
    };
    if (!grid_barrier(p, 1, &s_flag, pf)) return;
  }
  if (wg == 0 && tid == 0) p.ctrl->ts[1] = wall_clock64();

  // ---------------- P2: HC down+inject GEMV (4 rows per wave task, K split,
  // fp32 atomics into dai)
  {
    const int K8 = HCH / 8, NG = (ND + 3) / 4;
    const int ksplit = max(1, (2 * GW + NG - 1) / NG);
    const int chunk = (K8 + ksplit - 1) / ksplit;
    for (int t = gw; t < NG * ksplit; t += GW) {
      const int g = t % NG, sidx = t / NG;
      const int lo = sidx * chunk, hi = min(K8, lo + chunk);
      const int nw = min(4, ND - 4 * g);
      const half* w[4];
#pragma unroll
      for (int r = 0; r < 4; r++) w[r] = p.wd + (size_t)min(4 * g + r, ND - 1) * HCH;
      for (int m0 = 0; m0 < M; m0 += 8) {
        float acc[4][8];
        zero48(acc);
        rows4_dot(w, nw, p.xn, HCH, m0, M, lo, hi, lane, acc);
        const float v = wave_transpose_sum(acc, lane);
        const int r = lane / 8, m = m0 + lane % 8;
        if (r < nw && m < M) atomicAdd(&p.dai[m * ND + 4 * g + r], v);
      }
    }
  }
  {
    auto pf = [&](int tt, uint32_t& sink, const volatile int* rel) {
      for (int w8 = 0; w8 < kWaves; w8++)
        for (int h = wg * kWaves + w8; h < H; h += GW)
          for (int c = 0; c < kHC; c++)
            touch16(p.wu + (size_t)(c * H + h) * R, R / 8, tt, sink, rel);
    };
    if (!grid_barrier(p, 2, &s_flag, pf)) return;
  }
  if (wg == 0 && tid == 0) p.ctrl->ts[2] = wall_clock64();

  // ---------------- P3: HC up GEMV + sigmoid + gated mean -> block_input.
  // One wave task per h: the 4 rows c*H + h share every lora load.
  for (int i = tid; i < M * R; i += kThreads) {
    const int m = i / R, r = i % R;
    s_lora[i] = __float2half(silu_f(p.dai[m * ND + r] * (1.f / kHC)));
  }
  if (wg == 0 && tid < M * kHC)
    p.inj_out[tid] = __float2half(p.dai[(tid / kHC) * ND + R + tid % kHC]);
  __syncthreads();
  for (int h = gw; h < H; h += GW) {
    const half* w[4];
#pragma unroll
    for (int c = 0; c < 4; c++) w[c] = p.wu + (size_t)(c * H + h) * R;
    for (int m0 = 0; m0 < M; m0 += 8) {
      const int c = lane / 8, m = m0 + lane % 8;
      const float xv = m < M ? __half2float(p.xn[(size_t)m * HCH + c * H + h]) : 0.f;
      float acc[4][8];
      zero48(acc);
      rows4_dot(w, 4, s_lora, R, m0, M, 0, R / 8, lane, acc);
      float v = wave_transpose_sum(acc, lane);
      v = (m < M) ? sigmoid_f(v) * xv : 0.f;
      v += __shfl_xor(v, 8);
      v += __shfl_xor(v, 16);
      if (lane < 8 && m < M) p.bin[(size_t)m * H + h] = __float2half(v / kHC);
    }
  }
  {
    const int K8 = H / 8;
    const int NR = (p.E + 3) / 4, NS = p.Ise / 2, NT = NR + NS + 1;
    const int ksplit = max(1, GW / NT);
    const int chunk = (K8 + ksplit - 1) / ksplit;
    auto pf = [&](int tt, uint32_t& sink, const volatile int* rel) {
      for (int w8 = 0; w8 < kWaves; w8++)
        for (int t = wg * kWaves + w8; t < NT * ksplit; t += GW) {
          const int g = t % NT, sidx = t / NT;
          const int lo = sidx * chunk, hi = min(K8, lo + chunk);
          if (g < NR) {
            for (int r = 0; r < min(4, p.E - 4 * g); r++)
              touch16(p.wr + (size_t)(4 * g + r) * H + lo * 8, hi - lo, tt, sink, rel);
          } else if (g < NR + NS) {
            const int i0 = 2 * (g - NR);
            const int rr[4] = {i0, i0 + 1, p.Ise + i0, p.Ise + i0 + 1};
            for (int r = 0; r < 4; r++)
              touch16(p.se_w1 + (size_t)rr[r] * H + lo * 8, hi - lo, tt, sink, rel);
          }
        }
    };
    if (!grid_barrier(p, 3, &s_flag, pf)) return;
  }
  if (wg == 0 && tid == 0) p.ctrl->ts[3] = wall_clock64();

  // ---------------- P4: router (fp32 logits), shared-expert gate_up, expert
  // gate; 4 rows per wave task, K split, fp32 atomics.
  {
    const int K8 = H / 8;
    const int NR = (p.E + 3) / 4, NS = p.Ise / 2, NT = NR + NS + 1;
    const int ksplit = max(1, GW / NT);
    const int chunk = (K8 + ksplit - 1) / ksplit;
    for (int t = gw; t < NT * ksplit; t += GW) {
      const int g = t % NT, sidx = t / NT;
      const int lo = sidx * chunk, hi = min(K8, lo + chunk);
      const half* w[4];
      int nw = 4;
      if (g < NR) {
        nw = min(4, p.E - 4 * g);
#pragma unroll
        for (int r = 0; r < 4; r++) w[r] = p.wr + (size_t)min(4 * g + r, p.E - 1) * H;
      } else if (g < NR + NS) {
        const int i0 = 2 * (g - NR);  // gate i0, i0+1 ; up i0, i0+1
        w[0] = p.se_w1 + (size_t)i0 * H;
        w[1] = p.se_w1 + (size_t)(i0 + 1) * H;
        w[2] = p.se_w1 + (size_t)(p.Ise + i0) * H;
        w[3] = p.se_w1 + (size_t)(p.Ise + i0 + 1) * H;
      } else {
        nw = 1;
#pragma unroll
        for (int r = 0; r < 4; r++) w[r] = p.se_wg;
      }
      for (int m0 = 0; m0 < M; m0 += 8) {
        float acc[4][8];
        zero48(acc);
        rows4_dot(w, nw, p.bin, H, m0, M, lo, hi, lane, acc);
        const float v = wave_transpose_sum(acc, lane);
        const int r = lane / 8, m = m0 + lane % 8;
        if (r < nw && m < M) {
          if (g < NR) {
            atomicAdd(&p.logits[m * p.E + 4 * g + r], v);
          } else if (g < NR + NS) {
            const int i0 = 2 * (g - NR);
            const int col = r < 2 ? i0 + r : p.Ise + i0 + (r - 2);
            atomicAdd(&p.se_g[m * 2 * p.Ise + col], v);
          } else {
            atomicAdd(&p.se_gate[m], v);
          }
        }
      }
    }
  }
  if (!grid_barrier(p, 4, &s_flag)) return;
  if (wg == 0 && tid == 0) p.ctrl->ts[4] = wall_clock64();

  // ---------------- P5a: top-k (fp32 softmax, renormalized), every WG
  for (int m = wave; m < M; m += kWaves) {
    const float* lg = p.logits + (size_t)m * p.E;
    float v[16];  // E <= 512
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const int e = lane + 32 * j;
      v[j] = e < p.E ? lg[e] : -INFINITY;
      mx = fmaxf(mx, v[j]);
    }
    for (int off = 16; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off));
    float sel_sum = 0.f;
    for (int k = 0; k < kTopK; k++) {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < 16; j++) {
        const int e = lane + 32 * j;
        if (v[j] > bv || (v[j] == bv && e < bi)) { bv = v[j]; bi = e; }
      }
      for (int off = 16; off >= 1; off >>= 1) {
        const float ov = __shfl_xor(bv, off);
        const int oi = __shfl_xor(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (lane + 32 * j == bi) v[j] = -INFINITY;
      const float w = __expf(bv - mx);
      sel_sum += w;
      if (lane == 0) {
        s_tid[m * kTopK + k] = bi;
        s_tw[m * kTopK + k] = w;
      }
    }
    if (lane == 0) {
      const float inv = 1.f / sel_sum;
      for (int k = 0; k < kTopK; k++) s_tw[m * kTopK + k] *= inv;
    }
  }
  for (int e = tid; e < kLocalE; e += kThreads) {
    s_slot[e] = -1;
    s_ucnt[e] = 0;
  }
  __syncthreads();
  if (tid == 0) {  // <= 240 pairs; deterministic, identical in every WG
    int nu = 0;
    for (int pi = 0; pi < M * kTopK; pi++) {
      const int le = s_tid[pi] - p.expert_base;
      if (le < 0 || le >= kLocalE) continue;
      int u = s_slot[le];
      if (u < 0) {
        u = nu++;
        s_slot[le] = u;
        s_used[u] = le;
      }
      s_urow[u * kMaxM + s_ucnt[u]++] = (short)pi;
    }
    s_nused = nu;
  }
  __syncthreads();
  const int nused = s_nused;

  // ---------------- P5b: routed gate_up + SwiGLU. One WG per (expert, 64 gate
  // columns): lanes 0-15 take 4 gate columns each, lanes 16-31 the matching up
  // columns; the 8 waves split K and reduce in LDS.
  {
    const int I = p.I, N13 = 2 * I, K8 = H / 8, g8 = p.group / 8;
    const int nchunk = I / 64;
    const int kper = (K8 + kWaves - 1) / kWaves;
    for (int t = wg; t < nused * nchunk; t += G) {
      const int u = t / nchunk, c = t % nchunk;
      const int e = s_used[u];
      const int rows = s_ucnt[u];
      const int colg = c * 64 + (lane & 15) * 4;  // gate column of this lane
      const int col = lane < 16 ? colg : I + colg;
      const uint32_t* wq = p.w13 + (size_t)e * K8 * N13;
      const half* sc = p.s13 + (size_t)e * (H / p.group) * N13;
      const uint32_t* zq = p.z13 + (size_t)e * (H / p.group) * (N13 / 8);
      for (int r0 = 0; r0 < rows; r0 += kRowChunk) {
        const int nr = min(kRowChunk, rows - r0);
        for (int i2 = tid; i2 < kRowChunk * 128; i2 += kThreads) s_part[i2] = 0.f;
        __syncthreads();
        const half* xrow[kRowChunk];
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
          xrow[r] = p.bin + (size_t)(r < nr ? s_urow[u * kMaxM + r0 + r] / kTopK : 0) * H;
        float a[kRowChunk][4];
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
#pragma unroll
          for (int q = 0; q < 4; q++) a[r][q] = 0.f;
        q4_cols4<kRowChunk>(wq, sc, zq, N13, col, g8, p.zero_offset,
                            wave * kper, min(K8, (wave + 1) * kper), xrow, nr, a);
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
          if (r < nr)
#pragma unroll
            for (int q = 0; q < 4; q++) atomicAdd(&s_part[r * 128 + lane * 4 + q], a[r][q]);
        __syncthreads();
        for (int i2 = tid; i2 < nr * 64; i2 += kThreads) {
          const int r = i2 / 64, j = i2 % 64;
          const int pi = s_urow[u * kMaxM + r0 + r];
          // mirror the reference rounding: fp16 GEMM output, fp16 silu, fp16 mul
          const half g16 = __float2half(s_part[r * 128 + j]);
          const half u16 = __float2half(s_part[r * 128 + 64 + j]);
          const half sg = __float2half(silu_f(__half2float(g16)));
          p.act[(size_t)pi * I + c * 64 + j] =
              __float2half(__half2float(sg) * __half2float(u16));
        }
        __syncthreads();
      }
    }
    // shared-expert down: acc[m, h] = sigmoid(gate[m]) * (W2[h, :] . act[m]),
    // act = fp16(silu(g) * u) materialized per WG in LDS
    for (int k = tid; k < M * p.Ise; k += kThreads) {
      const int m = k / p.Ise, i2 = k % p.Ise;
      s_seact[k] = __float2half(silu_f(p.se_g[m * 2 * p.Ise + i2]) *
                                p.se_g[m * 2 * p.Ise + p.Ise + i2]);
    }
    __syncthreads();
    const int Ise8 = p.Ise / 8;
    for (int g = gw; g < H / 4; g += GW) {
      const half* w[4];
#pragma unroll
      for (int r = 0; r < 4; r++) w[r] = p.se_w2 + (size_t)(4 * g + r) * p.Ise;
      for (int m0 = 0; m0 < M; m0 += 8) {
        float acc[4][8];
        zero48(acc);
        rows4_dot(w, 4, s_seact, p.Ise, m0, M, 0, Ise8, lane, acc);
        const float v = wave_transpose_sum(acc, lane);
        const int r = lane / 8, m = m0 + lane % 8;
        if (m < M) p.acc[(size_t)m * H + 4 * g + r] = v * sigmoid_f(p.se_gate[m]);
      }
    }
  }
  {
    const int I = p.I, K8 = I / 8, nchunk = H / 128;
    auto pf = [&](int tt, uint32_t& sink, const volatile int* rel) {
      for (int t = wg; t < nused * nchunk; t += G) {
        const int u = t / nchunk, c = t % nchunk;
        const uint32_t* wq = p.w2 + (size_t)s_used[u] * K8 * H + c * 128;
        // 80 rows x 512 B: one 16-B unit per (row, 32-column slice)
        for (int i = tt; i < K8 * 32; i += 224) {
          if (*rel) return;
          const uint4 v = *reinterpret_cast<const uint4*>(wq + (size_t)(i / 32) * H + (i % 32) * 4);
          sink ^= v.x ^ v.w;
        }
      }
    };
    if (!grid_barrier(p, 5, &s_flag, pf)) return;
  }
  if (wg == 0 && tid == 0) p.ctrl->ts[5] = wall_clock64();

  // ---------------- P6: routed down. One WG per (expert, 128 columns), waves
  // split K, LDS reduce, * top-k weight, fp32 atomics into acc.
  {
    const int I = p.I, K8 = I / 8, g8 = p.group / 8;
    const int nchunk = H / 128;
    const int kper = (K8 + kWaves - 1) / kWaves;
    for (int t = wg; t < nused * nchunk; t += G) {
      const int u = t / nchunk, c = t % nchunk;
      const int e = s_used[u];
      const int rows = s_ucnt[u];
      const int col = c * 128 + lane * 4;
      const uint32_t* wq = p.w2 + (size_t)e * K8 * H;
      const half* sc = p.s2 + (size_t)e * (I / p.group) * H;
      const uint32_t* zq = p.z2 + (size_t)e * (I / p.group) * (H / 8);
      for (int r0 = 0; r0 < rows; r0 += kRowChunk) {
        const int nr = min(kRowChunk, rows - r0);
        for (int i2 = tid; i2 < kRowChunk * 128; i2 += kThreads) s_part[i2] = 0.f;
        __syncthreads();
        const half* xrow[kRowChunk];
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
          xrow[r] = p.act + (size_t)(r < nr ? s_urow[u * kMaxM + r0 + r] : 0) * I;
        float a[kRowChunk][4];
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
#pragma unroll
          for (int q = 0; q < 4; q++) a[r][q] = 0.f;
        q4_cols4<kRowChunk>(wq, sc, zq, H, col, g8, p.zero_offset,
                            wave * kper, min(K8, (wave + 1) * kper), xrow, nr, a);
#pragma unroll
        for (int r = 0; r < kRowChunk; r++)
          if (r < nr)
#pragma unroll
            for (int q = 0; q < 4; q++) atomicAdd(&s_part[r * 128 + lane * 4 + q], a[r][q]);
        __syncthreads();
        for (int i2 = tid; i2 < nr * 128; i2 += kThreads) {
          const int r = i2 / 128, j = i2 % 128;
          const int pi = s_urow[u * kMaxM + r0 + r];
          atomicAdd(&p.acc[(size_t)(pi / kTopK) * H + c * 128 + j],
                    s_part[r * 128 + j] * s_tw[pi]);
        }
        __syncthreads();
      }
    }
  }
  if (!grid_barrier(p, 6, &s_flag)) return;
  if (wg == 0 && tid == 0) p.ctrl->ts[6] = wall_clock64();

  // ---------------- P7: out = fp16(shared + routed)
  for (int i = wg * kThreads + tid; i < M * H; i += G * kThreads)
    p.out[i] = __float2half(p.acc[i]);
  if (wg == 0 && tid == 0) p.ctrl->ts[7] = wall_clock64();
}

}  // namespace mono_ffn

// Workspace size in bytes for M <= 24 (ctrl + scratch), 256-B aligned pieces.
int64_t mono_ffn_rdna2_workspace_bytes(int64_t H, int64_t R, int64_t ND,
                                       int64_t E, int64_t Ise, int64_t I) {
  using namespace mono_ffn;
  auto al = [](int64_t b) { return (b + 255) / 256 * 256; };
  const int64_t M = kMaxM;
  return al(sizeof(Ctrl)) + al(M * ND * 4) + al(M * H * 4) +
         al(M * kHC * H * 2) + al(M * H * 2) + al(M * E * 4) +
         al(M * 2 * Ise * 4) + al(M * 4) + al(M * kTopK * I * 2);
}

void mono_ffn_rdna2(const torch::Tensor& res, const torch::Tensor& blk,
                    const torch::Tensor& inj, const torch::Tensor& norm_w,
                    const torch::Tensor& wd, const torch::Tensor& wu,
                    const torch::Tensor& wr, const torch::Tensor& se_w1,
                    const torch::Tensor& se_w2, const torch::Tensor& se_wg,
                    const torch::Tensor& w13, const torch::Tensor& s13,
                    const torch::Tensor& z13, const torch::Tensor& w2,
                    const torch::Tensor& s2, const torch::Tensor& z2,
                    torch::Tensor& res_out, torch::Tensor& inj_out,
                    torch::Tensor& out, torch::Tensor& workspace,
                    int64_t expert_base, int64_t zero_offset, int64_t num_wg,
                    double eps, double timeout_ms) {
  using namespace mono_ffn;
  const int M = res.size(0);
  const int HCH = res.size(1);
  const int H = blk.size(1);
  TORCH_CHECK(HCH == kHC * H, "mono_ffn: res must be [M, 4*H]");
  TORCH_CHECK(M >= 1 && M <= kMaxM, "mono_ffn: 1 <= M <= 24");
  TORCH_CHECK(res.scalar_type() == at::kHalf && blk.scalar_type() == at::kHalf,
              "mono_ffn: fp16 only");
  const int ND = wd.size(0), R = wu.size(1), E = wr.size(0);
  const int Ise = se_w2.size(1), I = w2.size(1) * 8;
  const int El = w13.size(0);
  const int groups = s13.size(1);
  const int group = H / groups;
  TORCH_CHECK(R == 320 || R <= 320, "mono_ffn: lora rank <= 320");
  TORCH_CHECK(E <= 512 && El <= kLocalE, "mono_ffn: E <= 512, local <= 128");
  TORCH_CHECK(w13.size(2) == 2 * I && w13.size(1) * 8 == H, "mono_ffn: w13 shape");
  TORCH_CHECK(Ise <= 160 && Ise % 8 == 0, "mono_ffn: shared-expert size");
  TORCH_CHECK(H % 128 == 0 && I % 64 == 0 && group % 8 == 0,
              "mono_ffn: alignment");
  TORCH_CHECK(workspace.numel() * workspace.element_size() >=
                  mono_ffn_rdna2_workspace_bytes(H, R, ND, E, Ise, I),
              "mono_ffn: workspace too small");
  const at::cuda::OptionalCUDAGuard guard(device_of(res));
  auto* base = reinterpret_cast<char*>(workspace.data_ptr());
  auto al = [](int64_t b) { return (b + 255) / 256 * 256; };
  Params p{};
  p.res = reinterpret_cast<const half*>(res.data_ptr());
  p.blk = reinterpret_cast<const half*>(blk.data_ptr());
  p.inj = reinterpret_cast<const half*>(inj.data_ptr());
  p.res_out = reinterpret_cast<half*>(res_out.data_ptr());
  p.inj_out = reinterpret_cast<half*>(inj_out.data_ptr());
  p.out = reinterpret_cast<half*>(out.data_ptr());
  p.norm_w = reinterpret_cast<const half*>(norm_w.data_ptr());
  p.wd = reinterpret_cast<const half*>(wd.data_ptr());
  p.wu = reinterpret_cast<const half*>(wu.data_ptr());
  p.wr = reinterpret_cast<const half*>(wr.data_ptr());
  p.se_w1 = reinterpret_cast<const half*>(se_w1.data_ptr());
  p.se_w2 = reinterpret_cast<const half*>(se_w2.data_ptr());
  p.se_wg = reinterpret_cast<const half*>(se_wg.data_ptr());
  p.w13 = reinterpret_cast<const uint32_t*>(w13.data_ptr());
  p.s13 = reinterpret_cast<const half*>(s13.data_ptr());
  p.z13 = reinterpret_cast<const uint32_t*>(z13.data_ptr());
  p.w2 = reinterpret_cast<const uint32_t*>(w2.data_ptr());
  p.s2 = reinterpret_cast<const half*>(s2.data_ptr());
  p.z2 = reinterpret_cast<const uint32_t*>(z2.data_ptr());
  int64_t off = 0;
  auto take = [&](int64_t bytes) { char* q = base + off; off += al(bytes); return q; };
  p.ctrl = reinterpret_cast<Ctrl*>(take(sizeof(Ctrl)));
  p.dai = reinterpret_cast<float*>(take((int64_t)kMaxM * ND * 4));
  p.acc = reinterpret_cast<float*>(take((int64_t)kMaxM * H * 4));
  p.xn = reinterpret_cast<half*>(take((int64_t)kMaxM * HCH * 2));
  p.bin = reinterpret_cast<half*>(take((int64_t)kMaxM * H * 2));
  p.logits = reinterpret_cast<float*>(take((int64_t)kMaxM * E * 4));
  p.se_g = reinterpret_cast<float*>(take((int64_t)kMaxM * 2 * Ise * 4));
  p.se_gate = reinterpret_cast<float*>(take((int64_t)kMaxM * 4));
  p.act = reinterpret_cast<half*>(take((int64_t)kMaxM * kTopK * I * 2));
  p.M = M; p.H = H; p.R = R; p.ND = ND; p.E = E; p.Ise = Ise; p.I = I;
  p.group = group; p.expert_base = (int)expert_base;
  p.zero_offset = (int)zero_offset; p.eps = (float)eps;
  {
    const char* pf = getenv("MONO_FFN_PREFETCH");
    p.prefetch = (pf && pf[0] == '1') ? 1 : 0;
  }
  int khz = 0;
  hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, res.get_device());
  p.timeout_ticks = (unsigned long long)(timeout_ms * (khz > 0 ? khz : 100000));
  auto stream = at::cuda::getCurrentCUDAStream();
  // The grid barrier needs every WG resident. hipOccupancy* counts WGPs on
  // gfx10 and under-reports by 2x (P0 probe), so derive the per-CU limit from
  // the code object: waves per SIMD = 1024 / VGPRs (wave32), 2 SIMDs per CU,
  // LDS 64 KiB per CU; CUs = 2 x the reported WGP count.
  static int cap = 0;
  if (cap == 0) {
    hipFuncAttributes fa{};
    int units = 0;
    hipFuncGetAttributes(&fa, reinterpret_cast<const void*>(mono_ffn_kernel));
    hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount,
                          res.get_device());
    const int vgpr = std::max(8, (fa.numRegs + 7) / 8 * 8);
    const int waves_cu = 2 * std::min(16, 1024 / vgpr);
    const int by_waves = waves_cu / kWaves;
    const int by_lds = fa.sharedSizeBytes > 0 ? (int)(65536 / fa.sharedSizeBytes) : 8;
    cap = std::max(1, std::min(by_waves, by_lds) * 2 * units);
    if (getenv("MONO_FFN_VERBOSE"))
      fprintf(stderr, "mono_ffn: numRegs=%d lds=%zu units=%d -> cap %d WGs\n",
              fa.numRegs, fa.sharedSizeBytes, units, cap);
  }
  const int G = std::min(num_wg > 0 ? (int)num_wg : 72, cap);
  hipLaunchKernelGGL(mono_ffn_kernel, dim3(G), dim3(kThreads), 0, stream, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Sticky error record of a workspace: [code, phase, wg] (int32, CPU tensor).
// WG0 phase timestamps of the last launch (wall-clock ticks, 100 MHz on V620).
torch::Tensor mono_ffn_rdna2_phase_ts(const torch::Tensor& workspace) {
  // ts[8] then wg_done[7][256]
  return workspace.slice(0, 32, 32 + 8 * (8 + 7 * 256)).view(torch::kInt64).to(torch::kCPU);
}

torch::Tensor mono_ffn_rdna2_error(const torch::Tensor& workspace) {
  // Ctrl = {count, gen, err_code, err_phase, err_wg, ...}; workspace is uint8.
  return workspace.slice(0, 8, 24).view(torch::kInt32).to(torch::kCPU);
}
