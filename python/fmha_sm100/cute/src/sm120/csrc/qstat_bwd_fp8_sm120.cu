// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT
//
// Hand-written SM120 qstat backward with full e4m3 tensor-core math
// (EXPERIMENTAL: gradient quantization — enable with
// FMHA_SM120_QSTAT_GRADS=fp8; adoption should be gated on a loss-level A/B).
// Contains: the dQ kernel (S, dP, dS@K all e4m3; per-row quantization with
// every K/V scale folded outside the MMAs), the dK/dV kernel (quad-parity
// chunk streams, in-smem byte transposes for dim-major operands), and the
// row quantizer that prepares q8 / dO' once per backward. Measured gradient
// deviation vs the bf16 backward: cosine ~0.999, ~4-5% mean relative.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEVINL __device__ __forceinline__

namespace {


constexpr int kDim = 128;
constexpr int kBlkKV = 128;
constexpr int kSubN = 64;
constexpr int kMRows = 64;
constexpr int kWarps = 8;
constexpr float kFp8Max = 448.0f;

using bf16 = __nv_bfloat16;

DEVINL int sw_off8(int row, int col_b) {  // 128-byte rows
  const int unit = col_b >> 4;
  return row * kDim + ((unit ^ (row & 7)) << 4) + (col_b & 15);
}

DEVINL int sw_off8n(int row, int col_b) {  // 64-byte rows (dim-major chunks)
  const int unit = col_b >> 4;
  return row * kMRows + ((unit ^ (row & 3)) << 4) + (col_b & 15);
}

DEVINL unsigned mask_all() { return 0xffffffffu; }

DEVINL void mma_16x8x32_e4m3(const unsigned a[4], const unsigned b[2], float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEVINL unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

DEVINL void ldmatrix_x2(unsigned frag[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1]) : "r"(addr));
}

DEVINL void ldmatrix_x4(unsigned frag[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

DEVINL void cp_async_16(unsigned dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

DEVINL unsigned prmt(unsigned a, unsigned b, unsigned sel) {
  unsigned d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

DEVINL unsigned pack_e4m3x4(float x0, float x1, float x2, float x3) {
  const unsigned b0 = __nv_cvt_float_to_fp8(x0, __NV_SATFINITE, __NV_E4M3);
  const unsigned b1 = __nv_cvt_float_to_fp8(x1, __NV_SATFINITE, __NV_E4M3);
  const unsigned b2 = __nv_cvt_float_to_fp8(x2, __NV_SATFINITE, __NV_E4M3);
  const unsigned b3 = __nv_cvt_float_to_fp8(x3, __NV_SATFINITE, __NV_E4M3);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

struct SharedStorageDkdv {
  unsigned char k8[kSubN * kDim];        // 8KB, token-major
  unsigned char v8[kSubN * kDim];        // 8KB, token-major
  unsigned char q8[2][kMRows * kDim];    // per-quad chunk, token-major, 8KB
  unsigned char do8[2][kMRows * kDim];
  unsigned char q8t[2][kDim * kMRows];   // dim-major (64-byte rows), 8KB
  unsigned char do8t[2][kDim * kMRows];
  float lse[2][kMRows];
  float dl[2][kMRows];
  float qsc[2][kMRows];
  float dosc[2][kMRows];
  int qloc[2][kMRows];
  float ksc[kSubN];
};

template <int BLOCK_TQ>
__global__ void __launch_bounds__(kWarps * 32, 1)
qstat_dkdv_fp8_kernel(
    const unsigned char* __restrict__ q8g,    // (total, Hq, 128) e4m3
    const float* __restrict__ qscg,           // (total, Hq)
    const unsigned char* __restrict__ do8g,   // (total, Hq, 128) e4m3 (vs-folded)
    const float* __restrict__ doscg,          // (total, Hq)
    const unsigned char* __restrict__ k8g,    // (total, Hkv, 128)
    const unsigned char* __restrict__ v8g,
    const float* __restrict__ kscale,         // (total, Hkv)
    const float* __restrict__ vscale,         // (Hkv, 128)
    const float* __restrict__ lse_in,
    const float* __restrict__ delta_in,
    const int* __restrict__ k2q_row_ptr,
    const int* __restrict__ k2q_q_indices,
    const int* __restrict__ row_batch,
    const int* __restrict__ row_kv_block,
    void* __restrict__ dk_out,
    void* __restrict__ dv_out,
    long grad_split_stride,
    int total_q, int total_rows, int seq_len, int heads_q, int heads_kv,
    int topk, int nsplit, int out_f32, float scale) {
  constexpr int kG = kMRows / BLOCK_TQ;
  extern __shared__ char smem_raw[];
  SharedStorageDkdv& smem = *reinterpret_cast<SharedStorageDkdv*>(smem_raw);

  const int pid = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int nsub = kBlkKV / kSubN;
  const int row = pid / nsub;
  const int sub = pid % nsub;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int quad = warp >> 2;
  const int wtok = (warp & 3) * 16;

  const int b = row_batch[row];
  const int kv_block = row_kv_block[row];
  const long kv_base = (long)b * seq_len;
  const int pos0 = kv_block * kBlkKV + sub * kSubN;

  const int row_start = k2q_row_ptr[(long)kv_head * (total_rows + 1) + row];
  const int row_end = k2q_row_ptr[(long)kv_head * (total_rows + 1) + row + 1];
  const int row_count = row_end - row_start;

  int chunk_lo = 0, chunk_hi = row_count;
  if (nsplit > 1) {
    const int span = ((row_count + BLOCK_TQ - 1) / BLOCK_TQ + nsplit - 1) / nsplit * BLOCK_TQ;
    chunk_lo = split_idx * span;
    chunk_hi = min(row_count, chunk_lo + span);
  }

  // Stage K8/V8 sub + per-token k scales once.
  for (int idx = threadIdx.x; idx < kSubN * (kDim / 16); idx += kWarps * 32) {
    const int tok = idx >> 3;
    const int cb = (idx & 7) << 4;
    const long pos = pos0 + tok;
    const bool ok = pos < seq_len;
    const long g = ((kv_base + (ok ? pos : 0)) * heads_kv + kv_head) * kDim + cb;
    cp_async_16(smem_u32(&smem.k8[sw_off8(tok, cb)]), k8g + g, ok);
    cp_async_16(smem_u32(&smem.v8[sw_off8(tok, cb)]), v8g + g, ok);
  }
  if (threadIdx.x < kSubN) {
    const long pos = pos0 + threadIdx.x;
    smem.ksc[threadIdx.x] =
        (pos < seq_len) ? kscale[(kv_base + pos) * heads_kv + kv_head] : 0.f;
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // K/V A-fragments (e4m3): this warp's 16 tokens x 128 dims. 16 regs each.
  unsigned a_k[kDim / 32][4], a_v[kDim / 32][4];
  #pragma unroll
  for (int kc = 0; kc < kDim / 32; ++kc) {
    const int trow = wtok + (lane & 15);
    const int tcol = kc * 32 + ((lane & 16) ? 16 : 0);
    ldmatrix_x4(a_k[kc], smem_u32(&smem.k8[sw_off8(trow, tcol)]));
    ldmatrix_x4(a_v[kc], smem_u32(&smem.v8[sw_off8(trow, tcol)]));
  }
  const float ks_r[2] = {smem.ksc[wtok + (lane >> 2)],
                         smem.ksc[wtok + (lane >> 2) + 8]};

  float dk_acc[kDim / 2], dv_acc[kDim / 2];
  #pragma unroll
  for (int i = 0; i < kDim / 2; ++i) { dk_acc[i] = 0.f; dv_acc[i] = 0.f; }

  const int chunks_total = (chunk_hi - chunk_lo + BLOCK_TQ - 1) / BLOCK_TQ;
  const long qidx_base = (long)kv_head * ((long)total_q * topk);

  for (int c = quad; c < chunks_total; c += 2) {
    const int chunk = chunk_lo + c * BLOCK_TQ;
    const int qt = threadIdx.x & 127;
    if (qt < BLOCK_TQ) {
      const int off = row_start + chunk + qt;
      const int ql = (off < row_end) ? k2q_q_indices[qidx_base + off] : -1;
      #pragma unroll
      for (int gI = 0; gI < kG; ++gI) {
        const int m = gI * BLOCK_TQ + qt;
        smem.qloc[quad][m] = ql;
        const bool okr = ql >= 0 && ql < seq_len;
        const long gtq = kv_base + (okr ? ql : 0);
        const int h = kv_head * kG + gI;
        smem.lse[quad][m] = okr ? lse_in[gtq * heads_q + h] : -INFINITY;
        smem.dl[quad][m] = okr ? delta_in[gtq * heads_q + h] : 0.f;
        smem.qsc[quad][m] = okr ? qscg[gtq * heads_q + h] : 0.f;
        smem.dosc[quad][m] = okr ? doscg[gtq * heads_q + h] : 0.f;
      }
    }
    for (int idx = qt; idx < kMRows * (kDim / 16); idx += 128) {
      const int m = idx >> 3;
      const int cb = (idx & 7) << 4;
      const int ql = (row_start + chunk + (m % BLOCK_TQ) < row_end)
          ? k2q_q_indices[qidx_base + row_start + chunk + (m % BLOCK_TQ)] : -1;
      const bool okr = ql >= 0 && ql < seq_len;
      const long gtq = kv_base + (okr ? ql : 0);
      const int h = kv_head * kG + m / BLOCK_TQ;
      const long gaddr = (gtq * heads_q + h) * kDim + cb;
      cp_async_16(smem_u32(&smem.q8[quad][sw_off8(m, cb)]), q8g + gaddr, okr);
      cp_async_16(smem_u32(&smem.do8[quad][sw_off8(m, cb)]), do8g + gaddr, okr);
    }
    cp_async_commit();
    cp_async_wait<0>();
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + quad));

    // Byte-transpose q8/do8 -> dim-major (4x4 prmt tiles). Each quad thread
    // handles 4 tiles of its chunk.
    for (int t4 = qt; t4 < (kMRows / 4) * (kDim / 4); t4 += 128) {
      const int m0 = (t4 % (kMRows / 4)) * 4;
      const int d0 = (t4 / (kMRows / 4)) * 4;
      #pragma unroll
      for (int which = 0; which < 2; ++which) {
        const unsigned char* srcb = which ? smem.do8[quad] : smem.q8[quad];
        unsigned char* dstb = which ? smem.do8t[quad] : smem.q8t[quad];
        unsigned r[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          r[i] = *reinterpret_cast<const unsigned*>(&srcb[sw_off8(m0 + i, d0)]);
        const unsigned x0 = prmt(r[0], r[1], 0x5140);
        const unsigned x1 = prmt(r[0], r[1], 0x7362);
        const unsigned y0 = prmt(r[2], r[3], 0x5140);
        const unsigned y1 = prmt(r[2], r[3], 0x7362);
        unsigned o[4];
        o[0] = prmt(x0, y0, 0x5410);
        o[1] = prmt(x0, y0, 0x7632);
        o[2] = prmt(x1, y1, 0x5410);
        o[3] = prmt(x1, y1, 0x7632);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          *reinterpret_cast<unsigned*>(&dstb[sw_off8n(d0 + i, m0)]) = o[i];
      }
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + quad));

    // sT / dpT (16 tok x 64 rows), e4m3 with 2 k-steps of 32 dims... 4 steps.
    float sT[kMRows / 8][4], dpT[kMRows / 8][4];
    #pragma unroll
    for (int n = 0; n < kMRows / 8; ++n) {
      sT[n][0] = sT[n][1] = sT[n][2] = sT[n][3] = 0.f;
      dpT[n][0] = dpT[n][1] = dpT[n][2] = dpT[n][3] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kDim / 32; ++kc) {
        const int lrow = n * 8 + (lane & 7);
        const int lcol = kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned bq[2], bd[2];
        ldmatrix_x2(bq, smem_u32(&smem.q8[quad][sw_off8(lrow, lcol)]));
        ldmatrix_x2(bd, smem_u32(&smem.do8[quad][sw_off8(lrow, lcol)]));
        mma_16x8x32_e4m3(a_k[kc], bq, sT[n]);
        mma_16x8x32_e4m3(a_v[kc], bd, dpT[n]);
      }
    }

    // pT' and dsT' with all scales folded; per-token-row amax for quant.
    float ds_amax[2] = {0.f, 0.f}, p_amax[2] = {0.f, 0.f};
    #pragma unroll
    for (int n = 0; n < kMRows / 8; ++n) {
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int mrow = n * 8 + (lane & 3) * 2 + e;
        const float lse_r = smem.lse[quad][mrow];
        const float dl_r = smem.dl[quad][mrow];
        const float qs_m = smem.qsc[quad][mrow];
        const float dos_m = smem.dosc[quad][mrow];
        const int ql = smem.qloc[quad][mrow];
        // Sentinel-aware (-30000 finite lse for empty-selection rows, bb8fc8b).
        const bool fin = lse_r > -1e4f && ql >= 0;
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int tok = wtok + (lane >> 2) + rr * 8;
          const long pos = pos0 + tok;
          float& sv = sT[n][rr * 2 + e];
          float& dv_ = dpT[n][rr * 2 + e];
          const bool vis = fin && pos < seq_len && pos <= ql;
          const float p = vis
              ? __expf(sv * (ks_r[rr] * qs_m * scale) - (fin ? lse_r : 0.f)) : 0.f;
          const float dp_real = dv_ * dos_m;
          sv = p * dos_m;                       // pT'
          dv_ = p * (dp_real - dl_r) * qs_m;    // dsT'
          p_amax[rr] = fmaxf(p_amax[rr], fabsf(sv));
          ds_amax[rr] = fmaxf(ds_amax[rr], fabsf(dv_));
        }
      }
    }
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1) {
        p_amax[rr] = fmaxf(p_amax[rr], __shfl_xor_sync(mask_all(), p_amax[rr], w));
        ds_amax[rr] = fmaxf(ds_amax[rr], __shfl_xor_sync(mask_all(), ds_amax[rr], w));
      }
      p_amax[rr] = fmaxf(p_amax[rr], 1e-20f);
      ds_amax[rr] = fmaxf(ds_amax[rr], 1e-20f);
    }
    const float p_inv[2] = {kFp8Max / p_amax[0], kFp8Max / p_amax[1]};
    const float ds_inv[2] = {kFp8Max / ds_amax[0], kFp8Max / ds_amax[1]};

    // Repack pT'/dsT' C-fragments -> e4m3 A operands (k over 64 chunk rows).
    unsigned short pu0[kMRows / 8], pu1[kMRows / 8], du0[kMRows / 8], du1[kMRows / 8];
    #pragma unroll
    for (int n = 0; n < kMRows / 8; ++n) {
      const unsigned p00 = __nv_cvt_float_to_fp8(sT[n][0] * p_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned p01 = __nv_cvt_float_to_fp8(sT[n][1] * p_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned p10 = __nv_cvt_float_to_fp8(sT[n][2] * p_inv[1], __NV_SATFINITE, __NV_E4M3);
      const unsigned p11 = __nv_cvt_float_to_fp8(sT[n][3] * p_inv[1], __NV_SATFINITE, __NV_E4M3);
      const unsigned d00 = __nv_cvt_float_to_fp8(dpT[n][0] * ds_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned d01 = __nv_cvt_float_to_fp8(dpT[n][1] * ds_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned d10 = __nv_cvt_float_to_fp8(dpT[n][2] * ds_inv[1], __NV_SATFINITE, __NV_E4M3);
      const unsigned d11 = __nv_cvt_float_to_fp8(dpT[n][3] * ds_inv[1], __NV_SATFINITE, __NV_E4M3);
      pu0[n] = (unsigned short)(p00 | (p01 << 8));
      pu1[n] = (unsigned short)(p10 | (p11 << 8));
      du0[n] = (unsigned short)(d00 | (d01 << 8));
      du1[n] = (unsigned short)(d10 | (d11 << 8));
    }
    unsigned a_p[kMRows / 32][4], a_ds[kMRows / 32][4];
    {
      const int A = lane & 3;
      const int src = (lane & ~3) + ((A & 1) << 1);
      #pragma unroll
      for (int kc = 0; kc < kMRows / 32; ++kc) {
        unsigned plo[4], phi[4], dlo[4], dhi[4];
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          plo[tt] = __shfl_sync(mask_all(), (unsigned)pu0[4 * kc + tt], src);
          phi[tt] = __shfl_sync(mask_all(), (unsigned)pu0[4 * kc + tt], src | 1);
          dlo[tt] = __shfl_sync(mask_all(), (unsigned)du0[4 * kc + tt], src);
          dhi[tt] = __shfl_sync(mask_all(), (unsigned)du0[4 * kc + tt], src | 1);
        }
        unsigned plo1[4], phi1[4], dlo1[4], dhi1[4];
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          plo1[tt] = __shfl_sync(mask_all(), (unsigned)pu1[4 * kc + tt], src);
          phi1[tt] = __shfl_sync(mask_all(), (unsigned)pu1[4 * kc + tt], src | 1);
          dlo1[tt] = __shfl_sync(mask_all(), (unsigned)du1[4 * kc + tt], src);
          dhi1[tt] = __shfl_sync(mask_all(), (unsigned)du1[4 * kc + tt], src | 1);
        }
        const bool ht = (A >> 1) != 0;
        a_p[kc][0] = (ht ? plo[1] : plo[0]) | ((ht ? phi[1] : phi[0]) << 16);
        a_p[kc][1] = (ht ? plo1[1] : plo1[0]) | ((ht ? phi1[1] : phi1[0]) << 16);
        a_p[kc][2] = (ht ? plo[3] : plo[2]) | ((ht ? phi[3] : phi[2]) << 16);
        a_p[kc][3] = (ht ? plo1[3] : plo1[2]) | ((ht ? phi1[3] : phi1[2]) << 16);
        a_ds[kc][0] = (ht ? dlo[1] : dlo[0]) | ((ht ? dhi[1] : dhi[0]) << 16);
        a_ds[kc][1] = (ht ? dlo1[1] : dlo1[0]) | ((ht ? dhi1[1] : dhi1[0]) << 16);
        a_ds[kc][2] = (ht ? dlo[3] : dlo[2]) | ((ht ? dhi[3] : dhi[2]) << 16);
        a_ds[kc][3] = (ht ? dlo1[3] : dlo1[2]) | ((ht ? dhi1[3] : dhi1[2]) << 16);
      }
    }

    // dk += dsT'8 @ q8t ; dv += pT'8 @ do8t (k = 64 chunk rows, 2 e4m3 steps).
    #pragma unroll
    for (int nD = 0; nD < kDim / 8; ++nD) {
      float acck[4] = {0.f, 0.f, 0.f, 0.f};
      float accv[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int kc = 0; kc < kMRows / 32; ++kc) {
        const int drow = nD * 8 + (lane & 7);
        const int mcol = kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned bq[2], bd[2];
        ldmatrix_x2(bq, smem_u32(&smem.q8t[quad][sw_off8n(drow, mcol)]));
        ldmatrix_x2(bd, smem_u32(&smem.do8t[quad][sw_off8n(drow, mcol)]));
        mma_16x8x32_e4m3(a_ds[kc], bq, acck);
        mma_16x8x32_e4m3(a_p[kc], bd, accv);
      }
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          dk_acc[nD * 4 + rr * 2 + e] += acck[rr * 2 + e] * (ds_amax[rr] / kFp8Max);
          dv_acc[nD * 4 + rr * 2 + e] += accv[rr * 2 + e] * (p_amax[rr] / kFp8Max);
        }
      }
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + quad));
  }

  // Combine quad partials via smem (reuse chunk staging area), then store.
  float* comb = reinterpret_cast<float*>(&smem.q8[0][0]);  // 64x128 f32 = 32KB
  __syncthreads();
  #pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const float* acc = pass == 0 ? dk_acc : dv_acc;
    if (quad == 1) {
      #pragma unroll
      for (int nD = 0; nD < kDim / 8; ++nD)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tok = wtok + (lane >> 2) + rr * 8;
            const int col = nD * 8 + (lane & 3) * 2 + e;
            comb[tok * kDim + col] = acc[nD * 4 + rr * 2 + e];
          }
    }
    __syncthreads();
    if (quad == 0) {
      #pragma unroll
      for (int nD = 0; nD < kDim / 8; ++nD)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int tok = wtok + (lane >> 2) + rr * 8;
            const int col = nD * 8 + (lane & 3) * 2 + e;
            const long pos = pos0 + tok;
            if (pos >= seq_len) continue;
            float tot = acc[nD * 4 + rr * 2 + e] + comb[tok * kDim + col];
            if (pass == 0) tot *= scale;
            else tot /= vscale[kv_head * kDim + col];
            const long oaddr = split_idx * grad_split_stride +
                ((kv_base + pos) * heads_kv + kv_head) * kDim + col;
            if (pass == 0) {
              if (out_f32) reinterpret_cast<float*>(dk_out)[oaddr] = tot;
              else reinterpret_cast<bf16*>(dk_out)[oaddr] = __float2bfloat16(tot);
            } else {
              if (out_f32) reinterpret_cast<float*>(dv_out)[oaddr] = tot;
              else reinterpret_cast<bf16*>(dv_out)[oaddr] = __float2bfloat16(tot);
            }
          }
    }
    __syncthreads();
  }
}


// Exactly 48KB (see fwd fp8): no cudaFuncSetAttribute opt-in required.
struct SharedStorageDq {
  unsigned char k8[kBlkKV * kDim];    // token-major (S)
  unsigned char k8t[kDim * kBlkKV];   // dim-major (dsK)
  unsigned char v8[kBlkKV * kDim];    // token-major (dp)
};

// Round-trip a float through e4m3 (the exact value the MMA consumes).
__device__ __forceinline__ float e4m3_rt(float x) {
  const __nv_fp8_storage_t b8 =
      __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(b8, __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&hr));
}

template <int BLOCK_T>
__global__ void __launch_bounds__(kWarps * 32, 1)
qstat_dq_fp8_kernel(
    const bf16* __restrict__ q,
    const unsigned char* __restrict__ k8g,   // (total, Hkv, 128)
    const unsigned char* __restrict__ k8tg,  // (Hkv, 128, total)
    const unsigned char* __restrict__ v8g,   // (total, Hkv, 128)
    const float* __restrict__ kscale,        // (total, Hkv)
    const float* __restrict__ vscale,        // (Hkv, 128)
    const bf16* __restrict__ dout,
    const bf16* __restrict__ outp,
    const float* __restrict__ lse_in,
    const int* __restrict__ unions,
    const int* __restrict__ counts,
    const unsigned long long* __restrict__ selbits,
    bf16* __restrict__ dq,
    float* __restrict__ delta_out,
    float* __restrict__ delta_q_out,
    int batch, int seq_len, int ntiles, int heads_q, int heads_kv,
    int u_max, float scale) {
  constexpr int kG = kMRows / BLOCK_T;
  extern __shared__ char smem_raw[];
  SharedStorageDq& smem = *reinterpret_cast<SharedStorageDq*>(smem_raw);

  const int pid = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = pid / ntiles;
  const int tile = pid % ntiles;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int strip = warp & 3;
  const int half = warp >> 2;
  const int t0 = tile * BLOCK_T;
  const int row0 = strip * 16 + (lane >> 2);
  const int row1 = row0 + 8;
  const long kv_base = (long)b * seq_len;
  const long total_kv = (long)batch * seq_len;

  const long tile_idx = ((long)kv_head * batch + b) * ntiles + tile;
  const int cnt = counts[tile_idx];
  const int* union_row = unions + tile_idx * u_max;
  const unsigned long long* sel_row = selbits + tile_idx * u_max;

  int t_r[2], h_r[2];
  bool valid_r[2];
  long base_r[2];
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int m = r == 0 ? row0 : row1;
    t_r[r] = t0 + (m % BLOCK_T);
    h_r[r] = kv_head * kG + m / BLOCK_T;
    valid_r[r] = t_r[r] < seq_len;
    base_r[r] = (((long)b * seq_len + (valid_r[r] ? t_r[r] : 0)) * heads_q + h_r[r]) * kDim;
  }

  // Quantize Q per-row and dO' = dO * v_scale[channel] per-row into e4m3 A
  // fragments; compute delta = sum(dout*out) inline (raw dO, for the Triton
  // dK/dV consumer) AND delta_q = sum(dequant(e4m3(dO'))*out) — the delta that
  // is self-consistent with the quantized dO' the dp MMA actually consumes.
  // Using raw delta against quantized dp breaks the softmax-gradient shift
  // invariance dS = P*(dP - delta): the quantization noise's P-weighted row
  // mean survives as a deterministic, step-correlated rank-one bias on dq
  // (observed as a val-loss flatline from sparse activation). delta_q restores
  // the identity exactly under the quantized gradient field.
  unsigned a_q[kDim / 32][4], a_do[kDim / 32][4];
  float q_sc[2], do_sc[2], dl[2] = {0.f, 0.f}, dl_q[2] = {0.f, 0.f};
  {
    float qv[2][kDim / 32][8], dv[2][kDim / 32][8];
    float qa[2] = {0.f, 0.f}, da[2] = {0.f, 0.f};
    #pragma unroll
    for (int kc = 0; kc < kDim / 32; ++kc) {
      const int c0 = kc * 32 + (lane & 3) * 4;
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int c = c0 + (e >> 2) * 16 + (e & 3);
        const float vs = vscale[kv_head * kDim + c];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float qx = valid_r[r] ? __bfloat162float(q[base_r[r] + c]) : 0.f;
          const float dx = valid_r[r] ? __bfloat162float(dout[base_r[r] + c]) : 0.f;
          const float ox = valid_r[r] ? __bfloat162float(outp[base_r[r] + c]) : 0.f;
          dl[r] += dx * ox;
          qv[r][kc][e] = qx;
          dv[r][kc][e] = dx * vs;
          qa[r] = fmaxf(qa[r], fabsf(qx));
          da[r] = fmaxf(da[r], fabsf(dx * vs));
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1) {
        qa[r] = fmaxf(qa[r], __shfl_xor_sync(mask_all(), qa[r], w));
        da[r] = fmaxf(da[r], __shfl_xor_sync(mask_all(), da[r], w));
        dl[r] += __shfl_xor_sync(mask_all(), dl[r], w);
      }
      qa[r] = fmaxf(qa[r], 1e-6f);
      da[r] = fmaxf(da[r], 1e-6f);
      q_sc[r] = qa[r] / kFp8Max;
      do_sc[r] = da[r] / kFp8Max;
      const float qi = kFp8Max / qa[r], di = kFp8Max / da[r];
      #pragma unroll
      for (int kc = 0; kc < kDim / 32; ++kc) {
        a_q[kc][r] = pack_e4m3x4(qv[r][kc][0] * qi, qv[r][kc][1] * qi,
                                 qv[r][kc][2] * qi, qv[r][kc][3] * qi);
        a_q[kc][2 + r] = pack_e4m3x4(qv[r][kc][4] * qi, qv[r][kc][5] * qi,
                                     qv[r][kc][6] * qi, qv[r][kc][7] * qi);
        a_do[kc][r] = pack_e4m3x4(dv[r][kc][0] * di, dv[r][kc][1] * di,
                                  dv[r][kc][2] * di, dv[r][kc][3] * di);
        a_do[kc][2 + r] = pack_e4m3x4(dv[r][kc][4] * di, dv[r][kc][5] * di,
                                      dv[r][kc][6] * di, dv[r][kc][7] * di);
      }
      // delta_q: same channel walk with the row scale now known.  dq~O_c =
      // rt(dv*di)/(di*vs) is exactly the raw-dO-space value the dp MMA sees;
      // outp/vscale re-reads are one-time L2 hits (dwarfed by the block loop).
      if (valid_r[r]) {
        #pragma unroll
        for (int kc = 0; kc < kDim / 32; ++kc) {
          const int c0 = kc * 32 + (lane & 3) * 4;
          #pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int c = c0 + (e >> 2) * 16 + (e & 3);
            const float vs = vscale[kv_head * kDim + c];
            const float ox = __bfloat162float(outp[base_r[r] + c]);
            dl_q[r] += e4m3_rt(dv[r][kc][e] * di) / (di * vs) * ox;
          }
        }
      }
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1)
        dl_q[r] += __shfl_xor_sync(mask_all(), dl_q[r], w);
    }
  }

  float lse_r[2];
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    lse_r[r] = valid_r[r] ? lse_in[((long)b * seq_len + t_r[r]) * heads_q + h_r[r]]
                          : -INFINITY;
    if (valid_r[r] && (lane & 3) == 0 && half == 0) {
      delta_out[((long)b * seq_len + t_r[r]) * heads_q + h_r[r]] = dl[r];
      delta_q_out[((long)b * seq_len + t_r[r]) * heads_q + h_r[r]] = dl_q[r];
    }
  }
  // Sentinel-aware: empty-selection rows carry the finite -30000 lse sentinel
  // (bb8fc8b), not -inf.
  const bool fin_r[2] = {lse_r[0] > -1e4f, lse_r[1] > -1e4f};
  const float lse_safe[2] = {fin_r[0] ? lse_r[0] : 0.f, fin_r[1] ? lse_r[1] : 0.f};

  float dq_acc[2][kDim / 4];
  #pragma unroll
  for (int r = 0; r < 2; ++r)
    #pragma unroll
    for (int i = 0; i < kDim / 4; ++i) dq_acc[r][i] = 0.f;

  auto issue_kv = [&](int blk) {
    #pragma unroll
    for (int it = 0; it < (kBlkKV * kDim / 16) / (kWarps * 32); ++it) {
      const int idx = it * kWarps * 32 + threadIdx.x;
      const int tok = idx >> 3;
      const int cb = (idx & 7) << 4;
      const long pos = (long)blk * kBlkKV + tok;
      const bool ok = pos < seq_len;
      const long g = ((kv_base + (ok ? pos : 0)) * heads_kv + kv_head) * kDim + cb;
      cp_async_16(smem_u32(&smem.k8[sw_off8(tok, cb)]), k8g + g, ok);
      cp_async_16(smem_u32(&smem.v8[sw_off8(tok, cb)]), v8g + g, ok);
      // dim-major K: same (row, 16B-chunk) indices, 16B = 16 tokens at one
      // dim (seq % 16 == 0 host-checked).
      const long post = (long)blk * kBlkKV + cb;
      const bool okt = post + 15 < seq_len;
      const long gt = ((long)kv_head * kDim + tok) * total_kv + kv_base + (okt ? post : 0);
      cp_async_16(smem_u32(&smem.k8t[sw_off8(tok, cb)]), k8tg + gt, okt);
    }
  };

  for (int u = 0; u < cnt; ++u) {
    const int blk = union_row[u];
    const unsigned long long sel = sel_row[u];

    __syncthreads();  // protect smem reuse from previous iteration
    issue_kv(blk);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // s over this warp's 64-token half (8 n-tiles), e4m3.
    float s_acc[kBlkKV / 16][4], dp_acc[kBlkKV / 16][4];
    #pragma unroll
    for (int n = 0; n < kBlkKV / 16; ++n) {
      const int n_g = half * (kBlkKV / 16) + n;
      s_acc[n][0] = s_acc[n][1] = s_acc[n][2] = s_acc[n][3] = 0.f;
      dp_acc[n][0] = dp_acc[n][1] = dp_acc[n][2] = dp_acc[n][3] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kDim / 32; ++kc) {
        const int lrow = n_g * 8 + (lane & 7);
        const int lcol = kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned bk[2], bv[2];
        ldmatrix_x2(bk, smem_u32(&smem.k8[sw_off8(lrow, lcol)]));
        ldmatrix_x2(bv, smem_u32(&smem.v8[sw_off8(lrow, lcol)]));
        mma_16x8x32_e4m3(a_q[kc], bk, s_acc[n]);
        mma_16x8x32_e4m3(a_do[kc], bv, dp_acc[n]);
      }
    }

    const bool sel_r[2] = {bool((sel >> (row0 % BLOCK_T)) & 1ull),
                           bool((sel >> (row1 % BLOCK_T)) & 1ull)};

    // ds' = p * (dp - delta) * k_scale[token]; track per-row amax for quant.
    float ds_amax[2] = {0.f, 0.f};
    #pragma unroll
    for (int n = 0; n < kBlkKV / 16; ++n) {
      const int n_g = half * (kBlkKV / 16) + n;
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tok = n_g * 8 + (lane & 3) * 2 + e;
          const long pos = (long)blk * kBlkKV + tok;
          float& sv = s_acc[n][r * 2 + e];
          float& dv_ = dp_acc[n][r * 2 + e];
          const bool vis = sel_r[r] && valid_r[r] && fin_r[r] &&
                           pos < seq_len && pos <= t_r[r];
          const float ks = vis ? __ldg(&kscale[(kv_base + pos) * heads_kv + kv_head]) : 0.f;
          const float p = vis ? __expf(sv * (q_sc[r] * scale) * ks - lse_safe[r]) : 0.f;
          const float dp_real = dv_ * do_sc[r];
          dv_ = p * (dp_real - dl_q[r]) * ks;
          ds_amax[r] = fmaxf(ds_amax[r], fabsf(dv_));
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1)
        ds_amax[r] = fmaxf(ds_amax[r], __shfl_xor_sync(mask_all(), ds_amax[r], w));
      ds_amax[r] = fmaxf(ds_amax[r], 1e-20f);
    }
    const float ds_inv[2] = {kFp8Max / ds_amax[0], kFp8Max / ds_amax[1]};

    // Repack ds' C-fragments into e4m3 A operands via shfl (fwd v2 pattern).
    unsigned short u16r0[kBlkKV / 16], u16r1[kBlkKV / 16];
    #pragma unroll
    for (int n = 0; n < kBlkKV / 16; ++n) {
      const unsigned b00 = __nv_cvt_float_to_fp8(dp_acc[n][0] * ds_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned b01 = __nv_cvt_float_to_fp8(dp_acc[n][1] * ds_inv[0], __NV_SATFINITE, __NV_E4M3);
      const unsigned b10 = __nv_cvt_float_to_fp8(dp_acc[n][2] * ds_inv[1], __NV_SATFINITE, __NV_E4M3);
      const unsigned b11 = __nv_cvt_float_to_fp8(dp_acc[n][3] * ds_inv[1], __NV_SATFINITE, __NV_E4M3);
      u16r0[n] = (unsigned short)(b00 | (b01 << 8));
      u16r1[n] = (unsigned short)(b10 | (b11 << 8));
    }
    unsigned a_ds[kBlkKV / 64][4];
    {
      const int A = lane & 3;
      const int src = (lane & ~3) + ((A & 1) << 1);
      #pragma unroll
      for (int kc = 0; kc < kBlkKV / 64; ++kc) {  // 2 k-steps of 32 tokens per half
        unsigned lo0[4], hi0[4], lo1[4], hi1[4];
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          lo0[tt] = __shfl_sync(mask_all(), (unsigned)u16r0[4 * kc + tt], src);
          hi0[tt] = __shfl_sync(mask_all(), (unsigned)u16r0[4 * kc + tt], src | 1);
          lo1[tt] = __shfl_sync(mask_all(), (unsigned)u16r1[4 * kc + tt], src);
          hi1[tt] = __shfl_sync(mask_all(), (unsigned)u16r1[4 * kc + tt], src | 1);
        }
        const bool ht = (A >> 1) != 0;
        a_ds[kc][0] = (ht ? lo0[1] : lo0[0]) | ((ht ? hi0[1] : hi0[0]) << 16);
        a_ds[kc][1] = (ht ? lo1[1] : lo1[0]) | ((ht ? hi1[1] : hi1[0]) << 16);
        a_ds[kc][2] = (ht ? lo0[3] : lo0[2]) | ((ht ? hi0[3] : hi0[2]) << 16);
        a_ds[kc][3] = (ht ? lo1[3] : lo1[2]) | ((ht ? hi1[3] : hi1[2]) << 16);
      }
    }

    // dq_acc += ds8' @ k8t over this half's tokens (k = 64 -> 2 e4m3 steps).
    #pragma unroll
    for (int nD = 0; nD < kDim / 8; ++nD) {
      float o_frag[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int kc = 0; kc < kBlkKV / 64; ++kc) {
        const int vdim = nD * 8 + (lane & 7);
        const int tcol = half * 64 + kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned b_frag[2];
        ldmatrix_x2(b_frag, smem_u32(&smem.k8t[sw_off8(vdim, tcol)]));
        mma_16x8x32_e4m3(a_ds[kc], b_frag, o_frag);
      }
      #pragma unroll
      for (int r = 0; r < 2; ++r)
        #pragma unroll
        for (int e = 0; e < 2; ++e)
          dq_acc[r][nD * 2 + e] += o_frag[r * 2 + e] * (ds_amax[r] / kFp8Max);
    }
  }

  // Cross-half combine via smem, then half 0 scales and stores.
  float* comb = reinterpret_cast<float*>(smem_raw);
  __syncthreads();
  if (half == 1) {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int m = r == 0 ? row0 : row1;
      #pragma unroll
      for (int nD = 0; nD < kDim / 8; ++nD)
        #pragma unroll
        for (int e = 0; e < 2; ++e)
          comb[m * kDim + nD * 8 + (lane & 3) * 2 + e] = dq_acc[r][nD * 2 + e];
    }
  }
  __syncthreads();
  if (half == 0) {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (!valid_r[r]) continue;
      const int m = r == 0 ? row0 : row1;
      const long gt = (long)b * seq_len + t_r[r];
      #pragma unroll
      for (int nD = 0; nD < kDim / 8; ++nD) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = nD * 8 + (lane & 3) * 2 + e;
          const float tot = dq_acc[r][nD * 2 + e] + comb[m * kDim + col];
          dq[(gt * heads_q + h_r[r]) * kDim + col] = __float2bfloat16(tot * scale);
        }
      }
    }
  }
}


// Row quantizer for the fp8 backward: per (token, q-head) row, quantize
// Q to e4m3 and dO' = dO * v_scale[channel of its kv head] to e4m3, with
// per-row amax scales. One 128-thread block per row.
__global__ void qstat_quant_rows_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ dout,
    const float* __restrict__ vscale, unsigned char* __restrict__ q8,
    float* __restrict__ qsc, unsigned char* __restrict__ do8,
    float* __restrict__ dosc, int heads_q, int g) {
  const long row = blockIdx.x;
  const int tid = threadIdx.x;
  const int kvh = (int)(row % heads_q) / g;
  const float qx = __bfloat162float(q[row * kDim + tid]);
  const float dx = __bfloat162float(dout[row * kDim + tid]) * vscale[kvh * kDim + tid];
  __shared__ float red[2][4];
  float qa = fabsf(qx), da = fabsf(dx);
  #pragma unroll
  for (int w = 16; w >= 1; w >>= 1) {
    qa = fmaxf(qa, __shfl_xor_sync(mask_all(), qa, w));
    da = fmaxf(da, __shfl_xor_sync(mask_all(), da, w));
  }
  if ((tid & 31) == 0) { red[0][tid >> 5] = qa; red[1][tid >> 5] = da; }
  __syncthreads();
  qa = fmaxf(fmaxf(red[0][0], red[0][1]), fmaxf(red[0][2], red[0][3]));
  da = fmaxf(fmaxf(red[1][0], red[1][1]), fmaxf(red[1][2], red[1][3]));
  qa = fmaxf(qa, 1e-6f);
  da = fmaxf(da, 1e-6f);
  q8[row * kDim + tid] = __nv_cvt_float_to_fp8(qx * (kFp8Max / qa), __NV_SATFINITE, __NV_E4M3);
  do8[row * kDim + tid] = __nv_cvt_float_to_fp8(dx * (kFp8Max / da), __NV_SATFINITE, __NV_E4M3);
  if (tid == 0) { qsc[row] = qa / kFp8Max; dosc[row] = da / kFp8Max; }
}
}  // namespace


void qstat_dkdv_fp8(
    torch::Tensor q8, torch::Tensor qsc, torch::Tensor do8, torch::Tensor dosc,
    torch::Tensor k8, torch::Tensor v8, torch::Tensor kscale, torch::Tensor vscale,
    torch::Tensor lse, torch::Tensor delta, torch::Tensor k2q_row_ptr,
    torch::Tensor k2q_q_indices, torch::Tensor row_batch, torch::Tensor row_kv_block,
    torch::Tensor dk_out, torch::Tensor dv_out,
    int64_t seq_len, int64_t block_tq, int64_t topk, int64_t nsplit, double scale) {
  const at::cuda::CUDAGuard device_guard{q8.device()};
  const int heads_q = q8.size(1);
  const int heads_kv = k8.size(1);
  const int total_q = q8.size(0);
  const int total_rows = row_batch.size(0);
  const int nsub = kBlkKV / kSubN;
  const bool out_f32 = dk_out.dtype() == torch::kFloat32;
  const long stride = (nsplit > 1) ? (long)total_q * heads_kv * kDim : 0;
  dim3 grid(total_rows * nsub, heads_kv, nsplit);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(SharedStorageDkdv);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(block_tq * (heads_q / heads_kv) == kMRows, "block_tq * g must be 64");
  #define DISPATCH(BT) \
    if (block_tq == BT) { \
      { \
        cudaError_t _fa = cudaFuncSetAttribute(qstat_dkdv_fp8_kernel<BT>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        TORCH_CHECK(_fa == cudaSuccess, \
            "cudaFuncSetAttribute failed (", cudaGetErrorString(_fa), \
            "): known failure mode in heavyweight multi-fatbin processes; " \
            "set FMHA_SM120_QSTAT_GRADS=bf16 to route around this kernel " \
            "(the fp8 backward wrapper normally probes and falls back to " \
            "the Triton dK/dV automatically)."); \
      } \
      qstat_dkdv_fp8_kernel<BT><<<grid, block, smem, stream>>>( \
          q8.data_ptr<unsigned char>(), qsc.data_ptr<float>(), \
          do8.data_ptr<unsigned char>(), dosc.data_ptr<float>(), \
          k8.data_ptr<unsigned char>(), v8.data_ptr<unsigned char>(), \
          kscale.data_ptr<float>(), vscale.data_ptr<float>(), \
          lse.data_ptr<float>(), delta.data_ptr<float>(), \
          k2q_row_ptr.data_ptr<int>(), k2q_q_indices.data_ptr<int>(), \
          row_batch.data_ptr<int>(), row_kv_block.data_ptr<int>(), \
          dk_out.data_ptr(), dv_out.data_ptr(), stride, \
          total_q, total_rows, seq_len, heads_q, heads_kv, \
          (int)topk, (int)nsplit, (int)out_f32, (float)scale); \
      C10_CUDA_KERNEL_LAUNCH_CHECK(); return; }
  DISPATCH(4) DISPATCH(8) DISPATCH(16) DISPATCH(32) DISPATCH(64)
  #undef DISPATCH
  TORCH_CHECK(false, "unsupported block_tq");
}



torch::Tensor qstat_dq_fp8(
    torch::Tensor q, torch::Tensor k8, torch::Tensor k8t, torch::Tensor v8,
    torch::Tensor kscale, torch::Tensor vscale, torch::Tensor dout,
    torch::Tensor outp, torch::Tensor lse, torch::Tensor unions,
    torch::Tensor counts, torch::Tensor selbits, torch::Tensor delta_out,
    torch::Tensor delta_q_out,
    int64_t batch, int64_t seq_len, int64_t block_t, double scale) {
  const at::cuda::CUDAGuard device_guard{q.device()};
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k8.dtype() == torch::kUInt8 && k8t.dtype() == torch::kUInt8);
  TORCH_CHECK(seq_len % 16 == 0, "dq fp8 requires seq_len % 16 == 0");
  const int heads_q = q.size(1);
  const int heads_kv = k8.size(1);
  const int ntiles = (seq_len + block_t - 1) / block_t;
  const int u_max = unions.size(-1);
  auto dq_out = torch::empty_like(q);
  dim3 grid(batch * ntiles, heads_kv);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(SharedStorageDq);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(block_t * (heads_q / heads_kv) == kMRows, "block_t * g must be 64");
  #define DISPATCH(BT) \
    if (block_t == BT) { \
      if (smem > 48 * 1024) { \
        cudaFuncSetAttribute(qstat_dq_fp8_kernel<BT>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      } \
      qstat_dq_fp8_kernel<BT><<<grid, block, smem, stream>>>( \
          reinterpret_cast<const bf16*>(q.data_ptr()), \
          k8.data_ptr<unsigned char>(), k8t.data_ptr<unsigned char>(), \
          v8.data_ptr<unsigned char>(), kscale.data_ptr<float>(), \
          vscale.data_ptr<float>(), \
          reinterpret_cast<const bf16*>(dout.data_ptr()), \
          reinterpret_cast<const bf16*>(outp.data_ptr()), \
          lse.data_ptr<float>(), unions.data_ptr<int>(), counts.data_ptr<int>(), \
          reinterpret_cast<const unsigned long long*>(selbits.data_ptr()), \
          reinterpret_cast<bf16*>(dq_out.data_ptr()), delta_out.data_ptr<float>(), \
          delta_q_out.data_ptr<float>(), \
          batch, seq_len, ntiles, heads_q, heads_kv, u_max, \
          static_cast<float>(scale)); \
      C10_CUDA_KERNEL_LAUNCH_CHECK(); return dq_out; }
  DISPATCH(4) DISPATCH(8) DISPATCH(16) DISPATCH(32) DISPATCH(64)
  #undef DISPATCH
  TORCH_CHECK(false, "unsupported block_t");
}


void qstat_quant_rows(
    torch::Tensor q, torch::Tensor dout, torch::Tensor vscale,
    torch::Tensor q8, torch::Tensor qsc, torch::Tensor do8, torch::Tensor dosc,
    int64_t g) {
  const at::cuda::CUDAGuard device_guard{q.device()};
  const long rows = (long)q.size(0) * q.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  qstat_quant_rows_kernel<<<rows, kDim, 0, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(dout.data_ptr()),
      vscale.data_ptr<float>(), q8.data_ptr<unsigned char>(),
      qsc.data_ptr<float>(), do8.data_ptr<unsigned char>(),
      dosc.data_ptr<float>(), (int)q.size(1), (int)g);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool qstat_dkdv_fp8_supported() {
  // dkdv needs >48KB dynamic smem, hence cudaFuncSetAttribute — which fails
  // with invalid-resource-handle in some heavyweight multi-fatbin processes.
  // Probe once so callers can route dK/dV to the Triton kernel instead.
  cudaError_t e = cudaFuncSetAttribute(
      qstat_dkdv_fp8_kernel<16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)sizeof(SharedStorageDkdv));
  if (e != cudaSuccess) {
    (void)cudaGetLastError();  // clear
    return false;
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qstat_dq_fp8", &qstat_dq_fp8, "qstat dQ backward, full e4m3");
  m.def("qstat_dkdv_fp8", &qstat_dkdv_fp8, "qstat dK/dV backward, full e4m3");
  m.def("qstat_quant_rows", &qstat_quant_rows, "per-row e4m3 quant of Q and dO*vs");
  m.def("qstat_dkdv_fp8_supported", &qstat_dkdv_fp8_supported,
        "probe whether the >48KB dkdv kernel can opt in on this process");
}
