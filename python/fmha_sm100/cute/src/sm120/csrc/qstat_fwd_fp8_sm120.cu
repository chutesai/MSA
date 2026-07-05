// SPDX-License-Identifier: MIT
// SM120 qstat forward, FP8 score-dots: S = Q@K^T via mma.sync m16n8k32
// e4m3 (2x the bf16 mma rate), matching the repo qstat_fp8 contract:
// Q quantized per-row to e4m3 in-kernel, K pre-quantized e4m3 with per-token
// scales, all scales applied outside the MMA. PV stays bf16 (more precise
// than the Triton fp8 path, which also quantizes P). K ring shrinks to
// 2x16KB; V bf16 32KB; ~65KB shared memory total.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int kDim = 128;
constexpr int kBlkKV = 128;
constexpr int kMRows = 64;
constexpr int kWarps = 4;
constexpr float kFp8Max = 448.0f;

using bf16 = __nv_bfloat16;

// bf16 tiles: XOR swizzle on 16B units within a 256B row.
DEVINL int sw_off(int row, int col_e) {
  const int unit = col_e >> 3;
  return row * kDim + ((unit ^ (row & 7)) << 3) + (col_e & 7);
}

// fp8 tiles: rows are 128 bytes, 16B units.
DEVINL int sw_off8(int row, int col_b) {
  const int unit = col_b >> 4;
  return row * kDim + ((unit ^ (row & 7)) << 4) + (col_b & 15);
}

DEVINL unsigned mask_all() { return 0xffffffffu; }

DEVINL void mma_16x8x16(const unsigned a[4], const unsigned b[2], float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

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

DEVINL void ldmatrix_x2_trans(unsigned frag[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1]) : "r"(addr));
}

DEVINL void cp_async_16(unsigned dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

DEVINL unsigned pack_bf16x2(bf16 lo, bf16 hi) {
  unsigned u;
  __nv_bfloat162 t = __halves2bfloat162(lo, hi);
  memcpy(&u, &t, 4);
  return u;
}

DEVINL unsigned pack_e4m3x4(float x0, float x1, float x2, float x3) {
  const unsigned b0 = __nv_cvt_float_to_fp8(x0, __NV_SATFINITE, __NV_E4M3);
  const unsigned b1 = __nv_cvt_float_to_fp8(x1, __NV_SATFINITE, __NV_E4M3);
  const unsigned b2 = __nv_cvt_float_to_fp8(x2, __NV_SATFINITE, __NV_E4M3);
  const unsigned b3 = __nv_cvt_float_to_fp8(x3, __NV_SATFINITE, __NV_E4M3);
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// Exactly 48KB: staying at or below the default dynamic-smem limit keeps
// this kernel off the cudaFuncSetAttribute opt-in path, which fails with
// invalid-resource-handle in heavyweight multi-fatbin processes (mesh
// forwards). Per-token K scales are read through L2 instead of smem.
struct SharedStorage {
  unsigned char k8[2][kBlkKV * kDim];   // e4m3 K ring: 2x16KB (swizzled bytes)
  unsigned char v8t[kDim * kBlkKV];     // e4m3 V, dim-major rows: 16KB
};

template <int BLOCK_T>
__global__ void __launch_bounds__(kWarps * 32, 2)
qstat_fwd_fp8v2_kernel(
    const bf16* __restrict__ q,
    const unsigned char* __restrict__ k8,   // (total, Hkv, 128) e4m3
    const unsigned char* __restrict__ v8t,  // (Hkv, 128, total) e4m3 dim-major
    const float* __restrict__ kscale,       // (total, Hkv)
    const float* __restrict__ vscale,       // (Hkv, 128) per-channel
    const int* __restrict__ unions,
    const int* __restrict__ counts,
    const unsigned long long* __restrict__ selbits,
    bf16* __restrict__ out,
    float* __restrict__ lse,
    int batch, int seq_len, int ntiles, int heads_q, int heads_kv,
    int u_max, float scale) {
  constexpr int kG = kMRows / BLOCK_T;
  extern __shared__ char smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int pid = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int b = pid / ntiles;
  const int tile = pid % ntiles;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int t0 = tile * BLOCK_T;
  const int row0 = warp * 16 + (lane >> 2);
  const int row1 = row0 + 8;
  const long kv_base = (long)b * seq_len;

  const long tile_idx = ((long)kv_head * batch + b) * ntiles + tile;
  const int cnt = counts[tile_idx];
  const int* union_row = unions + tile_idx * u_max;
  const unsigned long long* sel_row = selbits + tile_idx * u_max;

  // Load this lane's Q slice in the e4m3 A layout, find the per-row absmax
  // across the 4-lane row group, then quantize into packed A fragments.
  // A m16n8k32: a0=(row, k 4c..4c+3), a1=(row+8,..), a2=(row, k+16), a3=(row+8, k+16).
  unsigned a_q[kDim / 32][4];
  float q_scale_r[2];
  {
    const int t_r0 = t0 + (row0 % BLOCK_T);
    const int t_r1 = t0 + (row1 % BLOCK_T);
    const int h_r0 = kv_head * kG + row0 / BLOCK_T;
    const int h_r1 = kv_head * kG + row1 / BLOCK_T;
    const bool v0 = t_r0 < seq_len;
    const bool v1 = t_r1 < seq_len;
    const long base0 = (((long)b * seq_len + (v0 ? t_r0 : 0)) * heads_q + h_r0) * kDim;
    const long base1 = (((long)b * seq_len + (v1 ? t_r1 : 0)) * heads_q + h_r1) * kDim;
    float vals[2][kDim / 32][8];
    float amax[2] = {0.f, 0.f};
    #pragma unroll
    for (int kc = 0; kc < kDim / 32; ++kc) {
      const int c0 = kc * 32 + (lane & 3) * 4;
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        vals[0][kc][e] = v0 ? __bfloat162float(q[base0 + c0 + e]) : 0.f;
        vals[1][kc][e] = v1 ? __bfloat162float(q[base1 + c0 + e]) : 0.f;
        vals[0][kc][4 + e] = v0 ? __bfloat162float(q[base0 + c0 + 16 + e]) : 0.f;
        vals[1][kc][4 + e] = v1 ? __bfloat162float(q[base1 + c0 + 16 + e]) : 0.f;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
          amax[r] = fmaxf(amax[r], fabsf(vals[r][kc][e]));
          amax[r] = fmaxf(amax[r], fabsf(vals[r][kc][4 + e]));
        }
      }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1)
        amax[r] = fmaxf(amax[r], __shfl_xor_sync(mask_all(), amax[r], w));
      amax[r] = fmaxf(amax[r], 1e-6f);
      q_scale_r[r] = amax[r] / kFp8Max;
      const float inv = kFp8Max / amax[r];
      #pragma unroll
      for (int kc = 0; kc < kDim / 32; ++kc) {
        const unsigned lo = pack_e4m3x4(vals[r][kc][0] * inv, vals[r][kc][1] * inv,
                                        vals[r][kc][2] * inv, vals[r][kc][3] * inv);
        const unsigned hi = pack_e4m3x4(vals[r][kc][4] * inv, vals[r][kc][5] * inv,
                                        vals[r][kc][6] * inv, vals[r][kc][7] * inv);
        a_q[kc][r] = lo;       // a0 (r=0) / a1 (r=1): k 4c..4c+3
        a_q[kc][2 + r] = hi;   // a2 / a3: k+16
      }
    }
  }

  float m_i[2] = {-1e30f, -1e30f};
  float l_i[2] = {0.f, 0.f};
  float o_acc[2][kDim / 4];
  #pragma unroll
  for (int r = 0; r < 2; ++r)
    #pragma unroll
    for (int i = 0; i < kDim / 4; ++i) o_acc[r][i] = 0.f;

  auto issue_k = [&](int blk, int buf) {
    // 128 tokens x 128 bytes = 8 iterations of 16B chunks over 128 threads.
    #pragma unroll
    for (int it = 0; it < (kBlkKV * kDim / 16) / (kWarps * 32); ++it) {
      const int idx = it * kWarps * 32 + threadIdx.x;
      const int tok = idx >> 3;
      const int cb = (idx & 7) << 4;
      const long pos = (long)blk * kBlkKV + tok;
      const bool ok = pos < seq_len;
      const long g = ((kv_base + (ok ? pos : 0)) * heads_kv + kv_head) * kDim + cb;
      cp_async_16(smem_u32(&smem.k8[buf][sw_off8(tok, cb)]), k8 + g, ok);
    }
  };
  const long total_kv = (long)batch * seq_len;
  auto issue_v = [&](int blk) {
    // Transposed V: 16B chunk = 16 consecutive tokens at one dim.
    #pragma unroll
    for (int it = 0; it < (kDim * kBlkKV / 16) / (kWarps * 32); ++it) {
      const int idx = it * kWarps * 32 + threadIdx.x;
      const int dim = idx >> 3;
      const int t16 = (idx & 7) << 4;
      const long pos = (long)blk * kBlkKV + t16;
      const bool ok = pos + 15 < seq_len;  // seq_len % 16 == 0 (checked host-side)
      const long g = ((long)kv_head * kDim + dim) * total_kv + kv_base + (ok ? pos : 0);
      cp_async_16(smem_u32(&smem.v8t[sw_off8(dim, t16)]), v8t + g, ok);
    }
  };

  int buf = 0;
  if (cnt > 0) {
    issue_k(union_row[0], 0);
    cp_async_commit();
  }

  for (int u = 0; u < cnt; ++u) {
    const int blk = union_row[u];
    const unsigned long long sel = sel_row[u];

    issue_v(blk);
    cp_async_commit();
    if (u + 1 < cnt) {
      issue_k(union_row[u + 1], buf ^ 1);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    const unsigned char* kb = smem.k8[buf];

    // S = Q @ K^T in e4m3: 4 k-chunks of 32 dims (vs 8 chunks of 16 in bf16).
    float s_acc[kBlkKV / 8][4];
    #pragma unroll
    for (int n = 0; n < kBlkKV / 8; ++n) {
      s_acc[n][0] = s_acc[n][1] = s_acc[n][2] = s_acc[n][3] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kDim / 32; ++kc) {
        // B e4m3 (k=32 dims, n=8 tokens): lanes 0-7 -> token rows, byte col
        // kc*32; lanes 8-15 -> same rows, +16 bytes (k+16). b16-view tiles.
        const int lrow = n * 8 + (lane & 7);
        const int lcol = kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned b_frag[2];
        ldmatrix_x2(b_frag, smem_u32(&kb[sw_off8(lrow, lcol)]));
        mma_16x8x32_e4m3(a_q[kc], b_frag, s_acc[n]);
      }
    }

    const bool sel0 = (sel >> (row0 % BLOCK_T)) & 1ull;
    const bool sel1 = (sel >> (row1 % BLOCK_T)) & 1ull;

    float row_max[2] = {-1e30f, -1e30f};
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int m = r == 0 ? row0 : row1;
      const int t = t0 + (m % BLOCK_T);
      const bool tv = (t < seq_len) && (r == 0 ? sel0 : sel1);
      const float qs = q_scale_r[r] * scale;
      #pragma unroll
      for (int n = 0; n < kBlkKV / 8; ++n) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int tok = n * 8 + (lane & 3) * 2 + e;
          const long pos = (long)blk * kBlkKV + tok;
          float& sv = s_acc[n][r * 2 + e];
          const bool vis = tv && pos < seq_len && pos <= t;
          const float ks = vis ? __ldg(&kscale[(kv_base + pos) * heads_kv + kv_head]) : 0.f;
          sv = vis ? sv * qs * ks : -1e30f;
          row_max[r] = fmaxf(row_max[r], sv);
        }
      }
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1)
        row_max[r] = fmaxf(row_max[r], __shfl_xor_sync(mask_all(), row_max[r], w));
    }

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float m_new = fmaxf(m_i[r], row_max[r]);
      const float alpha = __expf(m_i[r] - m_new);
      float p_sum = 0.f;
      #pragma unroll
      for (int n = 0; n < kBlkKV / 8; ++n) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& sv = s_acc[n][r * 2 + e];
          sv = (sv > -1e29f) ? __expf(sv - m_new) : 0.f;
          p_sum += sv;
        }
      }
      #pragma unroll
      for (int w = 2; w >= 1; w >>= 1)
        p_sum += __shfl_xor_sync(mask_all(), p_sum, w);
      l_i[r] = l_i[r] * alpha + p_sum;
      m_i[r] = m_new;
      #pragma unroll
      for (int i = 0; i < kDim / 4; ++i) o_acc[r][i] *= alpha;
    }

    // PV in e4m3: quantize p*448 per lane pair, repack C->A via shfl.
    // Consumer lane L (A=L&3) reg a0 needs tokens 4A..4A+3: the u16 pairs of
    // lanes src=(L&~3)+((A&1)<<1) and src+1, from tile 4kc+(A>>1) (a2: +2).
    unsigned short u16r0[kBlkKV / 8], u16r1[kBlkKV / 8];
    #pragma unroll
    for (int n = 0; n < kBlkKV / 8; ++n) {
      const unsigned b00 = __nv_cvt_float_to_fp8(s_acc[n][0] * kFp8Max, __NV_SATFINITE, __NV_E4M3);
      const unsigned b01 = __nv_cvt_float_to_fp8(s_acc[n][1] * kFp8Max, __NV_SATFINITE, __NV_E4M3);
      const unsigned b10 = __nv_cvt_float_to_fp8(s_acc[n][2] * kFp8Max, __NV_SATFINITE, __NV_E4M3);
      const unsigned b11 = __nv_cvt_float_to_fp8(s_acc[n][3] * kFp8Max, __NV_SATFINITE, __NV_E4M3);
      u16r0[n] = (unsigned short)(b00 | (b01 << 8));
      u16r1[n] = (unsigned short)(b10 | (b11 << 8));
    }
    unsigned a_p[kBlkKV / 32][4];
    {
      const int A = lane & 3;
      const int src = (lane & ~3) + ((A & 1) << 1);
      #pragma unroll
      for (int kc = 0; kc < kBlkKV / 32; ++kc) {
        unsigned lo0[4], hi0[4], lo1[4], hi1[4];
        #pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          lo0[tt] = __shfl_sync(mask_all(), (unsigned)u16r0[4 * kc + tt], src);
          hi0[tt] = __shfl_sync(mask_all(), (unsigned)u16r0[4 * kc + tt], src | 1);
          lo1[tt] = __shfl_sync(mask_all(), (unsigned)u16r1[4 * kc + tt], src);
          hi1[tt] = __shfl_sync(mask_all(), (unsigned)u16r1[4 * kc + tt], src | 1);
        }
        const bool hi_tile = (A >> 1) != 0;
        a_p[kc][0] = (hi_tile ? lo0[1] : lo0[0]) | ((hi_tile ? hi0[1] : hi0[0]) << 16);
        a_p[kc][1] = (hi_tile ? lo1[1] : lo1[0]) | ((hi_tile ? hi1[1] : hi1[0]) << 16);
        a_p[kc][2] = (hi_tile ? lo0[3] : lo0[2]) | ((hi_tile ? hi0[3] : hi0[2]) << 16);
        a_p[kc][3] = (hi_tile ? lo1[3] : lo1[2]) | ((hi_tile ? hi1[3] : hi1[2]) << 16);
      }
    }
    #pragma unroll
    for (int nD = 0; nD < kDim / 8; ++nD) {
      float o_frag[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int kc = 0; kc < kBlkKV / 32; ++kc) {
        // B e4m3 (k=32 tokens, n=8 dims) from dim-major V rows.
        const int vdim = nD * 8 + (lane & 7);
        const int tcol = kc * 32 + ((lane & 8) ? 16 : 0);
        unsigned b_frag[2];
        ldmatrix_x2(b_frag, smem_u32(&smem.v8t[sw_off8(vdim, tcol)]));
        mma_16x8x32_e4m3(a_p[kc], b_frag, o_frag);
      }
      #pragma unroll
      for (int r = 0; r < 2; ++r)
        #pragma unroll
        for (int e = 0; e < 2; ++e)
          o_acc[r][nD * 2 + e] += o_frag[r * 2 + e];
    }
    __syncthreads();
    buf ^= 1;
  }

  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int m = r == 0 ? row0 : row1;
    const int t = t0 + (m % BLOCK_T);
    if (t >= seq_len) continue;
    const int h = kv_head * kG + m / BLOCK_T;
    const long gt = (long)b * seq_len + t;
    const float safe_l = (l_i[r] > 0.f ? l_i[r] : 1.f) * kFp8Max;
    #pragma unroll
    for (int nD = 0; nD < kDim / 8; ++nD) {
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = nD * 8 + (lane & 3) * 2 + e;
        const float vs = vscale[kv_head * kDim + col];
        out[(gt * heads_q + h) * kDim + col] =
            __float2bfloat16(o_acc[r][nD * 2 + e] * vs / safe_l);
      }
    }
    if ((lane & 3) == 0)
      lse[gt * heads_q + h] = l_i[r] > 0.f ? m_i[r] + __logf(l_i[r]) : -INFINITY;
  }
}

}  // namespace

torch::Tensor qstat_fwd_fp8v2(
    torch::Tensor q, torch::Tensor k8, torch::Tensor v8t, torch::Tensor kscale,
    torch::Tensor vscale, torch::Tensor unions, torch::Tensor counts,
    torch::Tensor selbits, torch::Tensor lse_out, int64_t batch,
    int64_t seq_len, int64_t block_t, double scale) {
  const at::cuda::CUDAGuard device_guard{q.device()};
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && q.is_cuda() && q.is_contiguous());
  TORCH_CHECK(k8.dtype() == torch::kUInt8 && k8.is_contiguous());
  TORCH_CHECK(v8t.dtype() == torch::kUInt8 && v8t.is_contiguous());
  TORCH_CHECK(kscale.dtype() == torch::kFloat32 && kscale.is_contiguous());
  TORCH_CHECK(vscale.dtype() == torch::kFloat32 && vscale.is_contiguous());
  TORCH_CHECK(seq_len % 16 == 0, "fp8v2 requires seq_len % 16 == 0");
  const int heads_q = q.size(1);
  const int heads_kv = k8.size(1);
  (void)0;
  const int ntiles = (seq_len + block_t - 1) / block_t;
  const int u_max = unions.size(-1);
  auto out = torch::empty_like(q);
  dim3 grid(batch * ntiles, heads_kv);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(SharedStorage);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(block_t * (heads_q / heads_kv) == kMRows, "block_t * g must be 64");
  #define DISPATCH(BT) \
    if (block_t == BT) { \
      if (smem > 48 * 1024) { \
        cudaFuncSetAttribute(qstat_fwd_fp8v2_kernel<BT>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
      } \
      qstat_fwd_fp8v2_kernel<BT><<<grid, block, smem, stream>>>( \
          reinterpret_cast<const bf16*>(q.data_ptr()), \
          k8.data_ptr<unsigned char>(), \
          v8t.data_ptr<unsigned char>(), \
          kscale.data_ptr<float>(), vscale.data_ptr<float>(), \
          unions.data_ptr<int>(), counts.data_ptr<int>(), \
          reinterpret_cast<const unsigned long long*>(selbits.data_ptr()), \
          reinterpret_cast<bf16*>(out.data_ptr()), lse_out.data_ptr<float>(), \
          batch, seq_len, ntiles, heads_q, heads_kv, u_max, \
          static_cast<float>(scale)); \
      C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
  DISPATCH(4) DISPATCH(8) DISPATCH(16) DISPATCH(32) DISPATCH(64)
  #undef DISPATCH
  TORCH_CHECK(false, "unsupported block_t");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qstat_fwd_fp8v2", &qstat_fwd_fp8v2, "qstat forward, full e4m3 MMA");
}
