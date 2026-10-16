/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <cstdint>
#include <string>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../torch_utils.h"
#include "../../cuda_compat.h"

namespace {

constexpr int kDimK = 128;
constexpr int kDimV = 128;
// One V row per kRowThreads threads, each holding kKPerThread contiguous K
// elements: a row's dot products reduce over 8 lanes, and state rows move as
// 16-byte vectors. A 256-thread group covers one kChunkV-row chunk; a block
// runs Groups of them at once.
constexpr int kGroupThreads = 256;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowThreads = 8;
constexpr int kKPerThread = kDimK / kRowThreads;
// Pads each thread's K slice in shared memory so the 8 slices of a row sit in
// distinct banks.
constexpr int kSlicePad = kKPerThread + 1;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

__device__ __forceinline__ float row_reduce_sum(float value) {
  // The 8 lanes of one V row are consecutive within the warp.
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& packed, float* out) {
  const __nv_bfloat162* pairs =
      reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 value = __bfloat1622float2(pairs[i]);
    out[2 * i] = value.x;
    out[2 * i + 1] = value.y;
  }
}
__device__ __forceinline__ uint4 pack_bf16x8(const float* in) {
  uint4 packed;
  __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pairs[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
  }
  return packed;
}

__device__ __forceinline__ int row_bin(const int* __restrict__ cu_seqlens,
                                       int request) {
  const int rows = cu_seqlens[request + 1] - cu_seqlens[request];
  return rows < 0 ? 0 : (rows > kMaxMtpTokens ? kMaxMtpTokens + 1 : rows);
}

// The request at `rank` once requests are ordered by row count, most rows
// first and ties by index. Every lane of the calling warp computes it.
__device__ __forceinline__ int request_at_rank(
    const int* __restrict__ cu_seqlens, int num_requests, int rank, int lane) {
  constexpr int kBins = kMaxMtpTokens + 2;
  int counts[kBins];
#pragma unroll
  for (int bin = 0; bin < kBins; ++bin) {
    counts[bin] = 0;
  }
  for (int base = 0; base < num_requests; base += 32) {
    const int request = base + lane;
    const int bin = request < num_requests ? row_bin(cu_seqlens, request) : -1;
#pragma unroll
    for (int b = 0; b < kBins; ++b) {
      counts[b] += __popc(__ballot_sync(0xffffffffu, bin == b));
    }
  }
  int target = 0;
  int offset = rank;
  bool found = false;
#pragma unroll
  for (int bin = kBins - 1; bin >= 0; --bin) {
    if (!found) {
      if (offset < counts[bin]) {
        target = bin;
        found = true;
      } else {
        offset -= counts[bin];
      }
    }
  }
  for (int base = 0; base < num_requests; base += 32) {
    const int request = base + lane;
    const int bin = request < num_requests ? row_bin(cu_seqlens, request) : -1;
    unsigned mask = __ballot_sync(0xffffffffu, bin == target);
    const int matches = __popc(mask);
    if (offset < matches) {
      for (int skip = 0; skip < offset; ++skip) {
        mask &= mask - 1;
      }
      return base + __ffs(mask) - 1;
    }
    offset -= matches;
  }
  return rank;
}

// One thread's row slice as raw 16-byte vectors, loaded ahead of its use.
template <typename StateT>
struct RowSliceVectors {
  static constexpr int kVectors = kKPerThread * sizeof(StateT) / sizeof(uint4);
  uint4 v[kVectors];

  __device__ __forceinline__ void load(const StateT* src) {
#pragma unroll
    for (int i = 0; i < kVectors; ++i) {
      v[i] = reinterpret_cast<const uint4*>(src)[i];
    }
  }
  __device__ __forceinline__ void unpack(float* h) const {
    if constexpr (sizeof(StateT) == 2) {
      unpack_bf16x8(v[0], h);
      unpack_bf16x8(v[1], h + 8);
    } else {
#pragma unroll
      for (int i = 0; i < kVectors; ++i) {
        const float4 value = *reinterpret_cast<const float4*>(&v[i]);
        h[4 * i] = value.x;
        h[4 * i + 1] = value.y;
        h[4 * i + 2] = value.z;
        h[4 * i + 3] = value.w;
      }
    }
  }
};

// The per-row states are written once and read at most once, a step later,
// so they stream past L2 (st.global.cs) instead of evicting reused data.
template <typename StateT>
__device__ __forceinline__ void store_row_slice(StateT* dst, const float* h);
template <>
__device__ __forceinline__ void store_row_slice<__nv_bfloat16>(
    __nv_bfloat16* dst, const float* h) {
  __stcs(reinterpret_cast<uint4*>(dst), pack_bf16x8(h));
  __stcs(reinterpret_cast<uint4*>(dst) + 1, pack_bf16x8(h + 8));
}
template <>
__device__ __forceinline__ void store_row_slice<float>(float* dst,
                                                       const float* h) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __stcs(reinterpret_cast<float4*>(dst) + i,
           make_float4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]));
  }
}

// RowsPerThread: each thread walks that many value rows, kChunkV apart, with the
// same K slice; every shared-memory read of q and k then serves all of them
// (one group only). Each row's arithmetic is the one-row walk's.
template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate,
          int Groups, int RowsPerThread, int MinBlocks>
__global__
__launch_bounds__(kGroupThreads* Groups, MinBlocks) void gdn_decode_post_conv_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides, bool longest_first) {
  constexpr int kThreads = kGroupThreads * Groups;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // longest_first: blockIdx.y is a rank in row count and blockIdx.x the head,
  // so the longest requests start first and short ones fill the last wave.
  const int value_head = longest_first ? blockIdx.x : blockIdx.y;
  const int request =
      longest_first ? request_at_rank(cu_seqlens, gridDim.y, blockIdx.y, lane)
                    : blockIdx.x;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }
  // The index row loads beside the accepted count rather than after it.
  const int slot_value =
      lane < state_indices_width
          ? state_indices[request * state_indices_width + lane]
          : 0;
  const int accepted = num_accepted_tokens[request];
  const bool has_source = accepted > 0 && accepted <= state_indices_width;
  const int source_lane = has_source ? accepted - 1 : 0;
  const int source_slot =
      has_source ? __shfl_sync(0xffffffffu, slot_value, source_lane) : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
          __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ float shared_q[kMaxMtpTokens][kRowThreads * kSlicePad];
  __shared__ float shared_k[kMaxMtpTokens][kRowThreads * kSlicePad];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];
  __shared__ int shared_slots[kMaxMtpTokens];
  if (warp == 0 && lane < kMaxMtpTokens) {
    shared_slots[lane] = slot_value;
  }

  // The first chunk's state and the output gate load under the prologue.
  const int group = tid / kGroupThreads;
  const int group_tid = tid % kGroupThreads;
  const int row = group_tid / kRowThreads;
  const int slice = group_tid % kRowThreads;
  const int64_t head_offset = static_cast<int64_t>(value_head) * kDimV * kDimK;
  const StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      head_offset;
  static_assert(RowsPerThread == 1 || Groups == 1,
                "several rows per thread walk the chunks in one group");
  constexpr int kChunkStep = Groups * RowsPerThread;
  RowSliceVectors<StateT> next_rows[RowsPerThread];
#pragma unroll
  for (int r = 0; r < RowsPerThread; ++r) {
    next_rows[r].load(source_state +
                      ((group * RowsPerThread + r) * kChunkV + row) * kDimK +
                      slice * kKPerThread);
  }
  float gate_inputs[4];

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gate_inputs[i] = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + lane + i * 32]);
    }
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      const int padded = (dim / kKPerThread) * kSlicePad + dim % kKPerThread;
      shared_q[t][padded] = q_values[i] * q_scale;
      shared_k[t][padded] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int chunk = group * RowsPerThread; chunk < kNumChunks;
       chunk += kChunkStep) {
    int value[RowsPerThread];
    int64_t row_offset[RowsPerThread];
    float h[RowsPerThread][kKPerThread];
#pragma unroll
    for (int r = 0; r < RowsPerThread; ++r) {
      value[r] = (chunk + r) * kChunkV + row;
      row_offset[r] = value[r] * kDimK + slice * kKPerThread;
      next_rows[r].unpack(h[r]);
    }
    if (chunk + kChunkStep < kNumChunks) {
#pragma unroll
      for (int r = 0; r < RowsPerThread; ++r) {
        next_rows[r].load(source_state + row_offset[r] +
                          kChunkStep * kChunkV * kDimK);
      }
    }
    for (int t = 0; t < num_tokens; ++t) {
      const float decay = shared_decay[t];
      const float* k_slice = &shared_k[t][slice * kSlicePad];
      const float* q_slice = &shared_q[t][slice * kSlicePad];
      float hk[RowsPerThread];
#pragma unroll
      for (int r = 0; r < RowsPerThread; ++r) {
        hk[r] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < kKPerThread; ++i) {
        const float key = k_slice[i];
#pragma unroll
        for (int r = 0; r < RowsPerThread; ++r) {
          h[r][i] *= decay;
          hk[r] += h[r][i] * key;
        }
      }
      float delta[RowsPerThread];
      float hq[RowsPerThread];
#pragma unroll
      for (int r = 0; r < RowsPerThread; ++r) {
        hk[r] = row_reduce_sum(hk[r]);
        delta[r] =
            (__bfloat162float(shared_v[t][value[r]]) - hk[r]) * shared_beta[t];
        hq[r] = 0.0f;
      }
#pragma unroll
      for (int i = 0; i < kKPerThread; ++i) {
        const float key = k_slice[i];
        const float query = q_slice[i];
#pragma unroll
        for (int r = 0; r < RowsPerThread; ++r) {
          h[r][i] += key * delta[r];
          hq[r] += h[r][i] * query;
        }
      }
      const int destination_slot = shared_slots[t];
#pragma unroll
      for (int r = 0; r < RowsPerThread; ++r) {
        hq[r] = row_reduce_sum(hq[r]);
        if (slice == 0) {
          shared_out[t][value[r]] = __float2bfloat16(hq[r]);
        }
        if (destination_slot > 0) {
          store_row_slice<StateT>(
              state +
                  static_cast<int64_t>(destination_slot) * strides.state_slot +
                  head_offset + row_offset[r],
              h[r]);
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate = SigmoidGate ? sigmoid_fast(gate_inputs[i])
                                     : silu_fast(gate_inputs[i]);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
void launch_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& norm_weight,
    torch::stable::Tensor& out, const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* output_gate, int num_key_heads, int num_value_heads,
    double scale, double norm_eps, GdnDecodeStrides strides,
    bool longest_first) {
  using torch::headeronly::ScalarType;

  const auto dt_bias_scalar_type = dt_bias.scalar_type();
  const int dt_bias_type =
      dt_bias_scalar_type == ScalarType::Float
          ? kDtBiasFloat32
          : (dt_bias_scalar_type == ScalarType::BFloat16 ? kDtBiasBFloat16
                                                         : kDtBiasFloat16);
  torch::stable::accelerator::DeviceGuard const device_guard(
      mixed_qkv.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(mixed_qkv.get_device_index());
  const int num_requests = static_cast<int>(state_indices.size(0));
  const auto launch = [&]<int Groups, int RowsPerThread, int MinBlocks>() {
    // Ordering pays off only where blocks outnumber the SMs (one group).
    const bool ordered = Groups == 1 && longest_first;
    const dim3 grid = ordered ? dim3(num_value_heads, num_requests)
                              : dim3(num_requests, num_value_heads);
    gdn_decode_post_conv_mtp_kernel<StateT, ValueHeadsPerKeyHead, SigmoidGate,
                                    Groups, RowsPerThread, MinBlocks>
        <<<grid, kGroupThreads * Groups, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), a, b,
            static_cast<const float*>(a_log.data_ptr()), dt_bias.data_ptr(),
            static_cast<const int*>(state_indices.data_ptr()),
            static_cast<const int*>(cu_seqlens.data_ptr()),
            static_cast<const int*>(num_accepted_tokens.data_ptr()),
            static_cast<StateT*>(state.data_ptr()), output_gate,
            norm_weight.data_ptr(), static_cast<__nv_bfloat16*>(out.data_ptr()),
            num_key_heads, num_value_heads,
            static_cast<int>(state_indices.size(1)), dt_bias_type,
            norm_weight.scalar_type() == ScalarType::BFloat16,
            static_cast<float>(scale), static_cast<float>(norm_eps), strides,
            ordered);
  };
  // Up to about one block per SM, a head's four chunks run side by side
  // (B200, 1-4 requests x 24 heads: 0.74-0.78x the time of walking them in
  // turn); beyond that, 256-thread blocks walk them in turn, two value rows per
  // thread so each shared-memory read of q and k serves both (B200, 8-64
  // requests x 24 heads: 0.86-0.97x the one-row walk), two blocks per SM up to
  // two blocks per SM's worth and three above.
  const int64_t blocks = static_cast<int64_t>(num_requests) * num_value_heads;
  if (blocks <= 148) {
    launch.template operator()<4, 1, 1>();
  } else if (blocks <= 2 * 148) {
    launch.template operator()<1, 2, 2>();
  } else {
    launch.template operator()<1, 2, 3>();
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "GDN decode MTP post-conv kernel launch failed: ",
                  cudaGetErrorString(error));
}

}  // namespace

void fused_gdn_decode_post_conv_mtp(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation,
    bool longest_first) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(
      mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == ScalarType::BFloat16,
      "mixed_qkv must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(a.is_cuda() && a.scalar_type() == ScalarType::BFloat16,
                  "a must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(b.is_cuda() && b.scalar_type() == ScalarType::BFloat16,
                  "b must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(a_log.is_cuda() && a_log.scalar_type() == ScalarType::Float,
                  "A_log must be a CUDA float32 tensor");
  const auto dt_bias_scalar_type = dt_bias.scalar_type();
  STD_TORCH_CHECK(
      dt_bias.is_cuda() && (dt_bias_scalar_type == ScalarType::Float ||
                            dt_bias_scalar_type == ScalarType::BFloat16 ||
                            dt_bias_scalar_type == ScalarType::Half),
      "dt_bias must be a CUDA float32, bfloat16, or float16 tensor");
  STD_TORCH_CHECK(
      state_indices.is_cuda() && state_indices.scalar_type() == ScalarType::Int,
      "state_indices must be a CUDA int32 tensor");
  STD_TORCH_CHECK(
      cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == ScalarType::Int,
      "cu_seqlens must be a CUDA int32 tensor");
  STD_TORCH_CHECK(num_accepted_tokens.is_cuda() &&
                      num_accepted_tokens.scalar_type() == ScalarType::Int,
                  "num_accepted_tokens must be a CUDA int32 tensor");
  const auto state_scalar_type = state.scalar_type();
  STD_TORCH_CHECK(
      state.is_cuda() && (state_scalar_type == ScalarType::Float ||
                          state_scalar_type == ScalarType::BFloat16),
      "state must be a CUDA float32 or bfloat16 tensor");
  STD_TORCH_CHECK(output_gate.is_cuda() &&
                      output_gate.scalar_type() == ScalarType::BFloat16,
                  "output_gate must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(norm_weight.is_cuda() &&
                      (norm_weight.scalar_type() == ScalarType::Float ||
                       norm_weight.scalar_type() == ScalarType::BFloat16),
                  "norm_weight must be a CUDA float32 or bfloat16 tensor");
  STD_TORCH_CHECK(out.is_cuda() && out.scalar_type() == ScalarType::BFloat16,
                  "out must be a CUDA bfloat16 tensor");
  STD_TORCH_CHECK(
      output_gate_activation == "silu" || output_gate_activation == "sigmoid",
      "output_gate_activation must be 'silu' or 'sigmoid'");

  STD_TORCH_CHECK(mixed_qkv.dim() == 2,
                  "mixed_qkv must have shape [L, 2 * H * 128 + HV * 128]");
  const int num_tokens = static_cast<int>(mixed_qkv.size(0));
  STD_TORCH_CHECK(num_tokens > 0,
                  "GDN decode MTP fusion requires at least one token");
  STD_TORCH_CHECK(
      state.dim() == 4 && state.size(2) == kDimV && state.size(3) == kDimK,
      "state must have shape [slots, HV, 128, 128]");
  const int num_value_heads = static_cast<int>(state.size(1));
  const int64_t key_width =
      mixed_qkv.size(1) - static_cast<int64_t>(num_value_heads) * kDimV;
  STD_TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
                  "mixed_qkv width is inconsistent with state");
  const int num_key_heads = static_cast<int>(key_width / (2 * kDimK));
  const int value_heads_per_key_head = num_value_heads / num_key_heads;
  STD_TORCH_CHECK(
      num_value_heads % num_key_heads == 0 &&
          ((value_heads_per_key_head >= 1 && value_heads_per_key_head <= 4) ||
           value_heads_per_key_head == 8),
      "GDN decode MTP fusion requires HV/H in {1, 2, 3, 4, 8}");

  STD_TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(0) > 0 &&
                      state_indices.size(1) > 0 &&
                      state_indices.size(1) <= kMaxMtpTokens,
                  "state_indices must have shape [N, S] with 1 <= S <= 8");
  const int num_requests = static_cast<int>(state_indices.size(0));
  STD_TORCH_CHECK(
      cu_seqlens.dim() == 1 && cu_seqlens.numel() == num_requests + 1,
      "cu_seqlens must have N + 1 elements");
  STD_TORCH_CHECK(num_accepted_tokens.dim() == 1 &&
                      num_accepted_tokens.numel() == num_requests,
                  "num_accepted_tokens must have N elements");
  STD_TORCH_CHECK(
      a.dim() == 2 && a.size(0) == num_tokens && a.size(1) == num_value_heads,
      "a must have shape [L, HV]");
  STD_TORCH_CHECK(
      b.dim() == 2 && b.size(0) == num_tokens && b.size(1) == num_value_heads,
      "b must have shape [L, HV]");
  STD_TORCH_CHECK(a_log.is_contiguous() && a_log.numel() == num_value_heads,
                  "A_log must be contiguous with HV elements");
  STD_TORCH_CHECK(dt_bias.is_contiguous() && dt_bias.numel() == num_value_heads,
                  "dt_bias must be contiguous with HV elements");
  STD_TORCH_CHECK(state_indices.is_contiguous(),
                  "state_indices must be contiguous");
  STD_TORCH_CHECK(cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous");
  STD_TORCH_CHECK(num_accepted_tokens.is_contiguous(),
                  "num_accepted_tokens must be contiguous");
  STD_TORCH_CHECK(output_gate.dim() == 3 && output_gate.size(0) == num_tokens &&
                      output_gate.size(1) == num_value_heads &&
                      output_gate.size(2) == kDimV,
                  "output_gate must have shape [L, HV, 128]");
  STD_TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV,
                  "norm_weight must be contiguous with 128 elements");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == num_tokens &&
                      out.size(1) == num_value_heads && out.size(2) == kDimV,
                  "out must have shape [L, HV, 128]");
  STD_TORCH_CHECK(mixed_qkv.stride(1) == 1,
                  "mixed_qkv channels must be contiguous");
  STD_TORCH_CHECK(a.stride(1) == 1 && b.stride(1) == 1,
                  "a and b heads must be contiguous");
  STD_TORCH_CHECK(state.stride(0) >= num_value_heads * kDimV * kDimK &&
                      state.stride(1) == kDimV * kDimK &&
                      state.stride(2) == kDimK && state.stride(3) == 1,
                  "state must have contiguous [HV, 128, 128] slot contents");
  const int state_elements_per_copy =
      state_scalar_type == ScalarType::Float ? 4 : 8;
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                      state.stride(0) % state_elements_per_copy == 0,
                  "state slots must preserve 16-byte alignment");
  STD_TORCH_CHECK(output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV,
                  "output_gate head rows must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  STD_TORCH_CHECK(norm_eps >= 0.0, "norm_eps must be non-negative");

  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                 output_gate.stride(0), state.stride(0)};
  const auto* a_ptr = static_cast<const __nv_bfloat16*>(a.data_ptr());
  const auto* b_ptr = static_cast<const __nv_bfloat16*>(b.data_ptr());
  const auto* output_gate_ptr =
      static_cast<const __nv_bfloat16*>(output_gate.data_ptr());
  const auto launch = [&]<typename StateT, int ValueHeadsPerKeyHead,
                          bool SigmoidGate>() {
    launch_gdn_decode_post_conv_mtp<StateT, ValueHeadsPerKeyHead, SigmoidGate>(
        mixed_qkv, a_log, dt_bias, state_indices, cu_seqlens,
        num_accepted_tokens, state, norm_weight, out, a_ptr, b_ptr,
        output_gate_ptr, num_key_heads, num_value_heads, scale, norm_eps,
        strides, longest_first);
  };
  const auto dispatch_state_type = [&]<int ValueHeadsPerKeyHead>() {
    if (state_scalar_type == ScalarType::Float) {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<float, ValueHeadsPerKeyHead, true>();
      } else {
        launch.template operator()<float, ValueHeadsPerKeyHead, false>();
      }
    } else {
      if (output_gate_activation == "sigmoid") {
        launch.template operator()<__nv_bfloat16, ValueHeadsPerKeyHead, true>();
      } else {
        launch
            .template operator()<__nv_bfloat16, ValueHeadsPerKeyHead, false>();
      }
    }
  };
  switch (value_heads_per_key_head) {
    case 1:
      dispatch_state_type.template operator()<1>();
      break;
    case 2:
      dispatch_state_type.template operator()<2>();
      break;
    case 3:
      dispatch_state_type.template operator()<3>();
      break;
    case 4:
      dispatch_state_type.template operator()<4>();
      break;
    default:
      dispatch_state_type.template operator()<8>();
      break;
  }
}
