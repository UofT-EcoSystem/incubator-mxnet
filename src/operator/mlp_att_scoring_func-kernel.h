#pragma once

#include "layer_norm-kernel.h"

namespace mxnet {
namespace op {
namespace {

template<typename AType, typename DType, typename IType>
__global__ void MLPAttScoringFuncForwardKernelContig(const int nbatch, const int nchannel,
                                                     const bool layer_norm, 
                                                     const AType eps,
                                                     const DType* __restrict__ src_hidden,
                                                     const DType* __restrict__ qry_hidden,
                                                           DType* __restrict__ in_data,
                                                     const DType* __restrict__ gamma,
                                                     const DType* __restrict__ beta,
                                                     DType* __restrict__ out_data,
                                                     DType* __restrict__ mean_data,
                                                     DType* __restrict__ std_data) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthread = blockDim.x * blockDim.y;
  IType count = 0;
  AType mean = 0;
  AType sigma2 = 0;

  if (bid < nbatch) {
    for (int i = tid; i < nchannel; i += nthread) {
      in_data[bid * nchannel + i] = qry_hidden[blockIdx.y * nchannel + i] +
                                    src_hidden[bid        * nchannel + i];
    }
    if (layer_norm) {
    extern __shared__ char buf[];  // Shared memory
    const DType* col_vals = in_data + bid * nchannel;
    BlockWelfordOnlineSum(col_vals, nchannel, mean, sigma2, count);

    // Merge the mean/sigma2 within a warp
    // Use the Chan's Parallel Algorithm to merge all (mean, sigma2, counts)
    // within a warp of threads.
    // After calling the function, threadIdx.x == 0 will store the result of
    // the aggregated (mean, sigma2, counts).
    for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
      AType meanB = warp_shfl_xor(mean, mask);
      AType sigma2B = warp_shfl_xor(sigma2, mask);
      IType countB = warp_shfl_xor(count, mask);
      ChanMergePartition(meanB, sigma2B, countB, mean, sigma2, count);
    }
    if (blockDim.y > 1) {
      // Inter-warp reduction. Copy the upper-half of the warps to shared memory
      // and merge with the lower-half warp
      AType* mean_buf = reinterpret_cast<AType*>(buf);
      AType* sigma2_buf =
        reinterpret_cast<AType*>(buf + sizeof(AType) * blockDim.y / 2 * blockDim.x);
      IType* count_buf = reinterpret_cast<IType*>(buf + sizeof(AType) * blockDim.y * blockDim.x);
      for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          mean_buf[idx] = mean;
          sigma2_buf[idx] = sigma2;
          count_buf[idx] = count;
        }
        __syncthreads();
        if (threadIdx.y < offset) {
          const int idx = threadIdx.y * blockDim.x + threadIdx.x;
          ChanMergePartition(mean_buf[idx], sigma2_buf[idx], count_buf[idx], mean, sigma2, count);
        }
        __syncthreads();
      }
      // Broadcast the result to all threads
      if (threadIdx.y == 0) {
        mean_buf[threadIdx.x] = mean;
        sigma2_buf[threadIdx.x] = sigma2;
      }
      __syncthreads();
      mean = mean_buf[threadIdx.x];
      sigma2 = sigma2_buf[threadIdx.x] / nchannel;
    } else {
      sigma2 /= nchannel;
    }
    // Calculate the out_data: gamma * (x - mean) / sqrt(var + eps) + beta
    AType std_eps = sqrt(sigma2 + eps);
    AType invstd_eps = DType(1.0) / std_eps;
    DType* out_col_val = out_data + bid * nchannel;

    if (gamma != NULL && beta != NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = gamma[i] * static_cast<DType>(invstd_eps *
                                                       (static_cast<AType>(col_vals[i]) - mean))
                                                         + beta[i];
      }
    } else if (gamma == NULL && beta != NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = static_cast<DType>(invstd_eps * (static_cast<AType>(col_vals[i]) - mean))
                                                       + beta[i];
      }
    } else if (gamma != NULL && beta == NULL) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = gamma[i] * static_cast<DType>(invstd_eps *
                                                       (static_cast<AType>(col_vals[i]) - mean));
      }
    } else {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = static_cast<DType>(invstd_eps * (static_cast<AType>(col_vals[i]) - mean));
      }
    }
    for (int i = tid; i < nchannel; i += nthread) {
      out_col_val[i] = tanh(out_col_val[i]);
    }
    // Write the out_data and var_data
    if (threadIdx.x == 0 && 
        threadIdx.y == 0 &&
        mean_data != nullptr &&
        std_data  != nullptr) {
      mean_data[bid] = static_cast<DType>(mean);
      std_data[bid] = static_cast<DType>(std_eps);
    }
    } else {  // if (!layer_norm)
    for (int i = tid; i < nchannel; i += nthread) {
      out_data[bid * nchannel + i] = tanh(in_data[bid * nchannel + i]);
    }
    }  // if (layer_norm)
  }  // if (bid < nbatch)
}

template<typename AType, typename DType>
__global__ void MLPAttScoringFuncBackwardKernel_PartGammaBeta(const int nbatch,
                                                              const int seqlen,
                                                              const int nchannel,
                                                              const DType* __restrict__ in_data,
                                                              const DType* __restrict__ out_grad,
                                                              const DType* __restrict__ out_data,
                                                              const DType* __restrict__ mean_data,
                                                              const DType* __restrict__ std_data,
                                                              AType* __restrict__ part_gamma_grad,
                                                              AType* __restrict__ part_beta_grad) {
  extern __shared__ char buf[];
  AType* d_buf = reinterpret_cast<AType*>(buf);
  const int npart = gridDim.y;
  const int block_row_num = (nbatch + npart - 1) / npart;
  // The rows are divided into `npart` parts. Each threadblock calculates the reduction result
  // within the corresponding row ranges.
  int row_stride = blockDim.x + 1;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r_begin = blockIdx.y * block_row_num;
  int r_end = min((blockIdx.y + 1) * block_row_num, nbatch);
  AType* buf_gamma_grad = d_buf;
  AType* buf_beta_grad = d_buf + blockDim.y * row_stride;
  AType local_gamma_grad = 0;
  AType local_beta_grad = 0;

  if (c < nchannel) {
    for (int r_b = r_begin; r_b < r_end; r_b += blockDim.y) {
      int r = r_b + threadIdx.y;
      if (r < r_end) {
        AType local_mean = static_cast<AType>(mean_data[r]);
        AType local_std = static_cast<AType>(std_data[r]);
        int read_idx = r * nchannel + c;
        AType local_in_data = static_cast<AType>(in_data[read_idx]);
        AType local_out_grad = static_cast<AType>(out_grad[read_idx] *
                                                  (1 - out_data[read_idx] *
                                                       out_data[read_idx]));
        local_gamma_grad += (local_in_data - local_mean) / local_std * local_out_grad;
        local_beta_grad += local_out_grad;
      }
    }
  }
  buf_gamma_grad[threadIdx.y * row_stride + threadIdx.x] = local_gamma_grad;
  buf_beta_grad[threadIdx.y * row_stride + threadIdx.x] = local_beta_grad;
  __syncthreads();
  for (int offset = blockDim.y/2;  offset > 1;  offset >>= 1) {
    if (threadIdx.y < offset) {
      int idx1 = threadIdx.y * row_stride + threadIdx.x;
      int idx2 = (threadIdx.y + offset) * row_stride + threadIdx.x;
      buf_gamma_grad[idx1] += buf_gamma_grad[idx2];
      buf_beta_grad[idx1] += buf_beta_grad[idx2];
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && c < nchannel) {
    part_gamma_grad[blockIdx.y * nchannel + c] = buf_gamma_grad[threadIdx.x]
                                                   + buf_gamma_grad[threadIdx.x + row_stride];
    part_beta_grad[blockIdx.y * nchannel + c] = buf_beta_grad[threadIdx.x]
                                                   + buf_beta_grad[threadIdx.x + row_stride];
  }
}

template<int LOAD_UNROLL, bool data_addto, typename AType, typename DType>
__global__ void MLPAttScoringFuncBackwardKernel_Data(const int nbatch,
                                                     const int nchannel,
                                                     const DType* __restrict__ in_data,
                                                     const DType* __restrict__ out_grad,
                                                     const DType* __restrict__ out_data,
                                                     const DType* __restrict__ mean_data,
                                                     const DType* __restrict__ std_data,
                                                     const DType* __restrict__ gamma,
                                                     DType* src_hidden_grad,
                                                     DType* qry_hidden_grad) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int nthread = blockDim.x * blockDim.y;
  if (bid < nbatch) {
    // Shared memory with size blockDim.y * blockDim.x * sizeof(DType)
    extern __shared__ char buf[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // 1. Calculate: mean(out_grad * gamma / std, axis=-1)
    //               mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    AType sum_val0 = 0;  // Stores mean(out_grad * gamma / std, axis=-1)
    AType sum_val1 = 0;  // Stores mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    AType mean = static_cast<AType>(mean_data[bid]);
    AType invstd_eps = AType(1) / static_cast<AType>(std_data[bid]);
    int l = LOAD_UNROLL * tid;
    for (; l + LOAD_UNROLL - 1 < nchannel; l += nthread * LOAD_UNROLL) {
#pragma unroll
      for (int i = 0; i < LOAD_UNROLL; ++i) {
        AType ele_og = static_cast<AType>(out_grad[bid * nchannel + l + i] *
                                          (1 - out_data[bid * nchannel + l + i] *
                                               out_data[bid * nchannel + l + i]));
        AType ele_x = static_cast<AType>(in_data[bid * nchannel + l + i]);
        AType ele_gamma = static_cast<AType>(gamma[l + i]);
        sum_val0 += ele_og * ele_gamma * invstd_eps;
        sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
      }
    }
    for (; l < nchannel; ++l) {
      AType ele_og = static_cast<AType>(out_grad[bid * nchannel + l] *
                                        (1 - out_data[bid * nchannel + l] *
                                             out_data[bid * nchannel + l]));
      AType ele_x = static_cast<AType>(in_data[bid * nchannel + l]);
      AType ele_gamma = static_cast<AType>(gamma[l]);
      sum_val0 += ele_og * ele_gamma * invstd_eps;
      sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
    }
    // Intra-warp reduction (all-reduce)
    for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
      sum_val0 += warp_shfl_xor(sum_val0, mask);
      sum_val1 += warp_shfl_xor(sum_val1, mask);
    }
    // Inter-warp reduction (all-reduce)
    if (blockDim.y > 1) {
      AType* sum_val0_buf = reinterpret_cast<AType*>(buf);
      AType* sum_val1_buf =
        reinterpret_cast<AType*>(buf + blockDim.y / 2 * blockDim.x * sizeof(AType));
      for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          sum_val0_buf[idx] = sum_val0;
          sum_val1_buf[idx] = sum_val1;
        }
        __syncthreads();
        if (threadIdx.y < offset) {
          const int idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_val0 += sum_val0_buf[idx];
          sum_val1 += sum_val1_buf[idx];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        sum_val0_buf[threadIdx.x] = sum_val0;
        sum_val1_buf[threadIdx.x] = sum_val1;
      }
      __syncthreads();
      sum_val0 = sum_val0_buf[threadIdx.x];
      sum_val1 = sum_val1_buf[threadIdx.x];
    }
    sum_val0 /= nchannel;
    sum_val1 /= nchannel;
    // 2. Calculate the gradient as
    //      out_grad * gamma / std - sum_val0 - (x - mean) / std * sum_val1
    for (int l = tid; l < nchannel; l += nthread) {
      AType ele_out_grad = static_cast<AType>(out_grad[bid * nchannel + l] * 
                                              (1 - out_data[bid * nchannel + l] *
                                                   out_data[bid * nchannel + l]));
      AType ele_x = static_cast<AType>(in_data[bid * nchannel + l]);
      AType ele_gamma = static_cast<AType>(gamma[l]);


      src_hidden_grad[bid * nchannel + l] +=
        static_cast<DType>(ele_out_grad * ele_gamma * invstd_eps
                             - sum_val0 - (ele_x - mean) * invstd_eps * sum_val1);
      atomicAdd(&qry_hidden_grad[blockIdx.y * nchannel  + l],
        static_cast<DType>(ele_out_grad * ele_gamma * invstd_eps
                             - sum_val0 - (ele_x - mean) * invstd_eps * sum_val1));
    }
  }
}

}  // namespace anonymous
}  // namespace op
}  // namespace mxnet
