#pragma once

#include "layer_norm-kernel.h"

namespace mxnet {
namespace op {
namespace {

template<typename AType, typename DType, typename IType>
__global__ void MLPAttScoringFuncForwardKernelContig(const int nbatch,
                                                     const int nchannel,
                                                     const bool  layer_norm
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
    } else {  // if (!layer_norm) {
    for (int i = tid; i < nchannel; i += nthread) {
      out_data[bid * nchannel + i] = tanh(in_data[bid * nchannel + i]);
    }
    }  // if (layer_norm)
  }  // if (bid < nbatch)
}

}  // namespace anonymous
}  // namespace op
}  // namespace mxnet
