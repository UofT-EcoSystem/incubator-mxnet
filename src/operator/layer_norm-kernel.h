#pragma once



template <typename DType>
__device__ __forceinline__ DType warp_shfl(DType value, int src_lane,
                                           int width = 32, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

template <typename DType>
__device__ __forceinline__ DType warp_shfl_xor(DType value, int laneMask,
                                               int width = 32, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}


/* A single updating step of the Welford's online algorithm to calculate the mean and variance.
 * The value 'curr' will be accumulated to the (mean, sigma2, count) triplet.
 *
 */
template<typename DType, typename IType>
__device__ __forceinline__ void StepWelfordOnlineSum(const DType curr,
                                                     DType& mean,         //NOLINT
                                                     DType& sigma2,       //NOLINT
                                                     IType& count) {      //NOLINT
  count += IType(1);
  DType delta = curr - mean;
  mean += delta / count;
  sigma2 += delta * (curr - mean);
}

/* Merge the mean/variance of two partitions. It's the key step of the Chan's parallel algorithm.
 * The (lhs_mean, lhs_sigma2, lhs_count) will be merged into (rhs_mean, rhs_sigma2, rhs_count)
 *
 * See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for more details.
 *
 *  TODO(sxjscience) Explore the possibility of int lhs_count and rhs_count
 */
template<typename DType, typename IType>
__device__ __inline__ void ChanMergePartition(const DType lhs_mean,
                                              const DType lhs_sigma2,
                                              const IType lhs_count,
                                              DType& rhs_mean,         //NOLINT
                                              DType& rhs_sigma2,       //NOLINT
                                              IType& rhs_count) {      //NOLINT
  DType delta = rhs_mean - lhs_mean;
  DType nA = static_cast<DType>(lhs_count);
  DType nB = static_cast<DType>(rhs_count);
  rhs_count = nA + nB;
  if (rhs_count > DType(0)) {
    nA = nA / rhs_count;
    nB = nB / rhs_count;
    rhs_mean = nA * lhs_mean + nB * rhs_mean;
    rhs_sigma2 = rhs_sigma2 + lhs_sigma2 + delta * delta * nA * nB * rhs_count;
  } else {
    rhs_mean = DType(0);
    rhs_sigma2 = DType(0);
  }
}

/* Split the input column into multiple partitions and compute the mean/sigma of each partition.
 * Each thread will keep a mean/sigma2. The mean/sigma2 can be further merged to get the mean and
 * sigma2 of the column.
 */
template<typename AType, typename DType, typename IType>
__device__ __forceinline__ void BlockWelfordOnlineSum(const DType* __restrict__ col_vals,
                                                      const int nchannel,
                                                      AType& mean,         //NOLINT
                                                      AType& sigma2,       //NOLINT
                                                      IType& count) {      //NOLINT
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int nthread = blockDim.x * blockDim.y;
  // Each thread takes charge of 4 consecutive numbers. This should optimize the loading speed using
  // vectorized types like float4.
  // Also, to minimize branch divergence, we split the for-loop into two parts.
  int l = 4 * tid;
  for (; l + 3 < nchannel; l += 4 * nthread) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      StepWelfordOnlineSum(static_cast<AType>(col_vals[l + i]), mean, sigma2, count);
    }
  }
  for (; l < nchannel; ++l) {
    StepWelfordOnlineSum(static_cast<AType>(col_vals[l]), mean, sigma2, count);
  }
}

template<>
__device__ __forceinline__
void BlockWelfordOnlineSum<float, mshadow::half::half_t, int>
                                          (const mshadow::half::half_t* __restrict__ col_vals,
                                           const int nchannel,
                                           float& mean,                    //NOLINT
                                           float& sigma2,                  //NOLINT
                                           int& count) {                 //NOLINT
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int nthread = blockDim.x * blockDim.y;
  // We cast the input half pointer to half2 to optimize the loading speed.
  // Here, we need to notice that CUDA forces memory alignment, i.e.,
  // ASSERT static_cast<size_t>(ptr) % sizeof(dtype) == 0.
  // Thus, we need to shift the address of the half pointer to be aligned by half2.
  int align_shift = (reinterpret_cast<size_t>(col_vals) % 4) != 0;
  int padding = (nchannel - align_shift) % 2;
  int half2_size = (nchannel - align_shift) / 2;
  const __half2* half2_col_vals = reinterpret_cast<const __half2*>(col_vals + align_shift);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (align_shift) {
      StepWelfordOnlineSum(__half2float(col_vals[0].cuhalf_), mean, sigma2, count);
    }
    if (padding) {
      StepWelfordOnlineSum(__half2float(col_vals[nchannel - 1].cuhalf_), mean, sigma2, count);
    }
  }

  for (int l = tid; l < half2_size; l += nthread) {
    float2 ele_val =  __half22float2(half2_col_vals[l]);
    StepWelfordOnlineSum(ele_val.x, mean, sigma2, count);
    StepWelfordOnlineSum(ele_val.y, mean, sigma2, count);
  }
}

/* Fused CUDA kernel for the forward pass of layer normalization.
 * It computes the LayerNorm when axis=-1, i.e., contiguous reduction scenario.
 * Shape of the input tensors:
 *      in_data = (nbatch, nchannel)
 *      gamma = (nchannel,)
 *      beta = (nchannel,)
 *      out_data = (nchannel,)
 *      mean_data = (nbatch,)
 *      var_data = (nbatch,)
 *  It's always launched with (blockDim.x, blockDim.y) = (WARP_SIZE, blockDim.y)
 *  Also, when blockDim.y > 1, it requires shared memory that has size:
 *      sizeof(AType) * blockDim.y + sizeof(int) * blockDim.y / 2
 */
template<typename AType, typename DType, typename IType>
__global__ void LayerNormFusedForwardKernelContig(const int nbatch,
                                                  const int nchannel,
                                                  const AType eps,
                                                  const DType* __restrict__ in_data,
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
    // Write the out_data and var_data
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean_data[bid] = static_cast<DType>(mean);
      std_data[bid] = static_cast<DType>(std_eps);
    }
  }
}