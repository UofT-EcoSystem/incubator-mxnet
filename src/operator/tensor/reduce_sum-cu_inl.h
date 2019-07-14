#pragma once

#include <cub/cub.cuh>

#include "reduce_sum-inl.h"

namespace mxnet {
        namespace op {

#if defined(__CUDACC__)

/**
 * Forward Pass of Reduce Sum
 */
template < typename RealType, bool TNorm >
static __global__ void _cuda_reduce_sum(
        const RealType * const __restrict__   data,
              RealType * const __restrict__ output,
        const std::size_t reduce_dim, const std::size_t stride);

template < typename RealType, bool TNorm >
static __global__ void _cuda_reduce_sum_backward(
              RealType * const __restrict__   data_grad,
        const RealType * const __restrict__ output_grad,
        const OpReqType req, const std::size_t reduce_dim, const std::size_t stride);

template < typename DType, bool TNorm = false >
class CUEcoReduceSumOp : public Operator
{
private:
        EcoReduceSumParam _param;
        std::size_t _reduce_dim,
                    _stride;

        bool _initialized = false;
public:
        explicit CUEcoReduceSumOp(EcoReduceSumParam param)
        {
                _param = param;
        }
        ~CUEcoReduceSumOp() {}
private:
        void _Init(mshadow::Stream < gpu > * cuda_stream,
                const std::vector < TBlob > &  in_data,
                const std::vector < TBlob > & out_data)
        {
                CHECK_EQ(_initialized, false);

                std::size_t reduce_axis = 
                        (_param.axis + in_data[int(EnumOpInputs::Data)].shape_.ndim()) % 
                                       in_data[int(EnumOpInputs::Data)].shape_.ndim();

                _reduce_dim = in_data[int(EnumOpInputs::Data)].shape_[reduce_axis];
                _stride = 1;
                for (int dim_idx = in_data[int(EnumOpInputs::Data)].shape_.ndim() - 1;
                         dim_idx > reduce_axis;
                       --dim_idx)
                {
                        _stride *= in_data[int(EnumOpInputs::Data)].shape_[dim_idx];
                }
                _initialized = true;
        }
public:
        virtual void  Forward(const OpContext & ctx,
                              const std::vector < TBlob > &  in_data,
                              const std::vector < OpReqType > &  req,
                              const std::vector < TBlob > & out_data,
                              const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t  in_expected = 1,
                            out_expected = 1;

                CHECK_EQ( in_data.size(),  in_expected); // data
                CHECK_EQ(out_data.size(), out_expected); // output

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                TBlob   data =  in_data[int(EnumOpInputs ::Data)];
                TBlob output = out_data[int(EnumOpOutputs::Output)];

                CHECK_EQ(  data.CheckContiguous(), true);
                CHECK_EQ(output.CheckContiguous(), true);

                if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}

                CUDA_CALL(cudaMemsetAsync(output.dptr_, 0,
                                          output.shape_.Size() * sizeof(DType),
                                          Stream < gpu > ::GetStream(cuda_stream)
                                          ));
                _cuda_reduce_sum < DType, TNorm > 
                        <<<
                                dim3(data.shape_.Size() / _reduce_dim,
                                     (_reduce_dim - 1) / 128 + 1
                                     ),
                                128, 0,
                                Stream < gpu > ::GetStream(cuda_stream)
                        >>>
                        (
                                reinterpret_cast < DType * > (data  .dptr_),
                                reinterpret_cast < DType * > (output.dptr_),
                                _reduce_dim, _stride
                        );
        }

        virtual void Backward(const OpContext & ctx,
                              const std::vector < TBlob > & out_grad,
                              const std::vector < TBlob > &  in_data,
                              const std::vector < TBlob > & out_data,
                              const std::vector < OpReqType > &  req,
                              const std::vector < TBlob > &  in_grad,
                              const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t in_expected = 1, out_expected = 1;

                CHECK_EQ( in_data.size(),  in_expected);
                CHECK_EQ( in_grad.size(),  in_expected);
                CHECK_EQ(     req.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected);
                CHECK_EQ(out_grad.size(), out_expected);

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                TBlob   data_grad =  in_grad[int(EnumOpInputs ::Data)];
                TBlob output_grad = out_grad[int(EnumOpOutputs::Output)];

                CHECK_EQ(  data_grad.CheckContiguous(), true);
                CHECK_EQ(output_grad.CheckContiguous(), true);

                _cuda_reduce_sum_backward < DType, TNorm >
                        <<<
                                dim3(data_grad.shape_.Size() 
                                       / _reduce_dim,
                                     (_reduce_dim - 1) / 128 + 1
                                     ),
                                 128, 0,
                                Stream < gpu > ::GetStream(cuda_stream)
                        >>>
                        (
                                reinterpret_cast < DType * > (data_grad  .dptr_), 
                                reinterpret_cast < DType * > (output_grad.dptr_),
                                req[int(EnumOpInputs::Data)], _reduce_dim, _stride
                        );
        }
};  // class CUEcoReduceSumOp

template < typename RealType, bool TNorm >
__global__ void _cuda_reduce_sum(
        const RealType * const __restrict__   data,
              RealType * const __restrict__ output,
        const std::size_t reduce_dim, const std::size_t stride)
{
        typedef cub::BlockReduce < RealType, 128 > BlockReduce;
        __shared__ typename BlockReduce::TempStorage workspace;

        const unsigned this_thread_idx = 
                (threadIdx.x + blockIdx.y * 128) * stride + 
                  blockIdx.x % stride + 
                  blockIdx.x * reduce_dim;

        RealType this_thread_data = 
                (threadIdx.x + blockIdx.y * 128) < reduce_dim ? 
                        data[this_thread_idx] : 0;

        RealType aggregate = BlockReduce(workspace).Sum(this_thread_data);

        if (threadIdx.x == 0)
        {
                atomicAdd(&output[blockIdx.x],
                          aggregate / (TNorm ? reduce_dim : 1));
        }
}

template < typename RealType, bool TNorm >
__global__ void _cuda_reduce_sum_backward(
              RealType * const __restrict__   data_grad,
        const RealType * const __restrict__ output_grad,
        const OpReqType req, const std::size_t reduce_dim, const std::size_t stride)
{
        RealType output_grad_reg = output_grad[blockIdx.x];

        if ((threadIdx.x + blockIdx.y * 128) < reduce_dim) {

        // printf("threadIdx_x: %d, blockIdx_x: %d, blockIdx_y: %d, Idx: %ld, "
        //        "Stride: %ld, Reduce_Dim: %ld, Size: %ld\n",
        //        threadIdx.x,     blockIdx.x,     blockIdx.y,
        //        (threadIdx.x + blockIdx.y * 128) * stride + 
        //          blockIdx.x % stride + 
        //          blockIdx.x * reduce_dim, 
        //        stride, reduce_dim, size);

        if (req == kWriteTo)
        {
                data_grad[(threadIdx.x + blockIdx.y * 128) * stride + 
                            blockIdx.x % stride + 
                            blockIdx.x * reduce_dim]  = output_grad_reg / (TNorm ? reduce_dim : 1);
        }
        if (req == kAddTo)
        {
                data_grad[(threadIdx.x + blockIdx.y * 128) * stride + 
                            blockIdx.x % stride + 
                            blockIdx.x * reduce_dim] += output_grad_reg / (TNorm ? reduce_dim : 1);
        }

        }  // if (threadIdx.x + blockIdx.y * 128 < reduce_sum)
}

#endif  // defined(__CUDACC__)

        }  // namespace op 
}  // namespace mxnet
