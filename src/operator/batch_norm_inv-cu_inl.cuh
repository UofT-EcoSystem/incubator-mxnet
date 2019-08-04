#pragma once

#include <vector>

#include "batch_norm_inv-inl.h"

namespace mxnet {
        namespace op {

template < typename RealType >
static __global__ void _cuda_batch_norm_inv_forward(
        const RealType * const __restrict__ output,
        const RealType * const __restrict__ mean,
        const RealType * const __restrict__ inv_var,
        const RealType * const __restrict__ gamma,
        const RealType * const __restrict__ beta,
              RealType * const __restrict__ data,
        const unsigned shape_size, 
        const unsigned batch_size,
        const unsigned stride_dim);

template < typename DType >
class CUBatchNormInvOp : public Operator 
{
private:
        BatchNormInvParam _param; bool _initialized = false;
public:
        explicit CUBatchNormInvOp(BatchNormInvParam param)
        {
                _param = param;
        }
private:
        void _Init(mshadow::Stream < gpu > * cuda_stream,
                   const std::vector < TBlob > &  in_data,
                   const std::vector < TBlob > & out_data)
        {
                using namespace mshadow;

                TBlob output = in_data[int(EnumOpInputs::Output)];

                _param.shape_size = output.shape_.Size();
                _param.batch_size = output.shape_[1];
                _param.stride_dim = 1;

                for (int idx = 2; idx < output.ndim(); ++idx)
                {
                        _param.stride_dim *= output.shape_[idx];
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

                std::size_t in_expected = 5, out_expected = 1;

                CHECK_EQ( in_data.size(),  in_expected); // output, mean, inv_var, 
                                                         // gamma, beta
                CHECK_EQ(out_data.size(), out_expected); // data

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > (); 

                TBlob output  =  in_data[int(EnumOpInputs ::Output)];
                TBlob mean    =  in_data[int(EnumOpInputs ::Mean)];
                TBlob inv_var =  in_data[int(EnumOpInputs ::InvVar)];
                TBlob gamma   =  in_data[int(EnumOpInputs ::Gamma)];
                TBlob beta    =  in_data[int(EnumOpInputs ::Beta)];
                TBlob data    = out_data[int(EnumOpOutputs::Data)];

                CHECK_EQ(output .CheckContiguous(), true);
                CHECK_EQ(mean   .CheckContiguous(), true);
                CHECK_EQ(inv_var.CheckContiguous(), true);
                CHECK_EQ(gamma  .CheckContiguous(), true);
                CHECK_EQ(beta   .CheckContiguous(), true);
                CHECK_EQ(data   .CheckContiguous(), true);

                if (!_initialized)
                {
                        _Init(cuda_stream, in_data, out_data);
                }

                _cuda_batch_norm_inv_forward < DType >
                        <<<
                                (_param.shape_size - 1) / 128 + 1, 128, 0,
                                Stream < gpu > ::GetStream(cuda_stream)
                        >>>
                        (
                                reinterpret_cast < DType * > (output .dptr_),
                                reinterpret_cast < DType * > (mean   .dptr_),
                                reinterpret_cast < DType * > (inv_var.dptr_),
                                reinterpret_cast < DType * > (gamma  .dptr_),
                                reinterpret_cast < DType * > (beta   .dptr_),
                                reinterpret_cast < DType * > (data   .dptr_),
                               _param.shape_size,
                               _param.batch_size,
                               _param.stride_dim
                        );
        }
};

template < typename RealType >
__global__ void _cuda_batch_norm_inv_forward(
        const RealType * const __restrict__ output,
        const RealType * const __restrict__ mean,
        const RealType * const __restrict__ inv_var,
        const RealType * const __restrict__ gamma,
        const RealType * const __restrict__ beta,
              RealType * const __restrict__ data,
        const unsigned shape_size, 
        const unsigned batch_size,
        const unsigned stride_dim)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (g_threadIdx >= shape_size) { return ; }

        const unsigned batch_idx = (g_threadIdx % (batch_size * stride_dim)) / stride_dim;
        const RealType gamma_x_inv_var = gamma[batch_idx] * inv_var[batch_idx];

        if (gamma_x_inv_var == 0)
        {
                data[g_threadIdx] = 0;
        } 
        else 
        {
                data[g_threadIdx] = (output[g_threadIdx] - beta[batch_idx]) / 
                        gamma_x_inv_var + mean[batch_idx];
        }
}

        }  // namespace op
}  // namespace mxnet
