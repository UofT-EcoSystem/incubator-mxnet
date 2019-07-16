#pragma once

#include <cub/cub.cuh>

#include "layer_norm-inl.h"

namespace mxnet {
        namespace op {

/**
 * Forward Pass of Layer Normalization
 */
template < typename RealType >
static __global__ void _cuda_layer_norm(
        const RealType * const __restrict__ data,
        const RealType * const __restrict__ gamma,
        const RealType * const __restrict__ beta,
              RealType * const __restrict__ output,
              RealType * const __restrict__ mean,
              RealType * const __restrict__ std,
        const RealType eps);

template < typename DType >
class CULayerNormOp : public Operator
{
private:
        LayerNormParam _param;
        bool _initialized = false;
public:
        explicit CULayerNormOp(LayerNormParam param)
        {
                _param = param;
        }
private:
        void _Init(mshadow::Stream < gpu > * cuda_stream,
                   const std::vector < TBlob > &  in_data,
                   const std::vector < TBlob > & out_data)
        {
                using namespace mshadow;

                CHECK_EQ(_initialized, false);

                _param.total_size = in_data.shape_.Size();
                _param.state_size = in_data.shape_[in_data.shape_.ndim() - 1];

                _initialized = true;
        }
public:
        virtual void  Forward(const Context & ctx,
                              const std::vector < TBlob > &  in_data,
                              const std::vector < OpReqType > &  req,
                              const std::vector < TBlob > & out_data,
                              const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t in_expected = 3, out_expected = 3;

                CHECK_EQ( in_data.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected);

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                TBlob data   =  in_data[int(EnumOpInputs ::Data)];
                TBlob gamma  =  in_data[int(EnumOpInputs ::Gamma)];
                TBlob beta   =  in_data[int(EnumOpInputs ::Beta)];
                TBlob output = out_data[int(EnumOpOutputs::Output)];
                TBlob mean   = out_data[int(EnumOpOutputs::Mean)];
                TBlob std    = out_data[int(EnumOpOutputs::STD)];

                CHECK_EQ(data  .CheckContiguous(), true);
                CHECK_EQ(gamma .CheckContiguous(), true);
                CHECK_EQ(beta  .CheckContiguous(), true);
                CHECK_EQ(output.CheckContiguous(), true);
                CHECK_EQ(mean  .CheckContiguous(), true);
                CHECK_EQ(std   .CheckContiguous(), true);

                if (!_initialized)
                {
                        _Init(cuda_stream, in_data, out_data);
                }

                _cuda_layer_norm <<<  >>>
        }
};



        } // namespace op 
}  // namespace mxnet
