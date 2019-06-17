#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "gru_cell-inl.h"

namespace mxnet {
        namespace op {

#if defined(__CUDACC__)

/// @brief Forward Pass of the GRU Nonlinear Block
/// @param input   [B x 3H]
/// @param state_h [B x 3H]
/// @param feature_map_reset_gate  [B x H]
/// @param feature_map_update_gate [B x H]
/// @param batch_size (parameter) Batch Size
/// @param state_size (parameter) State Size
template < typename RealType >
static __global__ void _cuda_gru_nonlin_block_forward(
        const RealType * const __restrict__ input,
        const RealType * const __restrict__ state_h,
              RealType * const __restrict__ feature_map_reset_gate,
              RealType * const __restrict__ feature_map_update_gate,
              RealType * const __restrict__ state_h_out,
        const unsigned batch_size,
        const unsigned state_size,
        const bool is_train);

/// @brief FullyConnected Layer $Y = X W^T$ Forward Pass
/// @param X [batch_size x input_size]
/// @param W [state_size x input_size]
/// @param Y [batch_size x state_size]
/// @param batch_size (parameter)
/// @param input_size (parameter)
/// @param state_size (parameter)
template < typename RealType >
static inline void FullyConnectedFW(cublasHandle_t cublas_handle,
	const RealType * const __restrict__ X,
	const RealType * const __restrict__ W,
	      RealType * const __restrict__ Y,
	const OpReqType req,
        const unsigned batch_size, 
	const unsigned input_size,
        const unsigned state_size);

/// @brief FullyConnected Layer $Y = X W^T$ Backward Pass on Weight ($dW = dY^T X$)
/// @param  X [batch_size x input_size]
/// @param dW [state_size x input_size]
/// @param dY [batch_size x state_size]
/// @param batch_size (parameter)
/// @param input_size (parameter)
/// @param state_size (parameter)
template < typename RealType >
static inline void FullyConnectedBWWeight(cublasHandle_t cublas_handle,
	const RealType * const __restrict__  X,
	      RealType * const __restrict__ dW,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,
        const unsigned batch_size, 
	const unsigned input_size,
        const unsigned state_size);

/// @brief FullyConnected Layer $Y = X W^T$ Backward Pass on Data ($dX = dY W$)
/// @param dX [batch_size x input_size]
/// @param  W [state_size x input_size]
/// @param dY [batch_size x state_size]
/// @param batch_size (parameter) 
/// @param input_size (parameter) 
/// @param state_size (parameter)
template < typename RealType >
static inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      RealType * const __restrict__ dX,
	const RealType * const __restrict__  W,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,
        const unsigned batch_size, 
	const unsigned input_size,
        const unsigned state_size);

template < typename DType >
class CUInvisGRUCellOp : public Operator
{
private:
        InvisGRUCellParam _param;

        bool _initailized = false;
        unsigned _temp_space_size;
public:
        explicit CUInvisGRUCellOp(InvisGRUCellParam param)
        {
                _param = param;
        }
        ~CUInvisGRUCellOp() {}
private:
        void _Init(mshadow::Stream < gpu > * cuda_stream,
                const std::vector < TBlob > &  in_data,
                const std::vector < TBlob > & out_data)
        {
                using namespace mshadow;

                CHECK_EQ(_initialized, false);

                Tensor < gpu, 2, DType > input   = in_data[int(EnumOpInputs::Input)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > state_h = in_data[int(EnumOpInputs::StateH)]
                        .get < gpu, 2, DType > (cuda_stream);
                
                _param.batch_size = input  .shape[0];
                _param.input_size = input  .shape[1];
                _param.state_size = state_h.shape[1];

                _temp_space_size = _param.batch_size * 3 * _param.state_size;
                _initailized = true;
        }
public:
        virtual void Forward(const OpContext & ctx,
                             const std::vector < TBlob > &  in_data,
                             const std::vector < OpReqType > &  req,
                             const std::vector < TBlob > & out_data,
                             const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t in_expected = 7, out_expected = 4;

                CHECK_EQ( in_data.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected); ///< state_h_out
                                                         ///< feature_map_reset_gate
                                                         ///< feature_map_update_gate
                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();


                Tensor < gpu, 2, DType > input      = in_data[int(EnumOpInputs::Input)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > state_h    = in_data[int(EnumOpInputs::StateH)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > i2h_weight = in_data[int(EnumOpInputs::I2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 1, DType > i2h_bias   = in_data[int(EnumOpInputs::I2HBias)]
                        .get < gpu, 1, DType > (cuda_stream);
                Tensor < gpu, 2, DType > h2h_weight = in_data[int(EnumOpInputs::H2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 1, DType > h2h_bias   = in_data[int(EnumOpInputs::H2HBias)]
                        .get < gpu, 1, DType > (cuda_stream);
                
                Tensor < gpu, 2, DType > state_h_out = out_data[int(EnumOpOutputs::StateHOut)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > feature_map_reset_gate = 
                        out_data[int(EnumOpOutputs::FeatureMapResetGate)]
                                .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > feature_map_update_gate = 
                        out_data[int(EnumOpOutputs::FeatureMapUpdateGate)]
                                .get < gpu, 2, DType > (cuda_stream);
                
                CHECK_EQ(input      .CheckContiguous(), true);
                CHECK_EQ(state_h    .CheckContiguous(), true);
                CHECK_EQ(i2h_weight .CheckContiguous(), true);
                CHECK_EQ(i2h_bias   .CheckContiguous(), true);
                CHECK_EQ(h2h_weight .CheckContiguous(), true);
                CHECK_EQ(h2h_bias   .CheckContiguous(), true);
                CHECK_EQ(state_h_out.CheckContiguous(), true)
                CHECK_EQ(feature_map_reset_gate .CheckContiguous(), true);
                CHECK_EQ(feature_map_update_gate.CheckContiguous(), true);

                if (!_initializer)
                {
                        _Init(cuda_stream, in_data, out_data);
                }

                Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
                        .get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);
                
                const unsigned BxH = _param.batch_size * _param.state_size;

                FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 input  .dptr_, i2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kWriteTo,
				_param.batch_size,
				_param.input_size,
				_param.state_size * 3);
		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 state_h.dptr_, h2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kAddTo,
				_param.batch_size,
				_param.state_size,
				_param.state_size * 3);
                _cuda_gru_cell_forward < DType > <<<
                        (BxH - 1) / 128 + 1, 128, 0,
                        Stream < gpu > ::GetStream(cuda_stream) >>> (
                        workspace  .dptr_,
                        i2h_bias   .dptr_,
			h2h_bias   .dptr_,
			feature_map_reset_gate .dptr_,
			feature_map_update_gate.dptr_,
                        state_h_out.dptr_,
		       _param.batch_size,
                       _param.state_size, 
			ctx.is_train);
        }

        virtual void Backward(const OpContext & ctx,
                              const std::vector < TBlob > & out_grad,
                              const std::vector < TBlob > &  in_data,
                              const std::vector < TBlob > & out_data,
                              const std::vector < OpReqType > &  req,
                              const std::vector < TBlob > &  in_grad,
                              const std::vector < TBlob > & aux_args)
        {

        }
};

template < typename RealType >
static __forceinline__ __device__ RealType __cu_sigmoid(RealType i)
{
	return 1.0 / (1.0 + exp(-i));
}

template < typename RealType > 
__global__ void _cuda_gru_cell_forward(
        const RealType * const __restrict__ workspace_i2h,
        const RealType * const __restrict__ workspace_h2h,
        const RealType * const __restrict__ i2h_bias,
        const RealType * const __restrict__ h2h_bias,
        const RealType * const __restrict__ state_h,
              RealType * const __restrict__ feature_map_reset_gate,
              RealType * const __restrict__ feature_map_update_gate,
              RealType * const __restrict__ state_h_out,
        const unsigned batch_size,
        const unsigned state_size,
        const bool is_train)
{
        const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
                BxH = batch_size * state_size;

        if (g_threadIdx >= BxH) { return ; }

        RealType reset_gate = __cu_sigmoid(
                workspace
                )
}


#endif  // defined(__CUDACC__)

        }  // namespace op
}  // namespace mxnet
