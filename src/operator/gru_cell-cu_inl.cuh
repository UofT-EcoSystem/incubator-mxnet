#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "gru_cell-inl.h"

namespace mxnet {
        namespace op {

#if defined(__CUDACC__)

/// @brief Forward Pass of the GRU Nonlinear Block
/// @param workspace_i2h [B x 3H]
/// @param workspace_h2h [B x 3H]
/// @param i2h_bias [3H]
/// @param h2h_bias [3H] 
/// @param state_h [B x 3H]
/// @param feature_map_reset_gate  [B x H]
/// @param feature_map_h2h         [B x H]
/// @param feature_map_update_gate [B x H]
/// @param batch_size (parameter) Batch Size
/// @param state_size (parameter) State Size
template < typename RealType >
static __global__ void _cuda_gru_cell_forward(
        const RealType * const __restrict__ workspace_i2h,
        const RealType * const __restrict__ workspace_h2h,
        const RealType * const __restrict__ i2h_bias,
        const RealType * const __restrict__ h2h_bias,
        const RealType * const __restrict__ state_h,
              RealType * const __restrict__ feature_map_reset_gate,
              RealType * const __restrict__ feature_map_h2h,
              RealType * const __restrict__ feature_map_update_gate,
              RealType * const __restrict__ state_h_out,
        const unsigned batch_size, const unsigned state_size, const bool is_train);

/// @brief Backward Pass of the GRU Nonlinear Block
/// @param state_h_grad  [B x  H]
/// @param workspace_i2h [B x 3H]
/// @param workspace_h2h [B x 3H]
/// @param i2h_bias_grad [3H]
/// @param h2h_bias_grad [3H]
/// @param feature_map_reset_gate  [B x H]
/// @param feature_map_h2h         [B x H]
/// @param feature_map_update_gate [B x H]
/// @param state_h          [B x 3H]
/// @param state_h_out      [B x 3H]
/// @param state_h_out_grad [B x 3H]
/// @param batch_size (parameter) Batch Size
/// @param state_size (parameter) State Size
template < typename RealType >
static __global__ void _cuda_gru_cell_backward(
              RealType * const __restrict__ state_h_grad,
              RealType * const __restrict__ workspace_i2h,
              RealType * const __restrict__ workspace_h2h,
              RealType * const __restrict__ i2h_bias_grad,
              RealType * const __restrict__ h2h_bias_grad,
        const RealType * const __restrict__ feature_map_reset_gate,
        const RealType * const __restrict__ feature_map_h2h,
        const RealType * const __restrict__ feature_map_update_gate,
        const RealType * const __restrict__ state_h,
        const RealType * const __restrict__ state_h_out,
        const RealType * const __restrict__ state_h_out_grad,
        const unsigned batch_size, const unsigned state_size);

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

        bool _initialized = false;
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
                
                _param.batch_size = input  .shape_[0];
                _param.input_size = input  .shape_[1];
                _param.state_size = state_h.shape_[1];

                // here we require double the amount of size
                _temp_space_size = _param.batch_size * 6 * _param.state_size;
                std::cout << "Temp Space Size: " << _temp_space_size 
                          << std::endl;
                _initialized = true;
        }
public:
        virtual void Forward(const OpContext & ctx,
                             const std::vector < TBlob > &  in_data,
                             const std::vector < OpReqType > &  req,
                             const std::vector < TBlob > & out_data,
                             const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t in_expected = 6, out_expected = 4;

                CHECK_EQ( in_data.size(),  in_expected); ///< input, state_h
                                                         ///< i2h_weight, i2h_bias
                                                         ///< h2h_weight, h2h_bias
                CHECK_EQ(out_data.size(), out_expected); ///< state_h_out
                                                         ///< feature_map_reset_gate
                                                         ///< feature_map_h2h
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
                Tensor < gpu, 2, DType > feature_map_h2h = 
                        out_data[int(EnumOpOutputs::FeatureMapH2H)]
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
                CHECK_EQ(state_h_out.CheckContiguous(), true);
                CHECK_EQ(feature_map_reset_gate .CheckContiguous(), true);
                CHECK_EQ(feature_map_h2h        .CheckContiguous(), true);
                CHECK_EQ(feature_map_update_gate.CheckContiguous(), true);

                if (!_initialized)
                {
                        _Init(cuda_stream, in_data, out_data);
                }

                Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
                        .get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);
                
                const unsigned BxH = _param.batch_size * _param.state_size;

                FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 input  .dptr_, i2h_weight.dptr_, 
                                 workspace.dptr_,
				 OpReqType::kWriteTo,
				_param.batch_size,
				_param.input_size,
				_param.state_size * 3);
		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 state_h.dptr_, h2h_weight.dptr_, 
                                 workspace.dptr_ + 3 * BxH,
				 OpReqType::kWriteTo,
				_param.batch_size,
				_param.state_size,
				_param.state_size * 3);
                _cuda_gru_cell_forward < DType > <<<
                        (BxH - 1) / 128 + 1, 128, 0,
                        Stream < gpu > ::GetStream(cuda_stream) >>> (
                        workspace.dptr_,
                        workspace.dptr_ + 3 * BxH,
                        i2h_bias.dptr_,
			h2h_bias.dptr_,
                        state_h.dptr_,
			feature_map_reset_gate .dptr_,
                        feature_map_h2h        .dptr_,
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
                using namespace mshadow;

                std::size_t in_expected = 6, out_expected = 4, visible_out_expected = 1;

                /// input, state_h, i2h_weight, i2h_bias
                ///                 h2h_weight, h2h_bias
                CHECK_EQ( in_data.size(),  in_expected);
                CHECK_EQ( in_grad.size(),  in_expected);
                CHECK_EQ(     req.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected);          ///< state_h_out
                                                                  ///< feature_map_reset_gate
                                                                  ///< feature_map_h2h
                                                                  ///< feature_map_update_gate
                CHECK_EQ(out_grad.size(), visible_out_expected);  ///< state_h_out

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                Tensor < gpu, 2, DType > input_grad      = in_grad[int(EnumOpInputs::Input)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > state_h_grad    = in_grad[int(EnumOpInputs::StateH)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > i2h_weight_grad = in_grad[int(EnumOpInputs::I2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 1, DType > i2h_bias_grad   = in_grad[int(EnumOpInputs::I2HBias)]
                        .get < gpu, 1, DType > (cuda_stream);
                Tensor < gpu, 2, DType > h2h_weight_grad = in_grad[int(EnumOpInputs::H2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 1, DType > h2h_bias_grad   = in_grad[int(EnumOpInputs::H2HBias)]
                        .get < gpu, 1, DType > (cuda_stream);
                
                Tensor < gpu, 2, DType > input      = in_data[int(EnumOpInputs::Input)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > state_h    = in_data[int(EnumOpInputs::StateH)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > i2h_weight = in_data[int(EnumOpInputs::I2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > h2h_weight = in_data[int(EnumOpInputs::H2HWeight)]
                        .get < gpu, 2, DType > (cuda_stream);
                
                Tensor < gpu, 2, DType > state_h_out = out_data[int(EnumOpOutputs::StateHOut)]
                        .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > feature_map_reset_gate = 
                        out_data[int(EnumOpOutputs::FeatureMapResetGate)]
                                .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > feature_map_h2h = 
                        out_data[int(EnumOpOutputs::FeatureMapH2H)]
                                .get < gpu, 2, DType > (cuda_stream);
                Tensor < gpu, 2, DType > feature_map_update_gate = 
                        out_data[int(EnumOpOutputs::FeatureMapUpdateGate)]
                                .get < gpu, 2, DType > (cuda_stream);
                
                Tensor < gpu, 2, DType > state_h_out_grad = 
                        out_grad[int(EnumOpOutputs::StateHOut)]
                                .get < gpu, 2, DType > (cuda_stream);
                
                CHECK_EQ(input           .CheckContiguous(), true);
                CHECK_EQ(input_grad      .CheckContiguous(), true);
                CHECK_EQ(state_h         .CheckContiguous(), true);
                CHECK_EQ(state_h_grad    .CheckContiguous(), true);
                CHECK_EQ(i2h_weight      .CheckContiguous(), true);
                CHECK_EQ(i2h_weight_grad .CheckContiguous(), true);
                CHECK_EQ(i2h_bias_grad   .CheckContiguous(), true);
                CHECK_EQ(h2h_weight      .CheckContiguous(), true);
                CHECK_EQ(h2h_weight_grad .CheckContiguous(), true);
                CHECK_EQ(h2h_bias_grad   .CheckContiguous(), true);
                CHECK_EQ(state_h_out     .CheckContiguous(), true);
                CHECK_EQ(state_h_out_grad.CheckContiguous(), true);

                Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
                        .get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);
                
                const unsigned BxH = _param.batch_size * _param.state_size;

                _cuda_gru_cell_backward < DType > <<<
                        (BxH - 1) / 128 + 1, 128, 0, 
                        Stream < gpu > ::GetStream(cuda_stream) >>> (
                        state_h_grad.dptr_,
                        workspace.dptr_,
                        workspace.dptr_ + 3 * BxH, 
                        i2h_bias_grad.dptr_,
                        h2h_bias_grad.dptr_,
                        feature_map_reset_gate .dptr_,
                        feature_map_h2h        .dptr_,
                        feature_map_update_gate.dptr_,
                        state_h         .dptr_,
                        state_h_out     .dptr_,
                        state_h_out_grad.dptr_,
                        _param.batch_size, 
                        _param.state_size);
                

                FullyConnectedBWWeight(Stream < gpu > ::GetBlasHandle(cuda_stream),
				       input  .dptr_, i2h_weight_grad.dptr_,
                                       workspace.dptr_,
				       req[int(EnumOpInputs::I2HWeight)],
				       _param.batch_size,
				       _param.input_size,
				       3 * _param.state_size);
		FullyConnectedBWWeight(Stream < gpu > ::GetBlasHandle(cuda_stream),
				       state_h.dptr_, h2h_weight_grad.dptr_, 
                                       workspace.dptr_ + 3 * BxH,
				       req[int(EnumOpInputs::H2HWeight)],
				       _param.batch_size,
				       _param.state_size,
				       3 * _param.state_size);
		FullyConnectedBWData  (Stream < gpu > ::GetBlasHandle(cuda_stream),
				       input_grad  .dptr_, i2h_weight.dptr_, workspace.dptr_,
				       req[int(EnumOpInputs::Input)],
				       _param.batch_size,
				       _param.input_size,
				       3 * _param.state_size);
		FullyConnectedBWData  (Stream < gpu > ::GetBlasHandle(cuda_stream),
				       state_h_grad.dptr_, h2h_weight.dptr_,
                                       workspace.dptr_ + 3 * BxH,
				       req[int(EnumOpInputs::StateH)],
				       _param.batch_size,
				       _param.state_size,
				       3 * _param.state_size);
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
              RealType * const __restrict__ feature_map_h2h,
              RealType * const __restrict__ feature_map_update_gate,
              RealType * const __restrict__ state_h_out,
        const unsigned batch_size, const unsigned state_size, const bool is_train)
{
        const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
                BxH = batch_size * state_size;

        if (g_threadIdx >= BxH) { return ; }

        const unsigned workspace_idx    = (g_threadIdx / state_size) * 3 * state_size + 
                                          (g_threadIdx % state_size),
                       workspace_stride = state_size;
        /*
        RealType reset_gate = __cu_sigmoid(
                workspace_i2h[workspace_idx + 0 * workspace_stride] + 
                workspace_h2h[workspace_idx + 0 * workspace_stride] + 
                i2h_bias[g_threadIdx % state_size + 0 * state_size] + 
                h2h_bias[g_threadIdx % state_size + 0 * state_size]);
        RealType update_gate = __cu_sigmoid(
                workspace_i2h[workspace_idx + 1 * workspace_stride] + 
                workspace_h2h[workspace_idx + 1 * workspace_stride] + 
                i2h_bias[g_threadIdx % state_size + 1 * state_size] + 
                h2h_bias[g_threadIdx % state_size + 1 * state_size]);
        RealType h2h = workspace_h2h[workspace_idx + 2 * workspace_stride] + 
                       h2h_bias[g_threadIdx % state_size + 2 * state_size];
        RealType state_h_out_tmp = tanh(
                workspace_i2h[workspace_idx + 2 * workspace_stride] + 
                i2h_bias[g_threadIdx % state_size + 2 * state_size] + reset_gate * h2h);
        
        state_h_out[g_threadIdx] = (1 - update_gate) * state_h_out_tmp + 
                                        update_gate  * state_h[g_threadIdx];
         */
        state_h_out[g_threadIdx] = 
                // workspace_i2h[workspace_idx + 2 * workspace_stride] + 
                // i2h_bias[g_threadIdx % state_size + 2 * state_size] + 
                // workspace_h2h[workspace_idx + 2 * workspace_stride] + 
                // h2h_bias[g_threadIdx % state_size + 2 * state_size];
                workspace_i2h[workspace_idx + 2 * workspace_stride] + 
                workspace_h2h[workspace_idx + 2 * workspace_stride];

        if (is_train)
        {
                // feature_map_reset_gate [g_threadIdx] = reset_gate;
                // feature_map_h2h        [g_threadIdx] = h2h;
                // feature_map_update_gate[g_threadIdx] = update_gate;
        }
}

template < typename RealType >
__global__ void _cuda_gru_cell_backward(
              RealType * const __restrict__ state_h_grad,
              RealType * const __restrict__ workspace_i2h,
              RealType * const __restrict__ workspace_h2h,
              RealType * const __restrict__ i2h_bias_grad,
              RealType * const __restrict__ h2h_bias_grad,
        const RealType * const __restrict__ feature_map_reset_gate,
        const RealType * const __restrict__ feature_map_h2h,
        const RealType * const __restrict__ feature_map_update_gate,
        const RealType * const __restrict__ state_h,
        const RealType * const __restrict__ state_h_out,
        const RealType * const __restrict__ state_h_out_grad,
        const unsigned batch_size, const unsigned state_size)
{
        const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
                BxH = batch_size * state_size;
        
        if (g_threadIdx >= BxH) { return ; }

        const unsigned workspace_idx    = (g_threadIdx / state_size) * 3 * state_size + 
                                          (g_threadIdx % state_size),
                       workspace_stride = state_size;

        RealType reset_gate  = feature_map_reset_gate [g_threadIdx];
        RealType h2h         = feature_map_h2h        [g_threadIdx];
        RealType update_gate = feature_map_update_gate[g_threadIdx];

        RealType state_h_out_tmp = (update_gate == 1) ? 0 : (
                 state_h_out[g_threadIdx] - 
                 state_h    [g_threadIdx] * update_gate) / (1 - update_gate);
        
        // `update_gate`
        RealType update_gate_grad = 
                state_h_out_grad[g_threadIdx] * state_h[g_threadIdx] - 
                state_h_out_grad[g_threadIdx] * state_h_out_tmp;
        update_gate_grad *= update_gate * (1 - update_gate);
        // `state_h_out_tmp`
        RealType state_h_out_tmp_grad = (1 - update_gate) * 
                 state_h_out_grad[g_threadIdx] * 
                (1 - state_h_out_tmp * state_h_out_tmp);
        // `reset_gate`
        RealType reset_gate_grad = state_h_out_tmp_grad * h2h * 
                 reset_gate * (1 - reset_gate);
        // `state_h` (previous time step)
        state_h_grad[g_threadIdx] += state_h_out_grad[g_threadIdx] * update_gate;

        workspace_i2h[workspace_idx + 0 * workspace_stride] = reset_gate_grad;
        workspace_h2h[workspace_idx + 0 * workspace_stride] = reset_gate_grad;
        workspace_i2h[workspace_idx + 1 * workspace_stride] = update_gate_grad;
        workspace_h2h[workspace_idx + 1 * workspace_stride] = update_gate_grad;
        workspace_i2h[workspace_idx + 2 * workspace_stride] = state_h_out_tmp_grad;
        workspace_h2h[workspace_idx + 2 * workspace_stride] = state_h_out_tmp_grad * reset_gate;

        atomicAdd(&i2h_bias_grad[g_threadIdx % state_size + 0 * state_size],  reset_gate_grad);
        atomicAdd(&h2h_bias_grad[g_threadIdx % state_size + 0 * state_size],  reset_gate_grad);
        atomicAdd(&i2h_bias_grad[g_threadIdx % state_size + 1 * state_size], update_gate_grad);
        atomicAdd(&h2h_bias_grad[g_threadIdx % state_size + 1 * state_size], update_gate_grad);
        atomicAdd(&i2h_bias_grad[g_threadIdx % state_size + 2 * state_size], state_h_out_tmp_grad);
        atomicAdd(&h2h_bias_grad[g_threadIdx % state_size + 2 * state_size], state_h_out_tmp_grad * reset_gate);
}

template <>
inline void FullyConnectedFW < float > (cublasHandle_t cublas_handle,
	const float * const __restrict__ X,
	const float * const __restrict__ W,
	      float * const __restrict__ Y,
	const OpReqType req,       const unsigned batch_size, 
	const unsigned input_size, const unsigned state_size)
{
	float alpha = 1.0, beta = float(req == kAddTo);
        CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle
                                CUBLAS_OP_T, // W.T
                                CUBLAS_OP_N, // X
                                state_size,  // Y.shape[1]
                                batch_size,  // Y.shape[0]
                                input_size,  // W.shape[1]
                                &alpha, W, input_size, X, input_size,
                                &beta,  Y, state_size));
}

template <>
inline void FullyConnectedBWWeight < float > (cublasHandle_t cublas_handle,
	const float * const __restrict__  X,
	      float * const __restrict__ dW,
	const float * const __restrict__ dY,
	const OpReqType grad_req,   const unsigned batch_size,
	const unsigned  input_size, const unsigned state_size)
{
	float alpha = 1.0, beta = float(grad_req == kAddTo);
	CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle
	                        CUBLAS_OP_N, //  X
				CUBLAS_OP_T, // dY^T
				input_size,  // dW.shape[1]
				state_size,  // dW.shape[0]
				batch_size,  //  X.shape[0]
	                        &alpha,  X, input_size, dY, state_size,
				&beta,  dW, input_size));
}
template <>
inline void FullyConnectedBWData < float > (cublasHandle_t cublas_handle,
	      float * const __restrict__ dX,
	const float * const __restrict__  W,
	const float * const __restrict__ dY, 
	const OpReqType grad_req,   const unsigned batch_size,
	const unsigned  input_size, const unsigned state_size)
{
	float alpha = 1.0, beta = float(grad_req == kAddTo);
	CUBLAS_CALL(cublasSgemm(cublas_handle, //cuBLAS Handle
	                        CUBLAS_OP_N, //  W
				CUBLAS_OP_N, // dY
				input_size,  // dX.shape[1]
				batch_size,  // dX.shape[0]
				state_size,  //  W.shape[0]
				&alpha,  W, input_size, dY, state_size,
				&beta,  dX, input_size));
}

#endif  // defined(__CUDACC__)

        }  // namespace op
}  // namespace mxnet
