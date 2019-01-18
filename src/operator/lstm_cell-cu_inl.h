
#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "lstm_cell-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

/**
 * Forward Pass of the LSTM Cell
 * This kernel shall be launched using the parameter <<< ceil(BxH / 128), 128, 0, cuda_stream >>>.
 * @param1 workspace      [B x 4H]
 * @param1 i2h_bias           [4H]
 * @param2 h2h_bias           [4H]
 * @param3 state_c        [B x  H]
 * @param4 reserved_space [B x 2H]
 * @param5 state_h_out    [B x  H]
 * @param6 state_c_out    [B x  H]
 * @param7 batch_size: (Parameter)
 * @param8 state_size: (Parameter)
 */
template < typename RealType >
__global__ void _cuda_lstm_cell__forward(
	const RealType * const __restrict__ workspace,
	const RealType * const __restrict__ i2h_bias,
	const RealType * const __restrict__ h2h_bias,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ reserved_space,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out,
	const unsigned batch_size, const unsigned state_size);

/**
 * Backward Pass of the LSTM Cell
 * This kernel shall be launched using the parameter <<< ceil(BxH / 128), 128, 0, cuda_stream >>>.
 * @param1  workspace_grad   [B x 4H]
 * @param2  i2h_bias_grad        [4H]
 * @param3  h2h_bias_grad        [4H]
 * @param4  state_c_grad     [B x  H]
 * @param5  reserved_space   [B x 2H]
 * @param6  state_c          [B x  H]
 * @param7  state_h_out      [B x  H]
 * @param8  state_c_out      [B x  H]
 * @param9  state_h_out_grad [B x  H]
 * @param10 state_c_out_grad [B x  H]
 * @param11 batch_size: (Parameter)
 * @param12 state_size: (Parameter)
 */
template < typename RealType >
__global__ void _cuda_lstm_cell_backward(
	      RealType * const __restrict__ state_c_grad,
	      RealType * const __restrict__ reserved_space,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ state_h_out,
	const RealType * const __restrict__ state_c_out,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size);

// FullyConnected Layer Y = X W^T Forward Pass
// @param1 X [batch_size x input_size]
// @param2 W [state_size x input_size]
// @param3 Y [batch_size x state_size]
// @param4 batch_size: (Parameter)
// @param5 input_size: (Parameter)
// @param6 state_size: (Parameter)
template < typename RealType >
static inline void FullyConnectedFW(cublasHandle_t cublas_handle,
	const RealType * const __restrict__ X,
	const RealType * const __restrict__ W,
	      RealType * const __restrict__ Y,
	const OpReqType req,       const unsigned batch_size, 
	const unsigned input_size, const unsigned state_size);

// FullyConnected Layer Y = XW^T Backward Pass on Weight (dW = dY^T X)
// @param1  X [batch_size x input_size]
// @param2 dW [state_size x input_size]
// @param3 dY [batch_size x state_size]
// @param4 batch_size: (Parameter)
// @param5 input_size: (Parameter)
// @param6 state_size: (Parameter)
template < typename RealType >
static inline void FullyConnectedBWWeight(cublasHandle_t cublas_handle,
	const RealType * const __restrict__  X,
	      RealType * const __restrict__ dW,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,   const unsigned batch_size, 
	const unsigned  input_size, const unsigned state_size);

// FullyConnected Layer Y = XW^T Backward Pass on Data (dX = dY W)
// @param1 dX [batch_size x input_size]
// @param2  W [state_size x input_size]
// @param3 dY [batch_size x state_size]
// @param4 batch_size: (Parameter) 
// @param5 input_size: (Parameter) 
// @param6 state_size: (Parameter)
template < typename RealType >
static inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      RealType * const __restrict__ dX,
	const RealType * const __restrict__  W,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,   const unsigned batch_size, 
	const unsigned  input_size, const unsigned state_size);

template < typename DType >
class CUEcoLSTMCellOp : public Operator
{
private:
	EcoLSTMCellParam _param;

	bool _initialized = false;

	Storage::Handle _reserved_space; unsigned _temp_space_size;
public:
	explicit CUEcoLSTMCellOp(EcoLSTMCellParam param)
	{
		_param = param;
	}
	~CUEcoLSTMCellOp()
	{
		if (_initialized)
		{
			Storage::Get()->Free(_reserved_space);
		}
	}

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

		// infer the parameters from the cell input and hidden state
		_param.batch_size = input  .shape_[0];
		_param.input_size = input  .shape_[1];
		_param.state_size = state_h.shape_[2];

		// allocate the reserved space [B x 2H]
		_reserved_space = Storage::Get()->Alloc(_param.batch_size * 2 * _param.state_size * sizeof(DType),
		                                        Context::GPU());

		// allocate the workspace size
		_temp_space_size = _param.batch_size * 4 * _param.state_size;

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

		std::size_t in_expected = 7, out_expected = 2;

		// input, state_h, state_c
		// i2h_weight, i2h_bias
		// h2h_weight, h2h_bias
		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 2, DType > input      = in_data[int(EnumOpInputs::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h    = in_data[int(EnumOpInputs::StateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c    = in_data[int(EnumOpInputs::StateC)]
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
		Tensor < gpu, 2, DType > state_c_out = out_data[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(input      .CheckContiguous(), true);
		CHECK_EQ(state_h    .CheckContiguous(), true);
		CHECK_EQ(state_c    .CheckContiguous(), true);
		CHECK_EQ(i2h_weight .CheckContiguous(), true);
		CHECK_EQ(i2h_bias   .CheckContiguous(), true);
		CHECK_EQ(h2h_weight .CheckContiguous(), true);
		CHECK_EQ(h2h_bias   .CheckContiguous(), true);
		CHECK_EQ(state_h_out.CheckContiguous(), true);
		CHECK_EQ(state_c_out.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}

		Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
			.get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);

		const unsigned BxH = _param.batch_size * _param.state_size;

		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 input,   i2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kWriteTo,
				 _param.batch_size,
				 _param.input_size,
				 _param.state_size * 4);
		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 state_h, h2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kAddTo,
				 _param.batch_size,
				 _param.state_size,
				 _param.state_size * 4);
		
		_cuda_lstm_nonlin_block__forward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				workspace.dptr_,
				state_c.dptr_,
				ctx.is_train ? reinterpret_cast < DType * > (_reserved_space.dptr) : nullptr,
				state_h_out.dptr_,
				state_c_out.dptr_,
				_param.batch_size, _param.state_size
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

		std::size_t in_expected = 7, out_expected = 2;

		// input, state_h, state_c
		// i2h_weight, i2h_bias
		// h2h_weight, h2h_bias
		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out
		CHECK_EQ(out_grad.size(), out_expected);

		// The gradients requests of the input variables can be anything, but most likely kAddTo.

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 2, DType > input_grad      = in_grad[int(EnumOpInputs::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h_grad    = in_grad[int(EnumOpInputs::StateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_grad    = in_grad[int(EnumOpInputs::StateC)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i2h_weight_grad = in_grad[int(EnumOpInputs::I2HWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 1, DType > i2h_bias_grad   = in_grad[int(EnumOpInputs::I2HBias)]
			.get < gpu, 1, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2h_weight_grad = in_grad[int(EnumOpInputs::H2HWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 1, DType > h2h_bias_grad   = in_grad[int(EnumOpInputs::H2HBias)]
			.get < gpu, 1, DType > (cuda_stream);
		
		Tensor < gpu, 2, DType > input       =  in_data[int(EnumOpInputs ::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h     =  in_data[int(EnumOpInputs ::StateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c     =  in_data[int(EnumOpInputs ::StateC)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h_out = out_data[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out = out_data[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);
		
		Tensor < gpu, 2, DType > state_h_out_grad = out_grad[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out_grad = out_grad[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(input           .CheckContiguous(), true);
		CHECK_EQ(input_grad      .CheckContiguous(), true);
		CHECK_EQ(state_h         .CheckContiguous(), true);
		CHECK_EQ(state_h_grad    .CheckContiguous(), true);
		CHECK_EQ(state_c         .CheckContiguous(), true);
		CHECK_EQ(state_c_grad    .CheckContiguous(), true);
		CHECK_EQ(i2h_weight_grad .CheckContiguous(), true);
		CHECK_EQ(i2h_bias_grad   .CheckContiguous(), true);
		CHECK_EQ(h2h_weight_grad .CheckContiguous(), true);
		CHECK_EQ(h2h_bias_grad   .CheckContiguous(), true);
		CHECK_EQ(state_h_out     .CheckContiguous(), true);
		CHECK_EQ(state_h_out_grad.CheckContiguous(), true);
		CHECK_EQ(state_c_out     .CheckContiguous(), true);
		CHECK_EQ(state_c_out_grad.CheckContiguous(), true);

		Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
			.get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);

		const unsigned BxH = _param.batch_size * _param.state_size;
	}
};

template < typename RealType >
static __forceinline__ __device__ RealType __cu_sigmoid(RealType i)
{
	return 1.0 / (1.0 + exp(-i));
}

template < typename RealType >
__global__ void _cuda_lstm_cell__forward(
	const RealType * const __restrict__ workspace,
	const RealType * const __restrict__ i2h_bias,
	const RealType * const __restrict__ h2h_bias,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ reserved_space,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out,
	const unsigned batch_size, const unsigned state_size)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;

	RealType  input_gate = __cu_sigmoid(workspace[batch_idx_x4H_plus_state_idx + 0 * state_size]);
	RealType forget_gate = __cu_sigmoid(workspace[batch_idx_x4H_plus_state_idx + 1 * state_size]);
	RealType  input_actv =         tanh(workspace[batch_idx_x4H_plus_state_idx + 2 * state_size]);
	RealType output_gate = __cu_sigmoid(workspace[batch_idx_x4H_plus_state_idx + 3 * state_size]);

	RealType state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;

	state_c_out[g_threadIdx] =      state_c_out_reg;
	state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	if (reserved_space != nullptr)
	{
		// preserve the gates to be used in the backward pass
		reserved_space[batch_idx_x4H_plus_state_idx + 0 * state_size] =  input_gate;
		reserved_space[batch_idx_x4H_plus_state_idx + 1 * state_size] = forget_gate;
	}
}

template <>
inline void FullyConnectedFW < float > (cublasHandle_t cublas_handle,
	const float * const __restrict__ X,
	const float * const __restrict__ W,
	      float * const __restrict__ Y,
	const OpReqType & req,     const unsigned batch_size, 
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

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet