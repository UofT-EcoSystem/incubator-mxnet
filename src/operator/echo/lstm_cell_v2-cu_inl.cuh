#pragma once

#include "lstm_cell_v2-inl.h"

namespace mxnet {
	namespace op {

/**
 * Forward Pass of the LSTM Cell
 * 
 * @param1  workspace   [B x 4H]
 * @param1  i2h_bias        [4H]
 * @param2  h2h_bias        [4H]
 * @param3  state_c     [B x  H]
 * @param4  input_fm    [B x  H]
 * @param5  forget_fm   [B x  H]
 * @param6  state_h_out [B x  H]
 * @param7  state_c_out [B x  H]
 * @param8  batch_size: (Parameter)
 * @param9  state_size: (Parameter)
 * @param10 is_train: (Runtime Parameter)
 * @param11 batch_minor: (Runtime Parameter)
 */
template < typename RealType >
__global__ void cudaLSTMCellForward(
	const RealType * const __restrict__ workspace,
	const RealType * const __restrict__ i2h_bias,
	const RealType * const __restrict__ h2h_bias,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ input_fm,
	      RealType * const __restrict__ forget_fm,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out,
	const unsigned batch_size, const unsigned state_size, 
	const bool is_train, const bool batch_minor);

/**
 * Backward Pass of the LSTM Cell
 * 
 * @param1  workspace        [B x 4H]
 * @param2  bias_grad            [4H]
 * @param3  state_c_grad     [B x  H]
 * @param4  input_fm         [B x  H]
 * @param5  forget_fm        [B x  H]
 * @param6  state_c          [B x  H]
 * @param7  state_h_out      [B x  H]
 * @param8  state_c_out      [B x  H]
 * @param9  state_h_out_grad [B x  H]
 * @param10 state_c_out_grad [B x  H]
 * @param11 grad_req: (Parameter)
 * @param12 batch_size: (Parameter)
 * @param13 state_size: (Parameter)
 */
template < typename RealType >
__global__ void cudaLSTMCellBackward(
	      RealType * const __restrict__ workspace,
	      RealType * const __restrict__ bias_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ input_fm,
	const RealType * const __restrict__ forget_fm,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ state_h_out,
	const RealType * const __restrict__ state_c_out,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size);

// FullyConnected Layer $Y = X W^T$ Forward Pass
// @param1 X [batch_size x input_size]
// @param2 W [state_size x input_size]
// @param3 Y [batch_size x state_size]
// @param4 batch_size (parameter)
// @param5 input_size (parameter)
// @param6 state_size (parameter)
template < typename RealType >
static inline void FullyConnectedFW(cublasHandle_t cublas_handle,
	const RealType * const __restrict__ X,
	const RealType * const __restrict__ W,
	      RealType * const __restrict__ Y,
	const OpReqType req,
	const unsigned batch_size, 
	const unsigned input_size,
	const unsigned state_size, 
	const bool batch_minor);

// FullyConnected Layer $Y = X W^T$ Backward Pass on Weight ($dW = dY^T X$)
// @param1  X [batch_size x input_size]
// @param2 dW [state_size x input_size]
// @param3 dY [batch_size x state_size]
// @param4 batch_size (parameter)
// @param5 input_size (parameter)
// @param6 state_size (parameter)
template < typename RealType >
static inline void FullyConnectedBWWeight(cublasHandle_t cublas_handle,
	const RealType * const __restrict__  X,
	      RealType * const __restrict__ dW,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,  const unsigned batch_size, 
	const unsigned input_size, const unsigned state_size);

// FullyConnected Layer $Y = X W^T$ Backward Pass on Data ($dX = dY W$)
// @param1 dX [batch_size x input_size]
// @param2  W [state_size x input_size]
// @param3 dY [batch_size x state_size]
// @param4 batch_size (parameter) 
// @param5 input_size (parameter) 
// @param6 state_size (parameter)
template < typename RealType >
static inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      RealType * const __restrict__ dX,
	const RealType * const __restrict__  W,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req,  const unsigned batch_size, 
	const unsigned input_size, const unsigned state_size);

template < typename DType >
class CULSTMCellV2Op : public Operator
{
private:
	LSTMCellV2Param _param;
	bool _initialized = false;
	bool _batch_minor = false;
	unsigned _temp_space_size;
public:
	explicit CULSTMCellV2Op(LSTMCellV2Param param)
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

		Tensor < gpu, 2, DType > input   = in_data[int(EnumOpInputs::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h = in_data[int(EnumOpInputs::StateH)]
			.get < gpu, 2, DType > (cuda_stream);

		// infer the parameters from the cell input and hidden state
		_param.batch_size = input  .shape_[0];
		_param.input_size = input  .shape_[1];
		_param.state_size = state_h.shape_[1];

		// allocate the workspace size
		_temp_space_size = _param.batch_size * 4 * _param.state_size;

		const char * batch_minor = getenv("USE_BATCH_MINOR_FW");

		if (batch_minor == nullptr)
		{
			_batch_minor = false;
		}
		else if (!strcmp(batch_minor, "0"))
		{
			_batch_minor = false;
		}
		else
		{
			// LOG(INFO) << "MXNet will be using batch_minor "
			// 	"for the Fully-Connected layers.";
			_batch_minor = true;
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

		std::size_t in_expected = 7, out_expected = 4;

		// input, state_h, state_c
		// i2h_weight, i2h_bias
		// h2h_weight, h2h_bias
		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out
		                                         // input_fm, forget_fm

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
		Tensor < gpu, 2, DType >    input_fm = out_data[int(EnumOpOutputs::InputFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   forget_fm = out_data[int(EnumOpOutputs::ForgetFM)]
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
		                 input  .dptr_, i2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kWriteTo,
				 _param.batch_size,
				 _param.input_size,
				 _param.state_size * 4,
				 _batch_minor);
		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 state_h.dptr_, h2h_weight.dptr_, workspace.dptr_,
				 OpReqType::kAddTo,
				 _param.batch_size,
				 _param.state_size,
				 _param.state_size * 4,
				 _batch_minor);
		
		cudaLSTMCellForward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, 
				Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				workspace.dptr_,
				i2h_bias.dptr_,
				h2h_bias.dptr_,
				state_c.dptr_,
				input_fm.dptr_,
				forget_fm.dptr_,
				state_h_out.dptr_,
				state_c_out.dptr_,
				_param.batch_size, _param.state_size, 
				ctx.is_train, _batch_minor
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

		std::size_t in_expected = 7, out_expected = 4, visible_out_expected = 2;

		// input, state_h, state_c
		// i2h_weight, i2h_bias
		// h2h_weight, h2h_bias
		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out, input_fm, forget_fm
		CHECK_EQ(out_grad.size(), visible_out_expected); // state_h_out, state_c_out,

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
		
		Tensor < gpu, 2, DType > input      = in_data[int(EnumOpInputs::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h    = in_data[int(EnumOpInputs::StateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c    = in_data[int(EnumOpInputs::StateC)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i2h_weight = in_data[int(EnumOpInputs::I2HWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2h_weight = in_data[int(EnumOpInputs::H2HWeight)]
			.get < gpu, 2, DType > (cuda_stream);

		Tensor < gpu, 2, DType > state_h_out = out_data[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out = out_data[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >    input_fm = out_data[int(EnumOpOutputs::InputFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   forget_fm = out_data[int(EnumOpOutputs::ForgetFM)]
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
		CHECK_EQ(i2h_weight      .CheckContiguous(), true);
		CHECK_EQ(i2h_weight_grad .CheckContiguous(), true);
		CHECK_EQ(i2h_bias_grad   .CheckContiguous(), true);
		CHECK_EQ(h2h_weight      .CheckContiguous(), true);
		CHECK_EQ(h2h_weight_grad .CheckContiguous(), true);
		CHECK_EQ(h2h_bias_grad   .CheckContiguous(), true);
		CHECK_EQ(state_h_out     .CheckContiguous(), true);
		CHECK_EQ(state_h_out_grad.CheckContiguous(), true);
		CHECK_EQ(state_c_out     .CheckContiguous(), true);
		CHECK_EQ(state_c_out_grad.CheckContiguous(), true);

		Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
			.get_space_typed < gpu, 1, DType > (Shape1(_temp_space_size), cuda_stream);

		const unsigned BxH = _param.batch_size * _param.state_size;
	
		CHECK_EQ(req[int(EnumOpInputs::I2HBias)], req[int(EnumOpInputs::H2HBias)]);

		if (req[int(EnumOpInputs::I2HBias)] == OpReqType::kWriteTo)
		{
			CUDA_CALL(cudaMemsetAsync(i2h_bias_grad.dptr_, 0, 
				4 * _param.state_size * sizeof(DType), 
				Stream < gpu > ::GetStream(cuda_stream)));
		}

		cudaLSTMCellBackward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, 
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				workspace.dptr_,
				i2h_bias_grad.dptr_,
				state_c_grad.dptr_,
				input_fm.dptr_,
				forget_fm.dptr_,
				state_c.dptr_,
				state_h_out.dptr_,
				state_c_out.dptr_,
				state_h_out_grad.dptr_,
				state_c_out_grad.dptr_,
				_param.batch_size, _param.state_size
			);

		CUDA_CALL(cudaMemcpyAsync(h2h_bias_grad.dptr_, i2h_bias_grad.dptr_, 
			4 * _param.state_size * sizeof(DType), 
			cudaMemcpyDeviceToDevice, Stream < gpu > ::GetStream(cuda_stream)));
		
		FullyConnectedBWWeight(Stream < gpu > ::GetBlasHandle(cuda_stream),
				       input  .dptr_, i2h_weight_grad.dptr_, workspace.dptr_,
				       req[int(EnumOpInputs::I2HWeight)],
				       _param.batch_size,
				       _param.input_size,
				       4 * _param.state_size);
		FullyConnectedBWWeight(Stream < gpu > ::GetBlasHandle(cuda_stream),
				       state_h.dptr_, h2h_weight_grad.dptr_, workspace.dptr_,
				       req[int(EnumOpInputs::H2HWeight)],
				       _param.batch_size,
				       _param.state_size,
				       4 * _param.state_size);
		FullyConnectedBWData  (Stream < gpu > ::GetBlasHandle(cuda_stream),
				       input_grad  .dptr_, i2h_weight.dptr_, workspace.dptr_,
				       req[int(EnumOpInputs::Input)],
				       _param.batch_size,
				       _param.input_size,
				       4 * _param.state_size);
		FullyConnectedBWData  (Stream < gpu > ::GetBlasHandle(cuda_stream),
				       state_h_grad.dptr_, h2h_weight.dptr_, workspace.dptr_,
				       req[int(EnumOpInputs::StateH)],
				       _param.batch_size,
				       _param.state_size,
				       4 * _param.state_size);
	}
};

template < typename RealType >
static __forceinline__ __device__ RealType __cu_sigmoid(RealType i)
{
	return 1.0 / (1.0 + exp(-i));
}

template < typename RealType >
__global__ void cudaLSTMCellForward(
	const RealType * const __restrict__ workspace,
	const RealType * const __restrict__ i2h_bias,
	const RealType * const __restrict__ h2h_bias,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ input_fm,
	      RealType * const __restrict__ forget_fm,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out,
	const unsigned batch_size, const unsigned state_size, 
	const bool is_train, const bool batch_minor)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	unsigned workspace_idx, workspace_stride;
	
	if (batch_minor)
	{
		workspace_idx    = (g_threadIdx % state_size) * batch_size +
		                   (g_threadIdx / state_size);
		workspace_stride = batch_size * state_size;
	}
	else
	{
		workspace_idx    = (g_threadIdx / state_size) * 4 * state_size + 
		                   (g_threadIdx % state_size);
		workspace_stride = state_size;
	}

	RealType  input_gate = __cu_sigmoid(workspace[workspace_idx + 0 * workspace_stride] + 
		i2h_bias[g_threadIdx % state_size + 0 * state_size] + 
		h2h_bias[g_threadIdx % state_size + 0 * state_size]);
	RealType forget_gate = __cu_sigmoid(workspace[workspace_idx + 1 * workspace_stride] + 
		i2h_bias[g_threadIdx % state_size + 1 * state_size] + 
		h2h_bias[g_threadIdx % state_size + 1 * state_size]);
	RealType  input_actv =         tanh(workspace[workspace_idx + 2 * workspace_stride] + 
		i2h_bias[g_threadIdx % state_size + 2 * state_size] + 
		h2h_bias[g_threadIdx % state_size + 2 * state_size]);
	RealType output_gate = __cu_sigmoid(workspace[workspace_idx + 3 * workspace_stride] + 
		i2h_bias[g_threadIdx % state_size + 3 * state_size] + 
		h2h_bias[g_threadIdx % state_size + 3 * state_size]);

	RealType state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;

	state_c_out[g_threadIdx] =      state_c_out_reg;
	state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	if (is_train)
	{
		// preserve the gates to be used in the backward pass
		input_fm [g_threadIdx] = input_gate;
		forget_fm[g_threadIdx] = forget_gate;
	}
}

template < typename RealType >
__global__ void cudaLSTMCellBackward(
	      RealType * const __restrict__ workspace,
	      RealType * const __restrict__ bias_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ input_fm,
	const RealType * const __restrict__ forget_fm,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ state_h_out,
	const RealType * const __restrict__ state_c_out,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;
	
	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;
	
	RealType input_gate  = input_fm [g_threadIdx];
	RealType forget_gate = forget_fm[g_threadIdx];
	
	RealType state_c_reg     = state_c    [g_threadIdx];
	RealType state_c_out_reg = state_c_out[g_threadIdx];

	RealType input_actv = input_gate == 0 ? 
		0 : (state_c_out_reg - forget_gate * state_c_reg) / input_gate;
	RealType state_c_out_actv = tanh(state_c_out_reg);
	RealType output_gate = state_c_out_actv == 0 ? 
		0 : state_h_out[g_threadIdx] / state_c_out_actv;

	// state_c_out[g_threadIdx] =      state_c_out_reg;
	// state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	RealType state_h_out_grad_reg = state_h_out_grad[g_threadIdx];

	RealType state_c_out_actv_grad = state_h_out_grad_reg * output_gate;
	RealType      output_gate_grad = state_h_out_grad_reg * state_c_out_actv;

	RealType state_c_out_grad_reg = state_c_out_grad[g_threadIdx] + 
		state_c_out_actv_grad * 
		(1 - state_c_out_actv * state_c_out_actv);

	// state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;
	RealType  forget_gate_grad = state_c_out_grad_reg * state_c_reg;
	state_c_grad[g_threadIdx]  = state_c_out_grad_reg * forget_gate;
	RealType   input_actv_grad = state_c_out_grad_reg * input_gate;
	RealType   input_gate_grad = state_c_out_grad_reg * input_actv;

	// RealType  input_gate = __cu_sigmoid(input  [0 * BxH + g_threadIdx] + 
	//                                     state_h[0 * BxH + g_threadIdx]);
	// RealType forget_gate = __cu_sigmoid(input  [1 * BxH + g_threadIdx] + 
	//                                     state_h[1 * BxH + g_threadIdx]);
	// RealType  input_actv =         tanh(input  [2 * BxH + g_threadIdx] + 
	//                                     state_h[2 * BxH + g_threadIdx]);
	// RealType output_gate = __cu_sigmoid(input  [3 * BxH + g_threadIdx] + 
	//                                     state_h[3 * BxH + g_threadIdx]);

	RealType  input_gate_input_grad =  input_gate_grad *  input_gate * (1 -  input_gate);
	RealType forget_gate_input_grad = forget_gate_grad * forget_gate * (1 - forget_gate);
	RealType  input_actv_input_grad =  input_actv_grad * (1 - input_actv * input_actv);
	RealType output_gate_input_grad = output_gate_grad * output_gate * (1 - output_gate);

	workspace[batch_idx_x4H_plus_state_idx + 0 * state_size] =  input_gate_input_grad;
	workspace[batch_idx_x4H_plus_state_idx + 1 * state_size] = forget_gate_input_grad;
	workspace[batch_idx_x4H_plus_state_idx + 2 * state_size] =  input_actv_input_grad;
	workspace[batch_idx_x4H_plus_state_idx + 3 * state_size] = output_gate_input_grad;

	atomicAdd(&bias_grad[g_threadIdx % state_size + 0 * state_size],  input_gate_input_grad);
	atomicAdd(&bias_grad[g_threadIdx % state_size + 1 * state_size], forget_gate_input_grad);
	atomicAdd(&bias_grad[g_threadIdx % state_size + 2 * state_size],  input_actv_input_grad);
	atomicAdd(&bias_grad[g_threadIdx % state_size + 3 * state_size], output_gate_input_grad);
}

template <>
inline void FullyConnectedFW < float > (cublasHandle_t cublas_handle,
	const float * const __restrict__ X,
	const float * const __restrict__ W,
	      float * const __restrict__ Y,
	const OpReqType req,       const unsigned batch_size, 
	const unsigned input_size, const unsigned state_size,
	const bool batch_minor)
{
	float alpha = 1.0, beta = float(req == kAddTo);
	
	if (batch_minor)
	{
		CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle,
					CUBLAS_OP_T, // X.T
					CUBLAS_OP_N, // W
					batch_size,  // Y.shape[1]
					state_size,  // Y.shape[0]
					input_size,  // X.shape[1]
					&alpha, X, input_size, W, input_size,
					&beta , Y, batch_size));
	}
	else 
	{
		CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle
					CUBLAS_OP_T, // W.T
					CUBLAS_OP_N, // X
					state_size,  // Y.shape[1]
					batch_size,  // Y.shape[0]
					input_size,  // W.shape[1]
					&alpha, W, input_size, X, input_size,
					&beta,  Y, state_size));
	}
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

	}  // namespace op
}  // namespace mxnet
