#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "lstm_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

/**
 * Forward Pass of the LSTM Nonlinear Block
 * @param1  input     [B x 4H]
 * @param2  state_h   [B x 4H]
 * @param3  state_c   [B x  H]
 * @param4  input_fm  [B x  H]
 * @param5  forget_fm [B x  H]
 * @param6  iactv_fm  [B x  H]
 * @param7  output_fm [B x  H]
 * @param8  state_h_out [B x  H]
 * @param9  state_c_out [B x  H]
 * @param10 batch_size: (Parameter)
 * @param11 state_size: (Parameter)
 * @param12 is_train: (Runtime Parameter)
 */
template < typename RealType >
static __global__ void _cuda_lstm_nonlin_block__forward(
	const RealType * const __restrict__ input_plus_state_h,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ input_fm,
	      RealType * const __restrict__ forget_fm,
	      RealType * const __restrict__ iactv_fm,
	      RealType * const __restrict__ output_fm,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out, 
	const unsigned batch_size, const unsigned state_size, const bool is_train);

/**
 * Backward Pass of the LSTM Nonlinear Block
 * @param1  input_grad   [B x 4H]
 * @param2  state_h_grad [B x 4H]
 * @param3  state_c_grad [B x  H]
 * @param4  state_c      [B x  H]
 * @param5  input_fm     [B x  H]
 * @param6  forget_fm    [B x  H]
 * @param7  iactv_fm     [B x  H]
 * @param8  output_fm    [B x  H]
 * @param9  state_h_out_grad [B x  H]
 * @param10 state_c_out_grad [B x  H]
 * @param11 batch_size: (Parameter)
 * @param12 state_size: (Parameter)
 */
template < typename RealType >
static __global__ void _cuda_lstm_nonlin_block_backward(
	      RealType * const __restrict__ input_plus_state_h_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ input_fm,
	const RealType * const __restrict__ forget_fm,
	const RealType * const __restrict__ iactv_fm,
	const RealType * const __restrict__ output_fm,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size);

template < typename DType >
class CULSTMNonLinBlockOp : public Operator
{
private:
	LSTMNonLinBlockParam _param; 
	bool _initialized = false;
public:
	explicit CULSTMNonLinBlockOp(LSTMNonLinBlockParam param)
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

		Tensor < gpu, 2, DType > input_plus_state_h = 
			in_data[int(EnumOpInputs::InputPlusStateH)]
			.get < gpu, 2, DType > (cuda_stream);
		
		// infer the parameters from the cell input
		_param.batch_size = input_plus_state_h.shape_[0];
		_param.state_size = input_plus_state_h.shape_[1] / 4;
		
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

		std::size_t in_expected = 2, out_expected = 6;

		CHECK_EQ( in_data.size(),  in_expected); // input_plus_state_h, state_c
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out
		                                         // input_fm, forget_fm
							 // iactv_fm, output_fm

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 2, DType > input_plus_state_h = 
			in_data[int(EnumOpInputs::InputPlusStateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c =
			in_data[int(EnumOpInputs::StateC)]
			.get < gpu, 2, DType > (cuda_stream);

		Tensor < gpu, 2, DType > state_h_out = out_data[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out = out_data[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > input_fm    = out_data[int(EnumOpOutputs::InputFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > forget_fm   = out_data[int(EnumOpOutputs::ForgetFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > iactv_fm    = out_data[int(EnumOpOutputs::IActvFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > output_fm   = out_data[int(EnumOpOutputs::OutputFM)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(input_plus_state_h.CheckContiguous(), true);
		CHECK_EQ(state_c    .CheckContiguous(), true);
		CHECK_EQ(state_h_out.CheckContiguous(), true);
		CHECK_EQ(state_c_out.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}

		const unsigned BxH = _param.batch_size * _param.state_size;

		_cuda_lstm_nonlin_block__forward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				input_plus_state_h.dptr_, 
				  state_c.dptr_,
				 input_fm.dptr_,
				forget_fm.dptr_,
				 iactv_fm.dptr_,
				output_fm.dptr_,
				state_h_out.dptr_,
				state_c_out.dptr_,
			       _param.batch_size, 
			       _param.state_size, ctx.is_train
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

		std::size_t in_expected = 2, out_expected = 6, visible_out_expected = 2;

		CHECK_EQ( in_data.size(),  in_expected); // input_plus_state_h, state_c
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out
		                                         // input_fm, forget_fm
							 // iactv_fm, output_fm
		CHECK_EQ(out_grad.size(), visible_out_expected); // state_h_out, state_c_out

		CHECK_NE(req[int(EnumOpInputs::InputPlusStateH)], kAddTo) 
			<< "AddTo is not supported for input_plus_state_h.";
		CHECK_NE(req[int(EnumOpInputs::StateC)], kAddTo) 
			<< "AddTo is not supported for state_c.";	

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
	
		Tensor < gpu, 2, DType > input_plus_state_h_grad = 
			in_grad[int(EnumOpInputs::InputPlusStateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_grad = 
			in_grad[int(EnumOpInputs::StateC)]
			.get < gpu, 2, DType > (cuda_stream);
		
		Tensor < gpu, 2, DType > state_c = in_data[int(EnumOpInputs::StateC)]
			.get < gpu, 2, DType > (cuda_stream);

		Tensor < gpu, 2, DType > input_fm  = out_data[int(EnumOpOutputs::InputFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > forget_fm = out_data[int(EnumOpOutputs::ForgetFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > iactv_fm  = out_data[int(EnumOpOutputs::IActvFM)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > output_fm = out_data[int(EnumOpOutputs::OutputFM)]
			.get < gpu, 2, DType > (cuda_stream);	

		Tensor < gpu, 2, DType > state_h_out_grad = out_grad[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out_grad = out_grad[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(input_plus_state_h_grad.CheckContiguous(), true);
		CHECK_EQ(state_c_grad    .CheckContiguous(), true);
		CHECK_EQ(state_c         .CheckContiguous(), true);
		CHECK_EQ(state_h_out_grad.CheckContiguous(), true);
		CHECK_EQ(state_c_out_grad.CheckContiguous(), true);

		const unsigned BxH = _param.batch_size * _param.state_size;

		_cuda_lstm_nonlin_block_backward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				input_plus_state_h_grad.dptr_,
				state_c_grad.dptr_,
				state_c.dptr_,
				 input_fm.dptr_,
				forget_fm.dptr_,
				 iactv_fm.dptr_,
				output_fm.dptr_,
				state_h_out_grad.dptr_,
				state_c_out_grad.dptr_,
			       _param.batch_size, _param.state_size
			);
	}
}; // class CULSTMNonLinBlockOp

template < typename RealType >
static __forceinline__ __device__ RealType __cu_sigmoid(RealType i)
{
	return 1.0 / (1.0 + exp(-i));
}

template < typename RealType >
__global__ void _cuda_lstm_nonlin_block__forward(
	const RealType * const __restrict__ input_plus_state_h,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ input_fm,
	      RealType * const __restrict__ forget_fm,
	      RealType * const __restrict__ iactv_fm,
	      RealType * const __restrict__ output_fm,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out, 
	const unsigned batch_size, const unsigned state_size, const bool is_train)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;

	RealType input_gate  = __cu_sigmoid(input_plus_state_h[batch_idx_x4H_plus_state_idx + 0 * state_size]);
	RealType forget_gate = __cu_sigmoid(input_plus_state_h[batch_idx_x4H_plus_state_idx + 1 * state_size]);
	RealType input_actv  =         tanh(input_plus_state_h[batch_idx_x4H_plus_state_idx + 2 * state_size]);
	RealType output_gate = __cu_sigmoid(input_plus_state_h[batch_idx_x4H_plus_state_idx + 3 * state_size]);

	RealType state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;

	state_c_out[g_threadIdx] =      state_c_out_reg;
	state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	if (is_train)
	{
		// preserve the gates to be used in the backward pass
		input_fm[g_threadIdx] = input_gate; forget_fm[g_threadIdx] = forget_gate;
		iactv_fm[g_threadIdx] = input_actv; output_fm[g_threadIdx] = output_gate;
	}
}

template < typename RealType >
__global__ void _cuda_lstm_nonlin_block_backward(
	      RealType * const __restrict__ input_plus_state_h_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ input_fm,
	const RealType * const __restrict__ forget_fm,
	const RealType * const __restrict__ iactv_fm,
	const RealType * const __restrict__ output_fm,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;

	// read the activations stored in the forward pass
	RealType input_gate = input_fm[g_threadIdx], forget_gate = forget_fm[g_threadIdx],
	         input_actv = iactv_fm[g_threadIdx], output_gate = output_fm[g_threadIdx];

	RealType state_c_reg = state_c[g_threadIdx];

	RealType state_c_out_actv = tanh(forget_gate * state_c_reg + input_actv * input_gate);

	// state_c_out[g_threadIdx] =      state_c_out_reg;
	// state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	RealType state_h_out_grad_reg = state_h_out_grad[g_threadIdx];

	RealType  output_gate_grad = state_h_out_grad_reg * state_c_out_actv;
	RealType state_c_actv_grad = state_h_out_grad_reg * output_gate;
	
	RealType state_c_out_grad_reg = state_c_out_grad[g_threadIdx] + state_c_actv_grad * (1 - state_c_out_actv * state_c_out_actv);

	// state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;
	RealType  forget_gate_grad = state_c_out_grad_reg * state_c_reg;
	RealType   input_actv_grad = state_c_out_grad_reg * input_gate;
	RealType   input_gate_grad = state_c_out_grad_reg * input_actv;

	state_c_grad[g_threadIdx]  = state_c_out_grad_reg * forget_gate;

	// RealType  input_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 0 * state_size] + 
	//                                     state_h[batch_idx_x4H_plus_state_idx + 0 * state_size]);
	// RealType forget_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 1 * state_size] + 
	//                                     state_h[batch_idx_x4H_plus_state_idx + 1 * state_size]);
	// RealType  input_actv =         tanh(input  [batch_idx_x4H_plus_state_idx + 2 * state_size] + 
	//                                     state_h[batch_idx_x4H_plus_state_idx + 2 * state_size]);
	// RealType output_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 3 * state_size] + 
	//                                     state_h[batch_idx_x4H_plus_state_idx + 3 * state_size]);

	input_plus_state_h_grad[batch_idx_x4H_plus_state_idx + 0 * state_size] = 
		 input_gate_grad *  input_gate * (1 -  input_gate);
	input_plus_state_h_grad[batch_idx_x4H_plus_state_idx + 1 * state_size] = 
		forget_gate_grad * forget_gate * (1 - forget_gate);
	input_plus_state_h_grad[batch_idx_x4H_plus_state_idx + 2 * state_size] = 
		 input_actv_grad * (1 - input_actv * input_actv);
	input_plus_state_h_grad[batch_idx_x4H_plus_state_idx + 3 * state_size] = 
		output_gate_grad * output_gate * (1 - output_gate);
}

	} // namespace op
} // namespace mxnet
