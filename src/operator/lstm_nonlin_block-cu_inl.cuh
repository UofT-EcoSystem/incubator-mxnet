#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "lstm_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

/**
 * Forward Pass of the LSTM Nonlinear Block
 * @param1 input   [B x 4H]
 * @param2 state_h [B x 4H]
 * @param3 state_c [B x  H]
 * @param4 reserved_space [B x 4H]
 * @param5 state_h_out [B x  H]
 * @param6 state_c_out [B x  H]
 * @param7 batch_size: (Parameter)
 * @param8 state_size: (Parameter)
 */
template < typename RealType >
static __global__ void _cuda_lstm_nonlin_block__forward(
	const RealType * const __restrict__ input,
	const RealType * const __restrict__ state_h,
	const RealType * const __restrict__ state_c,
	      RealType * const __restrict__ reserved_space,
	      RealType * const __restrict__ state_h_out,
	      RealType * const __restrict__ state_c_out, 
	const unsigned batch_size, const unsigned state_size);

/**
 * Backward Pass of the LSTM Nonlinear Block
 * @param1 input_grad   [B x 4H]
 * @param2 state_h_grad [B x 4H]
 * @param3 state_c_grad [B x  H]
 * @param4 state_c      [B x  H]
 * @param5 reserved_space [B x 4H]
 * @param6 state_h_out_grad [B x  H]
 * @param7 state_c_out_grad [B x  H]
 * @param8 batch_size: (Parameter)
 * @param9 state_size: (Parameter)
 */
template < typename RealType >
static __global__ void _cuda_lstm_nonlin_block_backward(
	      RealType * const __restrict__ input_grad,
	      RealType * const __restrict__ state_h_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ reserved_space,
	const RealType * const __restrict__ state_h_out_grad,
	const RealType * const __restrict__ state_c_out_grad,
	const unsigned batch_size, const unsigned state_size);

template < typename DType >
class CULSTMNonLinBlockOp : public Operator
{
private:
	LSTMNonLinBlockParam _param; 
	
	bool _initialized = false;

	// ReserveSpace
	Storage::Handle _reserved_space, _state_c_out_actv;
public:
	explicit CULSTMNonLinBlockOp(LSTMNonLinBlockParam param)
	{
		_param = param;
	}
	~CULSTMNonLinBlockOp()
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

		Tensor < gpu, 2, DType > input = in_data[int(EnumOpInputs::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		
		// infer the parameters from the cell input
		_param.batch_size = input.shape_[0];
		_param.state_size = input.shape_[1] / 4;
		
		// allocate the reserved space [B x 4H]
		_reserved_space = Storage::Get()->Alloc(_param.batch_size * 4 * _param.state_size * sizeof(DType), 
		                                        Context::GPU());
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

		std::size_t in_expected = 3, out_expected = 2;

		CHECK_EQ( in_data.size(),  in_expected); // input, state_h, state_c
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

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

		CHECK_EQ(input      .CheckContiguous(), true);
		CHECK_EQ(state_h    .CheckContiguous(), true);
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
				input  .dptr_, 
				state_h.dptr_,
				state_c.dptr_,
				ctx.is_train ? 
					reinterpret_cast < DType * > (_reserved_space.dptr) : nullptr,
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

		std::size_t in_expected = 3, out_expected = 2;

		CHECK_EQ( in_data.size(),  in_expected); // input, state_h, state_c
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected); // state_h_out, state_c_out
		CHECK_EQ(out_grad.size(), out_expected);

		CHECK_NE(req[int(EnumOpInputs::Input )], kAddTo) << "AddTo is not supported for input.";
		CHECK_NE(req[int(EnumOpInputs::StateH)], kAddTo) << "AddTo is not supported for state_h.";
		CHECK_NE(req[int(EnumOpInputs::StateC)], kAddTo) << "AddTo is not supported for state_c.";	

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
	
		Tensor < gpu, 2, DType > input_grad       =  in_grad[int(EnumOpInputs ::Input)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_h_grad     =  in_grad[int(EnumOpInputs ::StateH)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_grad     =  in_grad[int(EnumOpInputs ::StateC)]
			.get < gpu, 2, DType > (cuda_stream);
		
		Tensor < gpu, 2, DType > state_c          =  in_data[int(EnumOpInputs ::StateC)]
			.get < gpu, 2, DType > (cuda_stream);

		Tensor < gpu, 2, DType > state_h_out_grad = out_grad[int(EnumOpOutputs::StateHOut)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > state_c_out_grad = out_grad[int(EnumOpOutputs::StateCOut)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(input_grad      .CheckContiguous(), true);
		CHECK_EQ(state_h_grad    .CheckContiguous(), true);
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
				input_grad  .dptr_,
				state_h_grad.dptr_,
				state_c_grad.dptr_,
				state_c     .dptr_,
				reinterpret_cast < DType * > (_reserved_space.dptr),
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
	const RealType * const __restrict__ input,
	const RealType * const __restrict__ state_h,
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

	RealType  input_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 0 * state_size] + 
	                                    state_h[batch_idx_x4H_plus_state_idx + 0 * state_size]);
	RealType forget_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 1 * state_size] + 
	                                    state_h[batch_idx_x4H_plus_state_idx + 1 * state_size]);
	RealType  input_actv =         tanh(input  [batch_idx_x4H_plus_state_idx + 2 * state_size] + 
	                                    state_h[batch_idx_x4H_plus_state_idx + 2 * state_size]);
	RealType output_gate = __cu_sigmoid(input  [batch_idx_x4H_plus_state_idx + 3 * state_size] + 
	                                    state_h[batch_idx_x4H_plus_state_idx + 3 * state_size]);

	RealType state_c_out_reg = forget_gate * state_c[g_threadIdx] + input_actv * input_gate;

	state_c_out[g_threadIdx] =      state_c_out_reg;
	state_h_out[g_threadIdx] = tanh(state_c_out_reg) * output_gate;

	if (reserved_space != nullptr)
	{
		// preserve the gates to be used in the backward pass
		reserved_space[batch_idx_x4H_plus_state_idx + 0 * state_size] =  input_gate;
		reserved_space[batch_idx_x4H_plus_state_idx + 1 * state_size] = forget_gate;
		reserved_space[batch_idx_x4H_plus_state_idx + 2 * state_size] =  input_actv;
		reserved_space[batch_idx_x4H_plus_state_idx + 3 * state_size] = output_gate;
	}
}

template < typename RealType >
__global__ void _cuda_lstm_nonlin_block_backward(
	      RealType * const __restrict__ input_grad,
	      RealType * const __restrict__ state_h_grad,
	      RealType * const __restrict__ state_c_grad,
	const RealType * const __restrict__ state_c,
	const RealType * const __restrict__ reserved_space,
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
	RealType   input_gate = reserved_space[batch_idx_x4H_plus_state_idx + 0 * state_size];
	RealType  forget_gate = reserved_space[batch_idx_x4H_plus_state_idx + 1 * state_size];
	RealType   input_actv = reserved_space[batch_idx_x4H_plus_state_idx + 2 * state_size];
	RealType  output_gate = reserved_space[batch_idx_x4H_plus_state_idx + 3 * state_size];

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

	// RealType  input_gate = __cu_sigmoid(input  [0 * BxH + g_threadIdx] + 
	//                                     state_h[0 * BxH + g_threadIdx]);
	// RealType forget_gate = __cu_sigmoid(input  [1 * BxH + g_threadIdx] + 
	//                                     state_h[1 * BxH + g_threadIdx]);
	// RealType  input_actv =         tanh(input  [2 * BxH + g_threadIdx] + 
	//                                     state_h[2 * BxH + g_threadIdx]);
	// RealType output_gate = __cu_sigmoid(input  [3 * BxH + g_threadIdx] + 
	//                                     state_h[3 * BxH + g_threadIdx]);

	input_grad  [batch_idx_x4H_plus_state_idx + 0 * state_size] = 
	state_h_grad[batch_idx_x4H_plus_state_idx + 0 * state_size] =  input_gate_grad *  input_gate * (1 -  input_gate);
	input_grad  [batch_idx_x4H_plus_state_idx + 1 * state_size] = 
	state_h_grad[batch_idx_x4H_plus_state_idx + 1 * state_size] = forget_gate_grad * forget_gate * (1 - forget_gate);
	input_grad  [batch_idx_x4H_plus_state_idx + 2 * state_size] = 
	state_h_grad[batch_idx_x4H_plus_state_idx + 2 * state_size] =  input_actv_grad * (1 - input_actv * input_actv);
	input_grad  [batch_idx_x4H_plus_state_idx + 3 * state_size] = 
	state_h_grad[batch_idx_x4H_plus_state_idx + 3 * state_size] = output_gate_grad * output_gate * (1 - output_gate);
}

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet