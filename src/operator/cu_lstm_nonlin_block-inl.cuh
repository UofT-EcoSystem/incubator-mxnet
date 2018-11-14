#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "lstm_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

template < typename RealType >
static __forceinline__ __device__ RealType __cu_sigmoid(RealType i)
{
	return 1.0 / (1.0 + exp(-i));
}

/**
 * Forward Pass of the LSTM Nonlinear Block
 * This kernel shall be launched using the parameter <<< ceil(BxH / 128), 128 >>>.
 * @param1   i_cell_input [B x 4H]:  (Input) Input to the LSTM Cell from the Previous Layer
 * @param2 i_hidden_state [B x 4H]:  (Input) Hidden State from the Previous Time Step
 * @param3   i_cell_state [B x  H]:  (Input)   Cell State from the Previous Time Step
 * @param4 reserved_space [B x 4H]: (Output) Space reserved by the Forward Pass to facilitate Backward Pass Compute
 * @param5 o_hidden_state [B x  H]: (Output) Hidden State Output that goes to the Next Time Step and/or Layer
 * @param6   o_cell_state [B x  H]: (Output)   Cell State Output that goes to the Next Time Step and/or Layer
 * @param7 batch_size: (Parameter) Batch Size
 * @param8 state_size: (Parameter) State Size
 */
template < typename RealType >
static __global__ void _cuda_fused_lstm_nonlin_block__forward(
	const RealType * const __restrict__   i_cell_input,
	const RealType * const __restrict__ i_hidden_state,
	const RealType * const __restrict__   i_cell_state,
	      RealType * const __restrict__ reserved_space,
	      RealType * const __restrict__ o_hidden_state,
	      RealType * const __restrict__   o_cell_state, 
	const unsigned batch_size, const unsigned state_size)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;
	const unsigned batch_idx_x5H_plus_state_idx = (g_threadIdx / state_size) * 5 * state_size + 
	                                               g_threadIdx % state_size;

	RealType  input_gate = __cu_sigmoid(i_cell_input[batch_idx_x4H_plus_state_idx + 0 * state_size] + 
	                                  i_hidden_state[batch_idx_x4H_plus_state_idx + 0 * state_size]);
	RealType forget_gate = __cu_sigmoid(i_cell_input[batch_idx_x4H_plus_state_idx + 1 * state_size] + 
	                                  i_hidden_state[batch_idx_x4H_plus_state_idx + 1 * state_size]);
	RealType  input_actv =         tanh(i_cell_input[batch_idx_x4H_plus_state_idx + 2 * state_size] + 
	                                  i_hidden_state[batch_idx_x4H_plus_state_idx + 2 * state_size]);
	RealType output_gate = __cu_sigmoid(i_cell_input[batch_idx_x4H_plus_state_idx + 3 * state_size] + 
	                                  i_hidden_state[batch_idx_x4H_plus_state_idx + 3 * state_size]);

	RealType i_cell_state_reg = i_cell_state[g_threadIdx];
	RealType o_cell_state_reg = forget_gate * i_cell_state_reg + input_actv * input_gate;

	  o_cell_state[g_threadIdx] =      o_cell_state_reg;
	o_hidden_state[g_threadIdx] = tanh(o_cell_state_reg) * output_gate;

	if (reserved_space != nullptr)
	{
		// reserve the gates to be used in the backward pass
		reserved_space[batch_idx_x5H_plus_state_idx + 0 * state_size] =  input_gate;
		reserved_space[batch_idx_x5H_plus_state_idx + 1 * state_size] = forget_gate;
		reserved_space[batch_idx_x5H_plus_state_idx + 2 * state_size] =  input_actv;
		reserved_space[batch_idx_x5H_plus_state_idx + 3 * state_size] = output_gate;
		reserved_space[batch_idx_x5H_plus_state_idx + 4 * state_size] = i_cell_state_reg;
	}
}

/**
 * Backward Pass of the LSTM Nonlinear Block
 * This kernel shall be launched using the parameter <<< ceil(BxH / 128), 128 >>>.
 * All the parameters are the same as the forward kernel, except that the input now becomes output and vice versa.
 */
template < typename RealType >
static __global__ void _cuda_fused_lstm_nonlin_block_backward(
	      RealType * const __restrict__   i_cell_input_grad,
	      RealType * const __restrict__ i_hidden_state_grad,
	// const RealType * const __restrict__   i_cell_state,
	      RealType * const __restrict__   i_cell_state_grad,
	const RealType * const __restrict__ reserved_space,
	const RealType * const __restrict__ o_hidden_state_grad,
	const RealType * const __restrict__   o_cell_state_grad,
	const unsigned batch_size, const unsigned state_size)
{
	const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x,
		BxH = batch_size * state_size;

	if (g_threadIdx >= BxH) { return ; }

	const unsigned batch_idx_x4H_plus_state_idx = (g_threadIdx / state_size) * 4 * state_size + 
	                                               g_threadIdx % state_size;
	const unsigned batch_idx_x5H_plus_state_idx = (g_threadIdx / state_size) * 5 * state_size + 
	                                               g_threadIdx % state_size;

	// read the activations stored in the forward pass
	RealType   input_gate = reserved_space[batch_idx_x5H_plus_state_idx + 0 * state_size];
	RealType  forget_gate = reserved_space[batch_idx_x5H_plus_state_idx + 1 * state_size];
	RealType   input_actv = reserved_space[batch_idx_x5H_plus_state_idx + 2 * state_size];
	RealType  output_gate = reserved_space[batch_idx_x5H_plus_state_idx + 3 * state_size];
	RealType i_cell_state = reserved_space[batch_idx_x5H_plus_state_idx + 4 * state_size];
	RealType   cell_state_actv = tanh(i_cell_state);

	//   o_cell_state[g_threadIdx] =      cell_state;
	// o_hidden_state[g_threadIdx] = tanh(cell_state) * output_gate;

	RealType o_hidden_state_grad_reg = o_hidden_state_grad[g_threadIdx];

	RealType     output_gate_grad = o_hidden_state_grad_reg * cell_state_actv;
	RealType cell_state_actv_grad = o_hidden_state_grad_reg * output_gate;
	
	RealType cell_state_grad = o_cell_state_grad[g_threadIdx] + cell_state_actv_grad * (1 - cell_state_actv * cell_state_actv);

	// cell_state = forget_gate * i_cell_state[g_threadIdx] + input_actv * input_gate;
	RealType  forget_gate_grad = cell_state_grad * i_cell_state;
	RealType   input_actv_grad = cell_state_grad * input_gate;
	RealType   input_gate_grad = cell_state_grad * input_actv;

	i_cell_state_grad[g_threadIdx] = cell_state_grad * forget_gate;

	// RealType  input_gate = __cu_sigmoid(i_cell_input[0 * BxH + g_threadIdx] + 
	//                                   i_hidden_state[0 * BxH + g_threadIdx]);
	// RealType forget_gate = __cu_sigmoid(i_cell_input[1 * BxH + g_threadIdx] + 
	//                                   i_hidden_state[1 * BxH + g_threadIdx]);
	// RealType  input_actv =         tanh(i_cell_input[2 * BxH + g_threadIdx] + 
	//                                   i_hidden_state[2 * BxH + g_threadIdx]);
	// RealType output_gate = __cu_sigmoid(i_cell_input[3 * BxH + g_threadIdx] + 
	//                                   i_hidden_state[3 * BxH + g_threadIdx]);

	  i_cell_input_grad[batch_idx_x4H_plus_state_idx + 0 * state_size] = 
	i_hidden_state_grad[batch_idx_x4H_plus_state_idx + 0 * state_size] =  input_gate_grad *  input_gate * (1 -  input_gate);
	  i_cell_input_grad[batch_idx_x4H_plus_state_idx + 1 * state_size] = 
	i_hidden_state_grad[batch_idx_x4H_plus_state_idx + 1 * state_size] = forget_gate_grad * forget_gate * (1 - forget_gate);
	  i_cell_input_grad[batch_idx_x4H_plus_state_idx + 2 * state_size] = 
	i_hidden_state_grad[batch_idx_x4H_plus_state_idx + 2 * state_size] =  input_actv_grad * (1 - input_actv * input_actv);
	  i_cell_input_grad[batch_idx_x4H_plus_state_idx + 3 * state_size] = 
	i_hidden_state_grad[batch_idx_x4H_plus_state_idx + 3 * state_size] = output_gate_grad * output_gate * (1 - output_gate);
}

template < typename DType >
class CULSTMNonLinBlockOp : public Operator
{
private:
	LSTMNonLinBlockParam _param; 
	
	bool _initialized = false;

	// ReserveSpace
	Storage::Handle _reserved_space;
public:
	explicit CULSTMNonLinBlockOp(LSTMNonLinBlockParam param)
	{
		_param = param;
	}
	~CULSTMNonLinBlockOp()
	{
		// Storage::Get()->Free(_reserved_space);
	}

private:
	void _Init(mshadow::Stream < gpu > * cuda_stream,
	           const std::vector < TBlob > &  in_data,
		   const std::vector < TBlob > & out_data)
	{
		using namespace mshadow;

		CHECK_EQ(_initialized, false);

		Tensor < gpu, 2, DType > i_cell_input = in_data[int(EnumOpInputs::CellInput)].
			get < gpu, 2, DType > (cuda_stream);
		
		// infer the parameters from the cell input
		_param.batch_size = i_cell_input.shape_[0];
		_param.state_size = i_cell_input.shape_[1] / 4;
		
		// allocate the reserve space
		// _reserved_space = Storage::Get()->Alloc(_param.batch_size * 4 * _param.state_size * sizeof(DType), 
		//                                         Context::GPU());
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

		CHECK_EQ( in_data.size(),  in_expected); // CellInput, HiddenState, CellState
		CHECK_EQ(out_data.size(), out_expected); //            HiddenState, CellState

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 2, DType >   i_cell_input =  in_data[int(EnumOpInputs ::  CellInput)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i_hidden_state =  in_data[int(EnumOpInputs ::HiddenState)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   i_cell_state =  in_data[int(EnumOpInputs ::  CellState)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > o_hidden_state = out_data[int(EnumOpOutputs::HiddenState)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   o_cell_state = out_data[int(EnumOpOutputs::  CellState)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(  i_cell_input.CheckContiguous(), true);
		CHECK_EQ(i_hidden_state.CheckContiguous(), true);
		CHECK_EQ(  i_cell_state.CheckContiguous(), true);
		CHECK_EQ(o_hidden_state.CheckContiguous(), true);
		CHECK_EQ(  o_cell_state.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}

		if (ctx.is_train)
		{
			_reserved_space = Storage::Get()->Alloc(_param.batch_size * 5 * _param.state_size * sizeof(DType), 
		                                                Context::GPU());
		}

		const unsigned BxH = _param.batch_size * _param.state_size;

		_cuda_fused_lstm_nonlin_block__forward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				  i_cell_input.dptr_, 
				i_hidden_state.dptr_,
				  i_cell_state.dptr_,
				ctx.is_train ? 
					reinterpret_cast < DType * > (_reserved_space.dptr) : nullptr,
				o_hidden_state.dptr_,
				  o_cell_state.dptr_,
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

		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected);
		CHECK_EQ(out_grad.size(), out_expected);

		CHECK_NE(req[int(EnumOpInputs::  CellInput)], kAddTo) << "AddTo is not supported for "   
			  "cell input.";
		CHECK_NE(req[int(EnumOpInputs::HiddenState)], kAddTo) << "AddTo is not supported for " 
			"hidden state.";
		CHECK_NE(req[int(EnumOpInputs::  CellState)], kAddTo) << "AddTo is not supported for " 
			  "cell state.";	

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
	
		Tensor < gpu, 2, DType >   i_cell_input_grad =  in_grad[int(EnumOpInputs ::  CellInput)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i_hidden_state_grad =  in_grad[int(EnumOpInputs ::HiddenState)]
			.get < gpu, 2, DType > (cuda_stream);
		// Tensor < gpu, 2, DType >   i_cell_state      =  in_data[int(EnumOpInputs ::  CellState)]
		// 	.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   i_cell_state_grad =  in_grad[int(EnumOpInputs ::  CellState)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > o_hidden_state_grad = out_grad[int(EnumOpOutputs::HiddenState)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   o_cell_state_grad = out_grad[int(EnumOpOutputs::  CellState)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(  i_cell_input_grad.CheckContiguous(), true);
		CHECK_EQ(i_hidden_state_grad.CheckContiguous(), true);
		CHECK_EQ(  i_cell_state_grad.CheckContiguous(), true);
		CHECK_EQ(o_hidden_state_grad.CheckContiguous(), true);
		CHECK_EQ(  o_cell_state_grad.CheckContiguous(), true);

		const unsigned BxH = _param.batch_size * _param.state_size;

		_cuda_fused_lstm_nonlin_block_backward < DType >
			<<<
				(BxH - 1) / 128 + 1, 128, 0, Stream < gpu > ::GetStream(cuda_stream)
			>>> 
			(
				  i_cell_input_grad.dptr_,
				i_hidden_state_grad.dptr_,
				//   i_cell_state     .dptr_,
				  i_cell_state_grad.dptr_,
				reinterpret_cast < DType * > (_reserved_space.dptr),
				o_hidden_state_grad.dptr_,
				  o_cell_state_grad.dptr_,
				_param.batch_size, _param.state_size
			);
		
		Storage::Get()->Free(_reserved_space);
	}
}; // class CULSTMNonLinBlockOp

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet