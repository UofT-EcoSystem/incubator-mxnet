#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "lstm_nonlin_block-inl.h"
// #include "lstm_nonlin_block-impl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

template < typename DType >
class CULSTMNonLinBlockOp : public Operator
{
private:
	LSTMNonLinBlockParam _param; bool _initialized = false;
public:
	explicit CULSTMNonLinBlockOp(LSTMNonLinBlockParam param)
	{
		_param = param;
	}
	
	~CULSTMNonLinBlockOp()
	{
		// TODO: Free the storage allocated for ReserveSpace.
	}

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

		Stream < gpu > * cuda_stream = ctx.get_stream();

		Tensor < gpu, 2, DType >   i_cell_input =  in_data[int(EnumOpInputs ::  CellInput)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i_hidden_state =  in_data[int(EnumOpInputs ::HiddenState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   i_cell_state =  in_data[int(EnumOpInputs ::  CellState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > o_hidden_state = out_data[int(EnumOpOutputs::HiddenState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   o_cell_state = out_data[int(EnumOpOutputs::  CellState)].
			get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(  i_cell_input.CheckContiguous(), true);
		CHECK_EQ(i_hidden_state.CheckContiguous(), true);
		CHECK_EQ(  i_cell_state.CheckContiguous(), true);
		CHECK_EQ(o_hidden_state.CheckContiguous(), true);
		CHECK_EQ(  o_cell_state.CheckContiguous(), true);

		if (!_initialized)
		{
			// TODO: Allocate the memory for the reserved space.
		}
		// TODO: Forward kernel goes here.
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

		// CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		// CHECK_EQ(out_data.size(), out_expected);
		CHECK_EQ(out_grad.size(), out_expected);

		CHECK_NE(req[int(EnumOpInputs::  CellInput)], kAddTo) << "AddTo is not supported for "   "cell input.";
		CHECK_NE(req[int(EnumOpInputs::HiddenState)], kAddTo) << "AddTo is not supported for " "hidden state.";
		CHECK_NE(req[int(EnumOpInputs::  CellState)], kAddTo) << "AddTo is not supported for "   "cell state.";	

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
	
		Tensor < gpu, 2, DType >   i_cell_input_grad =  in_grad[int(EnumOpInputs ::  CellInput)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > i_hidden_state_grad =  in_grad[int(EnumOpInputs ::HiddenState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   i_cell_state_grad =  in_grad[int(EnumOpInputs ::  CellState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > o_hidden_state_grad = out_grad[int(EnumOpOutputs::HiddenState)].
			get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType >   o_cell_state_grad = out_grad[int(EnumOpOutputs::  CellState)].
			get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(  i_cell_input_grad.CheckContiguous(), true);
		CHECK_EQ(i_hidden_state_grad.CheckContiguous(), true);
		CHECK_EQ(  i_cell_state_grad.CheckContiguous(), true);
		CHECK_EQ(o_hidden_state_grad.CheckContiguous(), true);
		CHECK_EQ(  o_cell_state_grad.CheckContiguous(), true);

		// TODO: Backward kernel goes here.
	}

}; // class CULSTMNonLinBlockOp

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet