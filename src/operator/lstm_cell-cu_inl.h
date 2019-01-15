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
 * @param1 i2h_bias           [4H]:  (Input) I2H Bias
 * @param2 h2h_bias           [4H]:  (Input) H2H Bias
 * @param3 state_c        [B x  H]:  (Input) Cell State from the Previous Time Step
 * @param4 reserved_space [B x 4H]:  (Inout) Reserved Space that has the Sum of I2H and H2H
 * @param5 state_h_out    [B x  H]: (Output) Hidden State Output that goes to the Next Time Step and/or Layer
 * @param6 state_c_out    [B x  H]: (Output) Cell   State Output that goes to the Next Time Step
 * @param7 batch_size: (Parameter) Batch Size
 * @param8 state_size: (Parameter) State Size
 */
template < typename RealType >
__global__ void _cuda_lstm_cell__forward(
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
 * All the parameters are the same as the forward kernel, except that the input now becomes output and vice versa.
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

template < typename DType >
class CUEcoLSTMCellOp : public Operator
{
private:
	EcoLSTMCellParam _param;

	bool _initialized = false;

	// ReservedSpace
	Storage::Handle _reserved_space;
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

		const unsigned BxH = _param.batch_size * _param.state_size;

		// @TODO Kernels go here.
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
		
		// Tensor < gpu, 2, DType >

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

		const unsigned BxH = _param.batch_size * _param.state_size;

		// @TODO Kernels go here.
	}
};

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet