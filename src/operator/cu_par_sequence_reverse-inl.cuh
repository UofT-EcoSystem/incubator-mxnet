#pragma once

#include "par_sequence_reverse-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

/**
 * Forward Pass of the Parallel Reverse Sequence
 * This kernel shall be launched using the parameter <<< (T, B), H, cuda_stream >>>
 * @param1 idata [T x B x H]:  (Input)  Input Data
 * @param2 odata [T x B x H]: (Output) Output Data
 * @param4 sequence_length [B]: Sequence Length of Each Batch
 */
template < typename RealType >
static __global__ void _cuda_par_reverse_sequence(
	const RealType * const __restrict__ idata,
	      RealType * const __restrict__ odata,
	const RealType * const __restrict__ sequence_length,
	const OpReqType req = OpReqType::kWriteTo);

template < typename DType >
class CUParSequenceReverseOp : public Operator 
{
private:
	ParSequenceReverseParam _param;

	bool _initialized = false;
public:
	explicit CUParSequenceReverseOp(ParSequenceReverseParam param)
	{
		_param = param;
	}
	~CUParSequenceReverseOp() {}
private:
	void _Init(mshadow::Stream < gpu > * cuda_stream,
	           const std::vector < TBlob > &  in_data,
		   const std::vector < TBlob > & out_data)
	{
		using namespace mshadow;

		CHECK_EQ(_initialized, false);

		Tensor < gpu, 3, DType > idata = in_data[int(EnumOpInputs::IData)].
			get < gpu, 3, DType > (cuda_stream);
		
		// infer the parameters from the input data
		_param.seq_length = idata.shape_[0];
		_param.batch_size = idata.shape_[1];
		_param.state_size = idata.shape_[2];

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

		std::size_t in_expected = _param.use_sequence_length ? 2 : 1, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected); // IData, SequenceLength
		CHECK_EQ(out_data.size(), out_expected); // OData

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 3, DType > idata =  in_data[int(EnumOpInputs ::IData)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 3, DType > odata = out_data[int(EnumOpOutputs::OData)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(idata.CheckContiguous(), true);
		CHECK_EQ(odata.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}

		DType * ptr_sequence_length = nullptr;

		if (_param.use_sequence_length)
		{
			Tensor < gpu, 1, DType > sequence_length = in_data[int(EnumOpInputs::SequenceLength)]
				.get < gpu, 1, DType > (cuda_stream);
			CHECK_EQ(sequence_length.CheckContiguous(), true);

			ptr_sequence_length = sequence_length.dptr_;
		}

		_cuda_par_reverse_sequence < DType >
			<<<
				dim3(_param.batch_size, 
				     _param.seq_length),
				_param.state_size, 0,
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				idata.dptr_,
				odata.dptr_,
				ptr_sequence_length
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

		std::size_t in_expected = _param.use_sequence_length ? 2 : 1, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected);
		CHECK_EQ(out_grad.size(), out_expected);

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
		
		Tensor < gpu, 3, DType > idata_grad =  in_grad[int(EnumOpInputs ::IData)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 3, DType > odata_grad = out_grad[int(EnumOpOutputs::OData)]
			.get < gpu, 3, DType > (cuda_stream);
		
		CHECK_EQ(idata_grad.CheckContiguous(), true);
		CHECK_EQ(odata_grad.CheckContiguous(), true);

		DType * ptr_sequence_length = nullptr;

		if (_param.use_sequence_length)
		{
			Tensor < gpu, 1, DType > sequence_length = in_data[int(EnumOpInputs::SequenceLength)]
				.get < gpu, 1, DType > (cuda_stream);
			CHECK_EQ(sequence_length.CheckContiguous(), true);

			ptr_sequence_length = sequence_length.dptr_;
		}

		_cuda_par_reverse_sequence < DType >
			<<<
				dim3(_param.batch_size, 
				     _param.seq_length),
				_param.state_size, 0,
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				odata_grad.dptr_,
				idata_grad.dptr_,
				ptr_sequence_length,
				req[int(EnumOpInputs::IData)]
			);
	}
}; // class CUParSequenceReverseOp

template < typename RealType >
__global__ void _cuda_par_reverse_sequence(
	const RealType * const __restrict__ idata,
	      RealType * const __restrict__ odata,
	const RealType * const __restrict__ sequence_length, const OpReqType req)
{
	const unsigned seq_idx = blockIdx.y, batch_idx = blockIdx.x;

	unsigned remapped_seq_idx = gridDim.y - seq_idx;

	if (sequence_length != nullptr)
	{
		if (seq_idx < sequence_length[batch_idx])
		{
			remapped_seq_idx = sequence_length[batch_idx] - 
			                   seq_idx - 1;
		}
		else
		{
			remapped_seq_idx = seq_idx;
		}
	}
	const unsigned batch_state_idx = blockIdx.x *  blockDim.x + 
						      threadIdx.x;

	if (req == OpReqType::kWriteTo || req == OpReqType::kWriteInplace)
	{
		odata[remapped_seq_idx * gridDim.x * blockDim.x + batch_state_idx] = 
			 idata[seq_idx * gridDim.x * blockDim.x + batch_state_idx];
	}
	else if (req == OpReqType::kAddTo)
	{
		odata[remapped_seq_idx * gridDim.x * blockDim.x + batch_state_idx] += 
			 idata[seq_idx * gridDim.x * blockDim.x + batch_state_idx];
	}
}

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet