#pragma once

#include <vector>

#include <mxnet/storage.h>

#include "mlp_att_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

template < typename DType >
class CUMlpAttNonLinBlockOp : public Operator
{
private:
	MlpAttNonLinBlockParam _param;

	bool _initialized = false;

	// ReserveSpace
	Storage::Handle _reserved_space;
public:
	explicit CUMlpAttNonLinBlockOp(MlpAttNonLinBlockParam param)
	{
		_param = param;
	}
	~CUMlpAttNonLinBlockOp() {}
private:
	void _Init(mshadow::Stream < gpu > * cuda_stream,
	           const std::vector < TBlob > &  in_data,
		   const std::vector < TBlob > & out_data)
	{
		using namespace mshadow;

		CHECK_EQ(_initialized, false);

		Tensor < gpu, 3, DType > src_hidden = in_data[int(EnumOpInputs::SrcHidden)].
			get < gpu, 3, DType > (cuda_stream);

		// infer the parameters from the source hidden state
		_param.batch_size = src_hidden.shape_[0];
		_param.seq_len    = src_hidden.shape_[1];
		_param.state_size = src_hidden.shape_[2];

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

		std::size_t in_expected = 2, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected); // SrcHidden, QryHidden
		CHECK_EQ(out_data.size(), out_expected); // AttHidden

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 3, DType > src_hidden =  in_data[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > qry_hidden =  in_data[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > att_hidden = out_data[int(EnumOpOutputs::AttHidden)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(src_hidden.CheckContiguous(), true);
		CHECK_EQ(qry_hidden.CheckContiguous(), true);
		CHECK_EQ(att_hidden.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}
		
		if (ctx.is_train)
		{
			// TODO: Allocate the reserved space for training.
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

		std::size_t in_expected = 2, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected);
		CHECK_EQ(out_grad.size(), out_expected);

		CHECK_NE(req[int(EnumOpInputs::SrcHidden)], kAddTo) << "AddTo is not supported for " 
			"source hidden state.";
		CHECK_NE(req[int(EnumOpInputs::QryHidden)], kAddTo) << "AddTo is not supported for "  
			 "query hidden state.";
		
		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 3, DType > src_hidden_grad =  in_grad[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > qry_hidden_grad =  in_grad[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > att_hidden_grad = out_grad[int(EnumOpOutputs::AttHidden)]
			.get < gpu, 2, DType > (cuda_stream);

		CHECK_EQ(src_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(qry_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(att_hidden_grad.CheckContiguous(), true);

		// TODO: Backward kernel goes here.
	}
}; // class CUMlpAttNonLinBlockOp

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet