#include "batch_norm_inv-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (BatchNormInvParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUBatchNormInvOp < float > (param); });

	return op;
}

void BatchNormInvComputeGPU(const nnvm::NodeAttrs & attrs,
                            const OpContext & ctx,
                            const std::vector < TBlob > &  in_data,
                 	    const std::vector < OpReqType > &  req,
                 	    const std::vector < TBlob > & out_data)
{
	using namespace mshadow;
	typedef float DType;

	std::size_t in_expected = 5, out_expected = 1;

	CHECK_EQ( in_data.size(),  in_expected); // output, mean, inv_var, 
						 // gamma, beta
	CHECK_EQ(out_data.size(), out_expected); // data

	Stream < gpu > * cuda_stream = ctx.get_stream < gpu > (); 

	TBlob output  =  in_data[int(EnumOpInputs ::Output)];
	TBlob mean    =  in_data[int(EnumOpInputs ::Mean)];
	TBlob inv_var =  in_data[int(EnumOpInputs ::InvVar)];
	TBlob gamma   =  in_data[int(EnumOpInputs ::Gamma)];
	TBlob beta    =  in_data[int(EnumOpInputs ::Beta)];
	TBlob data    = out_data[int(EnumOpOutputs::Data)];

	CHECK_EQ(output .CheckContiguous(), true);
	CHECK_EQ(mean   .CheckContiguous(), true);
	CHECK_EQ(inv_var.CheckContiguous(), true);
	CHECK_EQ(gamma  .CheckContiguous(), true);
	CHECK_EQ(beta   .CheckContiguous(), true);
	CHECK_EQ(data   .CheckContiguous(), true);

	_cuda_batch_norm_inv_forward < DType >
		<<<
			(output.shape_.Size() - 1) / 128 + 1, 128, 0,
			Stream < gpu > ::GetStream(cuda_stream)
		>>>
		(
			reinterpret_cast < DType * > (output .dptr_),
			reinterpret_cast < DType * > (mean   .dptr_),
			reinterpret_cast < DType * > (inv_var.dptr_),
			reinterpret_cast < DType * > (gamma  .dptr_),
			reinterpret_cast < DType * > (beta   .dptr_),
			reinterpret_cast < DType * > (data   .dptr_),
			output.shape_.Size(),
			output.shape_[1],
			output.shape_.Size() / (output.shape_[0] * 
				 		output.shape_[1])
		);
}

NNVM_REGISTER_OP(BatchNormInv)
	.set_attr<FCompute>("FCompute<gpu>", BatchNormInvComputeGPU);

	} // namespace op
} // namespace mxnet
