#include "batch_norm_inv-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (BatchNormInvParam param, int dtype)
{
	LOG(FATAL) << "BatchNormInv is only available for GPU at the moment.";

	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new BatchNormInvOp < cpu, DType > (param);});

	return op;
}

Operator * BatchNormInvProp::CreateOperatorEx(Context ctx,
                                              std::vector < TShape > * in_shape,
					      std::vector < int >    * in_type) const
{
	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BatchNormInvParam);

MXNET_REGISTER_OP_PROPERTY(BatchNormInv, BatchNormInvProp)
	.describe("Applies the BatchNormInv operator.")
	.add_argument ("output",  "NDArray-or-Symbol", "Output")
	.add_argument ("mean",    "NDArray-or-Symbol", "Mean")
	.add_argument ("inv_var", "NDArray-or-Symbol", "Inverse Variance")
	.add_argument ("gamma",   "NDArray-or-Symbol", "Gamma")
	.add_argument ("beta",    "NDArray-or-Symbol", "Beta")
	.add_arguments(BatchNormInvParam::__FIELDS__());

	} // namespace op
} // namespace mxnet
