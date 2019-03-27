#include "softmax_output_pfp-inl.h"

namespace mxnet {
	namespace op {

Operator * CreateOp < gpu > (EcoSoftmaxOutputParam param, int dtype)
{
	LOG(FATAL) << "Eco-Softmax Output is only available for GPU at the moment.";

	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new EcoSoftmaxOutputOp < cpu, DType > (param);});

	return op;
}

Operator * EcoSoftmaxOutputProp::CreateOperatorEx(Context ctx,
                                                  std::vector < TShape > * in_shape,
						  std::vector < int >    * in_type) const
{
	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(EcoSoftmaxOutputParam);

MXNET_REGISTER_OP_PROPERTY(EcoSoftmaxOutput, EcoSoftmaxOutputProp)
	.describe("Applies the Eco-Softmax Output for less memory footprint.")
	.add_argument ("data",  "NDArray-or-Symbol", "Logits")
	.add_argument ("label", "NDArray-or-Symbol", "Ground Truth Label")
	.add_arguments(EcoSoftmaxOutputParam::__FIELDS__());

	} // namespace op
} // namespace mxnet