#include "mlp_att_scoring_func-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (MlpAttScoringFuncParam param, int dtype) 
{
	LOG(FATAL) << "MLP attention layer implemention is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new MlpAttScoringFuncOp < cpu, DType > (param);});

	return op;
}

Operator * MlpAttScoringFuncProp::CreateOperatorEx(Context ctx,
                                                   std::vector < TShape > * in_shape,
                                                   std::vector < int >    * in_type) const
{
 	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(MlpAttScoringFuncParam);

MXNET_REGISTER_OP_PROPERTY(MlpAttScoringFunc, MlpAttScoringFuncProp)
	.describe("Applied the MLP attention non-linear block to the source and query hidden state.")
	.add_argument("QryHidden", "NDArray-or-Symbol",  "Query Hidden State")
	.add_argument("SrcHidden", "NDArray-or-Symbol", "Source Hidden State")
	.add_argument("H2SWeight", "NDArray-or-Symbol", "Hidden-to-Score Weight")
	.add_arguments(MlpAttScoringFuncParam::__FIELDS__());

	} // namespace op
} // namespace mxnet