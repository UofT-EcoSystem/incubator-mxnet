#include "mlp_att_scoring_func-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (MLPAttScoringFuncParam param, int dtype) 
{
	LOG(FATAL) << "MLP Attention Scoring Function is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new MLPAttScoringFuncOp < cpu, DType > (param);});

	return op;
}

Operator * MLPAttScoringFuncProp::CreateOperatorEx(Context ctx,
                                                   std::vector < TShape > * in_shape,
                                                   std::vector < int >    * in_type) const
{
 	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(MLPAttScoringFuncParam);

MXNET_REGISTER_OP_PROPERTY(MLPAttScoringFunc, MLPAttScoringFuncProp)
	.describe("Apply the MLP attention scoring function to the query and source hidden state.")
	.add_argument("qry_hidden", "NDArray-or-Symbol",  "Query Hidden State")
	.add_argument("src_hidden", "NDArray-or-Symbol", "Source Hidden State")
	.add_argument("h2s_weight", "NDArray-or-Symbol", "Hidden-to-Score Weight")
	.add_argument("gamma", "NDArray-or-Symbol", "Layer Normalization Coefficient (Gamma)")
	.add_argument("beta",  "NDArray-or-Symbol", "Layer Normalization Coefficient (Beta)")
	.add_arguments(MLPAttScoringFuncParam::__FIELDS__());

	} // namespace op
} // namespace mxnet
