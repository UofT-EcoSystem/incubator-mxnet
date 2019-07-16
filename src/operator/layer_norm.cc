#include "layer_norm-inl.h"

namespace mxnet {
        namespace op {

template <>
Operator * CreateOp < cpu > (LayerNormParam param, int dtype) 
{
	LOG(FATAL) << "Layer normalization is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new LayerNormOp < cpu, DType > (param);});

	return op;
}

Operator * LayerNormOp::CreateOperatorEx(Context ctx,
                                         std::vector < TShape > * in_shape,
                                         std::vector < int >    * in_type) const
{
        DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(LayerNormParam);

MXNET_REGISTER_OP_PROPERTY(LayerNorm, LayerNormProp)
        .describe("Apply layer normalization to the hidden state.")
        .add_argument ("data",  "Input Data")
        .add_argument ("gamma", "Layer Normalization Coefficient")
        .add_argument ("beta",  "Layer Normalization Coefficient")
        .add_arguments(LayerNormParam::__FIELDS__());

        }  // namespace op
}  // namespace mxnet
