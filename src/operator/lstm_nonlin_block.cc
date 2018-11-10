#include "lstm_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (LSTMNonLinBlockParam param, int dtype) 
{
	LOG(FATAL) << "LSTM cell implementation is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new LSTMNonLinBlockOp < cpu, DType > (param);});

	return op;
}

Operator * LSTMNonLinBlockProp::CreateOperatorEx(Context ctx,
                                                 std::vector < TShape > * in_shape,
                                                 std::vector < int >    * in_type) const
{
 	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(LSTMNonLinBlockParam);

MXNET_REGISTER_OP_PROPERTY(LSTMNonLinBlock, LSTMNonLinBlockProp)
	.describe("Applies the LSTM non-linear block to the input and hidden state.")
	.add_argument ("", "")
	.add_argument ("", "")
	.add_argument ("", "")
	.add_arguments(LSTMNonLinBlockParam::__FIELDS__());

	} // namespace op
} // namespace mxnet