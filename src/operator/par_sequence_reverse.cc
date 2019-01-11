#include "par_sequence_reverse-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (ParSequenceReverseParam param, int dtype)
{
	LOG(FATAL) << "Parallel Sequence Reverse operator is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new ParSequenceReverseOp < cpu, DType > (param);});

	return op;
}

Operator * ParSequenceReverseProp::CreateOperatorEx(Context ctx,
                                                    std::vector < TShape > * in_shape,
                                                    std::vector < int >    * in_type) const
{
 	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(ParSequenceReverseParam);

MXNET_REGISTER_OP_PROPERTY(ParSequenceReverse, ParSequenceReverseProp)
	.describe("Apply parallel sequence reverse to the input sequence.")
	.add_argument("data",            "NDArray-or-Symbol", "Input Data")
	.add_argument("sequence_length", "NDArray-or-Symbol", "Sequence Length")
	.add_arguments(ParSequenceReverseParam::__FIELDS__());

	} // namespace op
} // namespace mxnet