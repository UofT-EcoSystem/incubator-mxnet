#include "sequence_reverse_v2-inl.h"

namespace mxnet {
	namespace op {
		namespace v2 {

template <>
Operator * CreateOp < cpu > (SequenceReverseV2Param param, int dtype)
{
	LOG(FATAL) << "Operator SequenceReverseV2 is only available for GPU at the moment.";

	Operator * op = nullptr;
  	
	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new SequenceReverseV2Op < cpu, DType > (param);});

	return op;
}

Operator * SequenceReverseV2Prop::CreateOperatorEx(Context ctx,
        std::vector < TShape > * in_shape,
        std::vector < int >    * in_type) const
{
 	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SequenceReverseV2Param);

MXNET_REGISTER_OP_PROPERTY(SequenceReverseV2, SequenceReverseV2Prop)
	.describe("Apply parallel sequence reverse to the input sequence.")
	.add_argument("data",            "NDArray-or-Symbol", "Input Data")
	.add_argument("sequence_length", "NDArray-or-Symbol", "Sequence Length")
	.add_arguments(SequenceReverseV2Param::__FIELDS__());

		}  // namespace v2
	}  // namespace op
}  // namespace mxnet
