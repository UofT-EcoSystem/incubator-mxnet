#include "sequence_reverse_v2-cu_inl.cuh"

namespace mxnet {
	namespace op {
		namespace v2 {

template <> 
Operator * CreateOp < gpu > (SequenceReverseV2Param param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUSequenceReverseV2Op < float > (param); });

	return op;
}

		}  // namespace v2
	}  // namespace op
}  // namespace mxnet
