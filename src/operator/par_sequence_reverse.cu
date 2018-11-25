#include "cu_par_sequence_reverse-inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (ParSequenceReverseParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUParSequenceReverseOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet