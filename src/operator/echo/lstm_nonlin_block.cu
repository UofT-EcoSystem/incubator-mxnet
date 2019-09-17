#include "lstm_nonlin_block-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LSTMNonLinBlockParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CULSTMNonLinBlockOp < float > (param); });

	return op;
}

	}  // namespace op
}  // namespace mxnet
