#include "cu_lstm_nonlin_block-inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LSTMNonLinBlockParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CULSTMNonLinBlockOp < DType > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet