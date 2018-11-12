#include "lstm_nonlin_block-inl.h"
#include "cu_lstm_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LSTMNonLinBlockParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = nullptr; });

	return op;
}

	} // namespace op
} // namespace mxnet