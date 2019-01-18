#include "lstm_cell-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LSTMCellParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CULSTMCellOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet