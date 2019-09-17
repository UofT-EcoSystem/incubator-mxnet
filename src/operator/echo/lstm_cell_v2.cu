#include "lstm_cell_v2-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LSTMCellV2Param param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CULSTMCellV2Op < float > (param); });

	return op;
}

	}  // namespace op
}  // namespace mxnet
