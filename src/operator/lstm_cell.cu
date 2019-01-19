#include "lstm_cell-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (EcoLSTMCellParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUEcoLSTMCellOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet