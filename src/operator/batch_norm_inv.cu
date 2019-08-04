#include "batch_norm_inv-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (BatchNormInvParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUBatchNormInvOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet
