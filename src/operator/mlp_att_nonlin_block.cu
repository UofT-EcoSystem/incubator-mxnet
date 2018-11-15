#include "cu_mlp_att_nonlin_block-inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (MlpAttNonLinBlockParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUMlpAttNonLinBlockOp < DType > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet