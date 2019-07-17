#include "layer_norm-cu_inl.h"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (LayerNormParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CULayerNormOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet
