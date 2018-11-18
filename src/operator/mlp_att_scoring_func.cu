#include "cu_mlp_att_scoring_func-inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (MlpAttScoringFuncParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUMlpAttScoringFuncOp < DType > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet