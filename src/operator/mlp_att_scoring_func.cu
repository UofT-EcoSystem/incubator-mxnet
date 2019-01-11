#include "mlp_att_scoring_func-cu_inl.cuh"

namespace mxnet {
	namespace op {

template <> 
Operator * CreateOp < gpu > (MlpAttScoringFuncParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUMlpAttScoringFuncOp < float > (param); });

	return op;
}

	} // namespace op
} // namespace mxnet