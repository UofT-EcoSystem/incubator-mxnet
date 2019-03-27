#include "softmax_output_pfp-cu_inl.cuh"

namespace mxnet {
	namespace op {

Operator * CreateOp < gpu > (EcoSoftmaxOutputParam param, int dtype)
{
	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
		op = new CUEcoSoftmaxOutputOp < gpu, DType > (param);
	})
	return op;
}

	} // namespace op
} // namespace mxnet