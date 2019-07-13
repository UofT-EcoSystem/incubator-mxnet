#include "reduce_sum-cu_inl.h"

namespace mxnet {
        namespace op {

template <>
Operator * CreateOp < gpu > (EcoReduceSumParam param, int dtype)
{
        Operator * op = nullptr;

        MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, {
                op = new CUEcoReduceSumOp < float, false > (param);
        });
        return op;
}

        }  // namespace op
}  // namespace mxnet