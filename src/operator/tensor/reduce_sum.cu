#include "reduce_sum-cu_inl.h"

namespace mxnet {
        namespace op {

template <>
Operator * CreateOp < gpu > (EcoReduceSumParam param, int dtype,
        bool do_normalization)
{
        Operator * op = nullptr;

        if (do_normalization)
        {
                MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, {
                        op = new CUEcoReduceSumOp < float,  true > (param);
                });
        }
        else
        {
                MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, {
                        op = new CUEcoReduceSumOp < float, false > (param);
                });
        }
        return op;
}

        }  // namespace op
}  // namespace mxnet
