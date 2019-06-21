#include "gru_cell-cu_inl.cuh"

namespace mxnet {
        namespace op {

template <>
Operator * CreateOp < gpu > (InvisGRUCellParam param, int dtype)
{
        Operator * op = nullptr;

        MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, { op = new CUInvisGRUCellOp < float > (param); });

        return op;
}

        }  // namespace op
}  // namespace mxnet
