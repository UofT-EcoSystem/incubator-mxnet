#include "reduce_sum-inl.h"

namespace mxnet {
        namespace op {

template <>
Operator * CreateOp < cpu > (EcoReduceSumParam param, int dtype)
{
        LOG(FATAL) << "Eco-Reduce Sum is only available for GPU at the moment.";

        Operator * op = nullptr;

        MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { 
                op = new EcoReduceSumOp < cpu, DType > (param);
        });
	return op;
}

Operator * EcoReduceSumProp::CreateOperatorEx(Context ctx,
                                              std::vector < TShape > * in_shape,
                                              std::vector < int >    * in_type) const
{
        DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
} 

DMLC_REGISTER_PARAMETER(EcoReduceSumParam);

MXNET_REGISTER_OP_PROPERTY(EcoReduceSum, EcoReduceSumProp)
        .describe("Applied the Eco-ReduceSum operator for faster compute")
        .add_argument ("data", "NDArray-or-Symbol", "Input Data")
        .add_arguments(EcoReduceSumParam::__FIELDS__());        

        }  // namespace op 
}  // namespace mxnet
