#include "lstm_cell-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (EcoLSTMCellParam param, int dtype)
{
	LOG(FATAL) << "Eco-LSTM Cell is only available for GPU at the moment.";

	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new EcoLSTMCellOp < cpu, DType > (param);});

	return op;
}

Operator * EcoLSTMCellProp::CreateOperatorEx(Context ctx,
                                             std::vector < TShape > * in_shape,
					     std::vector < int >    * in_type) const
{
	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(EcoLSTMCellParam);

MXNET_REGISTER_OP_PROPERTY(EcoLSTMCell, EcoLSTMCellProp)
	.describe("Applies the Eco-LSTM Cell for faster compute and less memory footprint.")
	.add_argument ("input"  , "NDArray-or-Symbol", "Input to the LSTM Cell")
	.add_argument ("state_h", "NDArray-or-Symbol", "Hidden State of the Previous Time Step")
	.add_argument ("state_c", "NDArray-or-Symbol", "Cell ""State of the Previous Time Step")
	.add_argument ("i2h_weight", "NDArray-or-Symbol", "Input-to-Hidden Weight")
	.add_argument ("i2h_bias"  , "NDArray-or-Symbol", "Input-to-Hidden Bias")
	.add_argument ("h2h_weight", "NDArray-or-Symbol", "Hidden-to-Hidden Weight")
	.add_argument ("h2h_bias"  , "NDArray-or-Symbol", "Hidden-to-Hidden Bias")
	.add_arguments(EcoLSTMCellParam::__FIELDS__());

	} // namespace op
} // namespace mxnet
