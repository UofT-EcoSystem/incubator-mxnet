#include "./lstm_cell-cu_inl.cuh"

namespace mxnet {
	namespace op {
		
NNVM_REGISTER_OP(EcoLSTMCell)
	.set_attr < FCompute > ("FCompute<gpu>", EcoLSTMCellCompute < gpu > )

NNVM_REGISTER_OP(_backward_EcoLSTMCell)
	.set_attr < FCompute > ("FCompute<gpu>", EcoLSTMCellGradCompute < gpu > )

	} // namespace op
} // namespace mxnet