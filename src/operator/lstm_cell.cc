#include "./lstm_cell-cu_inl.cuh"

namespace mxnet {
	namespace op {

DMLC_REGISTER_PARAMETER(EcoLSTMCellParam);

NNVM_REGISTER_OP(EcoLSTMCell)
	.set_attr_parser(ParamParser < EcoLSTMCellParam > )
	.set_num_inputs (7)
  	.set_num_outputs(2)
	.set_attr < nnvm::FListInputNames > ("FListInputNames",  
		[](const nnvm::NodeAttrs & attrs)
		{
			return { "input", "state_h", "state_c",
		                 "i2h_weight", "i2h_bias",
			         "h2h_weight", "h2h_bias" };
		})
	.set_attr < nnvm::FListOutputNames > ("FListOutputNames", 
		[](const nnvm::NodeAttrs & attrs)
		{
			return { "state_h_out", "state_c_out" };
		})
	.set_attr < nnvm::FInferShape > ("FInferShape", 
		[](const nnvm::NodeAttrs & attrs,
		   std::vector < TShape > *  in_shape,
		   std::vector < TShape > * out_shape)
		{
			using namespace mshadow;

			CHECK_EQ(in_shape->size(), 7);
			
			const TShape & ishape = (*in_shape)[int(EnumOpInputs::Input )];
			const TShape & hshape = (*in_shape)[int(EnumOpInputs::StateH)];

			CHECK_EQ(ishape.ndim(), 2U) <<   "Input data should be rank-2 tensor of dim "
				"[batch size, input size]";
			CHECK_EQ(hshape.ndim(), 2U) << "Hidden state should be rank-2 tensor of dim "
				"[batch size, state size]";
			
			unsigned batch_size = ishape[0];
			unsigned input_size = ishape[1];
			unsigned state_size = hshape[1];

			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::StateH),
				Shape2(batch_size, state_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::StateC),
				Shape2(batch_size, state_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::I2HWeight),
				Shape2(4 * state_size, input_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::I2HBias),
				Shape1(4 * state_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2HWeight),
				Shape2(4 * state_size, state_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2HBias),
				Shape1(4 * state_size));
			
			out_shape->clear();

			out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]); // state_h_out
			out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // state_c_out

			return true;
		})
	.set_attr < nnvm::FInferType > ("FInferType",
		[](const nnvm::NodeAttrs & attrs,
		   std::vector < int > *  in_type,
		   std::vector < int > * out_type)
		{
			using namespace mshadow;

			CHECK_GE(in_type->size(), 1U);

			int itype = (*in_type)[0];

			CHECK_NE(itype, -1) << "First input must have specified type.";

			for (std::size_t i = 1; i < in_type->size(); ++i)
			{
				if ((*in_type)[i] == -1) 
				{
					(*in_type)[i] = itype;
				}
				else
				{
					CHECK_EQ((*in_type)[i], itype) << "This layer requires uniform type. " << 
						"Expected " << itype << " v.s. given " << 
						(*in_type)[i] << " at " << ListArguments()[i];
				}
			}

			out_type->clear();

			out_type->push_back(itype); // state_h_out
			out_type->push_back(itype); // state_c_out
			
			return true;
		})
	.set_attr < FResourceRequest > ("FResourceRequest",
		[](const nnvm::NodeAttrs & attrs)
		{
			return { ResourceRequest::kTempSpace };
		})
	.set_attr < nnvm::FGradient > ("FGradient",
		[](const NodePtr & n,
    		   const std::vector < NodeEntry > & ograds)
		{
			return MakeNonlossGradNode("_backward_EcoLSTMCell", n, ograds, 
				{ n->inputs[int(EnumOpInputs::Input)],
				  n->inputs[int(EnumOpInputs::StateH)],
				  n->inputs[int(EnumOpInputs::StateC)],
				  n->inputs[int(EnumOpInputs::I2HWeight)],
				  n->inputs[int(EnumOpInputs::H2HWeight)],
				  nnvm::NodeEntry{n, int(EnumOpOutputs::StateHOut), 0},
				  nnvm::NodeEntry{n, int(EnumOpOutputs::StateCOut), 0},
				n->attrs.dict});
		})
	.set_attr < FCompute > ("FCompute<cpu>", EcoLSTMCellCompute < cpu > )
	.add_argument ("input", "NDArray-or-Symbol", "Input to the LSTM Cell")
	.add_argument ("state_h", "NDArray-or-Symbol", "Hidden State of the Previous Time Step")
	.add_argument ("state_c", "NDArray-or-Symbol", "Cell ""State of the Previous Time Step")
	.add_argument ("i2h_weight", "NDArray-or-Symbol", "Input-to-Hidden Weight")
	.add_argument ("i2h_bias"  , "NDArray-or-Symbol", "Input-to-Hidden Bias")
	.add_argument ("h2h_weight", "NDArray-or-Symbol", "Hidden-to-Hidden Weight")
	.add_argument ("h2h_bias"  , "NDArray-or-Symbol", "Hidden-to-Hidden Bias")
	.add_arguments(EcoLSTMCellParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_EcoLSTMCell)
	.set_attr_parser(ParamParser < EcoLSTMCellParam > )
	.set_num_inputs (2)
	.set_num_outputs(7)
	.set_attr < nnvm::FListOutputNames > ("FListOutputNames",  
		[](const nnvm::NodeAttrs & attrs)
		{
			return { "input", "state_h", "state_c",
		                 "i2h_weight", "i2h_bias",
			         "h2h_weight", "h2h_bias" }
		})
	.set_attr < FResourceRequest > ("FResourceRequest",
		[](const nnvm::NodeAttrs & attrs)
		{
			return { ResourceRequest::kTempSpace };
		})
	.set_attr < FCompute > ("FCompute<cpu>", EcoLSTMCellGradCompute < cpu > )

	} // namespace op
} // namespace mxnet