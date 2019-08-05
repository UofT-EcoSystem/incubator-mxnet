#include "batch_norm_inv-inl.h"

namespace mxnet {
	namespace op {

template <>
Operator * CreateOp < cpu > (BatchNormInvParam param, int dtype)
{
	LOG(FATAL) << "BatchNormInv is only available for GPU at the moment.";

	Operator * op = nullptr;

	MSHADOW_REAL_TYPE_SWITCH(dtype, DType, { op = new BatchNormInvOp < cpu, DType > (param);});

	return op;
}

Operator * BatchNormInvProp::CreateOperatorEx(Context ctx,
                                              std::vector < TShape > * in_shape,
					      std::vector < int >    * in_type) const
{
	DO_BIND_DISPATCH(CreateOp, _param, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BatchNormInvParam);

bool BatchNormInvInferShape(const nnvm::NodeAttrs & attrs,
	std::vector < TShape > *  in_shape,
	std::vector < TShape > * out_shape)
{
	using namespace mshadow;

	// output, mean, inv_var, gamma, beta
	CHECK_EQ(in_shape->size(), 5); 

	const TShape & oshape = (*in_shape)[int(EnumOpInputs::Output)];
	
	unsigned batch_size = oshape[1];

	SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Mean),   Shape1(batch_size));
	SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::InvVar), Shape1(batch_size));
	SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Gamma),  Shape1(batch_size));
	SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Beta),   Shape1(batch_size));
	
	out_shape->clear();

	out_shape->push_back((*in_shape)[int(EnumOpInputs::Output)]); // Data 

	return true;
}

bool BatchNormInvInferType (const nnvm::NodeAttrs & attrs,
	std::vector < int > *  in_type,
	std::vector < int > * out_type)
{
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
				(*in_type)[i] << " at " << i;
		}
	}

	out_type->clear();

	out_type->push_back(itype); // Data
	
	return true;
}

// MXNET_REGISTER_OP_PROPERTY(BatchNormInv, BatchNormInvProp)
NNVM_REGISTER_OP(BatchNormInv)
	.set_num_inputs (5)
	.set_num_outputs(1)
	.set_attr_parser(ParamParser<BatchNormInvParam>)
	.set_attr<nnvm::FInferShape>("FInferShape", BatchNormInvInferShape)
  	.set_attr<nnvm::FInferType> ("FInferType",  BatchNormInvInferType)
  	.set_attr<nnvm::FInplaceOption>("FInplaceOption",
    		[](const NodeAttrs & attrs){
      			return std::vector < std::pair < int, int > >{{0, 0}};
    		})
	.set_attr<nnvm::FListInputNames>("FListInputNames",
  		[](const NodeAttrs& attrs) {
    			return std::vector < std::string > 
			    	{ "output", "mean", "inv_var", "gamma", "beta" };
  		})
	.set_attr<nnvm::FListInputNames>("FListOutputNames",
  		[](const NodeAttrs& attrs) {
    			return std::vector < std::string > { "data" };
  		})
	.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
	.describe("Applies the BatchNorm inverse operator.")
	.add_argument ("output",  "NDArray-or-Symbol", "Output")
	.add_argument ("mean",    "NDArray-or-Symbol", "Mean")
	.add_argument ("inv_var", "NDArray-or-Symbol", "Inverse Variance")
	.add_argument ("gamma",   "NDArray-or-Symbol", "Gamma")
	.add_argument ("beta",    "NDArray-or-Symbol", "Beta")
	.add_arguments(BatchNormInvParam::__FIELDS__());

	} // namespace op
} // namespace mxnet
