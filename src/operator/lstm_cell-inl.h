#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>

#include <mxnet/operator.h>

#include <vector>

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  {CellInput, HiddenState, CellState};
enum class EnumOpOutputs            {HiddenState, CellState};
// NO Need for Workspace for Reserve Space

struct LSTMCellParam : public dmlc::Parameter < LSTMCellParam >
{
	// All parameters do not require explicit declaration.
	// The reason is because they can be effectively inferred from the shape of the input data.
	std::uint32_t batch_size;
	std::uint32_t state_size;
};

template < typename xpu, typename DType >
class LSTMCellOp : public Operator
{
private:
	LSTMCellParam _param;
public:
	explicit LSTMCellOp(LSTMCellParam param)
	{
		// empty
	}
	virtual void  Forward(const OpContext & ctx,
	                      const std::vector < TBlob > & in_data,
			      const std::vector < OpReqType > & req,
			      const std::vector < TBlob > & out_data,
			      const std::vector < TBlob > & aux_data)
	{
		// empty
	}
	virtual void Backward(const OpContext & ctx,
	                      const std::vector < TBlob > & out_grad,
			      const std::vector < TBlob > &  in_data,
			      const std::vector < TBlob > & out_data,
			      const std::vector < OpReqType > &  req,
			      const std::vector < TBlob > &  in_grad,
			      const std::vector < TBlob > & aux_args)
	{
		// empty
	}
};

template<typename xpu>
Operator * CreateOp(LSTMCellParam param, int dtype);

#if DMLC_USE_CXX11

class LSTMCellProp : public OperatorProperty
{
private:
public:
	std::vector < std::string > ListArguments() const override
	{
		return {"CellInput", "HiddenState", "CellState"};
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return {"HiddenState", "CellState"};
	}

	int NumOutputs() const override 
	{
		return 2;
  	}

	void Init(const std::vector < std::pair < std::string, 
	                                          std::string > > & kwargs) override
	{
		param_.Init(kwargs);
	}

	std::map < std::string, std::string > GetParams() const override
	{
		return param_.__DICT__();
	}

	bool InferShape(std::vector < TShape > *  in_shape,
	                std::vector < TShape > * out_shape,
	                std::vector < TShape > * aux_shape) const override
	{
		using namespace mshadow;

		CHECK_EQ(in_shape->size(), 3U);

		// query the input shape and perform shape inference
		const TShape & ishape = (*in_shape)[EnumOpInputs::CellInput];

		if (dshape.ndim() ==  0) return false;
CHECK_EQ(dshape.ndim(), 3U) \
<< "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]";

		int state_size = ishape[]

int batch_size = dshape[1];
int input_size = dshape[2];
int numDirections = param_.bidirectional ? 2 : 1;
int total_layers = numDirections * param_.num_layers;  // double for bidirectional
SHAPE_ASSIGN_CHECK(*in_shape,
rnn_enum::kState,
Shape3(total_layers, batch_size, param_.state_size));
if (param_.mode == rnn_enum::kLstm)
SHAPE_ASSIGN_CHECK(*in_shape,
rnn_enum::kStateCell,
Shape3(total_layers, batch_size, param_.state_size));

// calculate parameter vector length
int param_size = rnn_param_size(param_.num_layers,
input_size,
param_.state_size,
param_.bidirectional,
param_.mode);
SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kParams, Shape1(param_size));

out_shape->clear();
// output: [sequence len, batch, output size]
TShape oshape = dshape;
oshape[2] = numDirections * param_.state_size;
out_shape->push_back(oshape);
if (!param_.state_outputs) {
return true;
} else {
// outStateShape: [layer_num, batch, state size]
TShape outStateShape = dshape;
outStateShape[0] = total_layers;
outStateShape[1] = batch_size;
outStateShape[2] = param_.state_size;
out_shape->push_back(outStateShape);
// Deal with lstm cell state
if (param_.mode == rnn_enum::kLstm)
out_shape->push_back(outStateShape);
return true;
}
}

	bool InferType(std::vector < int > *  in_type,
	               std::vector < int > * out_type,
	               std::vector < int > * aux_type) const override
	{
		CHECK_GE(in_type->size(), 1U);

		int itype = (*in_type)[0]; // query the input data type

		CHECK_NE(itype, -1) << "First input must have specified type.";

		for (std::size_t i = 0; i < in_type->size(); ++i)
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

CHECK_NE(dtype, -1) << "First input must have specified type";
for (index_t i = 0; i < in_type->size(); ++i) {
if ((*in_type)[i] == -1) {
(*in_type)[i] = dtype;
} else {
CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
<< "Expected " << dtype << " v.s. given "
<< (*in_type)[i] << " at " << ListArguments()[i];
}
}
out_type->clear();
out_type->push_back(dtype);
if (!param_.state_outputs) {
return true;
} else {
out_type->push_back(dtype);
// Deal with lstm cell state
if (param_.mode == rnn_enum::kLstm)
out_type->push_back(dtype);
			return true;
		}
	}

	OperatorProperty * Copy() const override
	{
		return new LSTMCellProp(_param);
	}

	std::string TypeString() const override
	{
		return "LSTMCell";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		/**
		 * According to the documentation description 
		 * 	Available here: https://mxnet.incubator.apache.org/doxygen/classmxnet_1_1OperatorProperty.html#abf9e6a8d40750f3ee81fe30cbe3e2aae
		 * It can be inferred that this method is used for memory optimization purpose,
		 * that is, ONLY variables that are returned in the list of dependencies
		 * will be preserved by the forward pass for use in the backward pass.
		 */
		// Note that here we deliberately ignore the `out_data`.
		return { in_data[EnumOpInputs ::  CellInput],
		         in_data[EnumOpInputs ::HiddenState],
			 in_data[EnumOpInputs ::  CellState],
			out_grad[EnumOpOutputs::HiddenState],
			out_grad[EnumOpOutputs::  CellState]};
	}

	std::vector < ResourceRequest >  ForwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		return {};
	}
	std::vector < ResourceRequest > BackwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		return {};
	}

	Operator * CreateOperator(Context ctx) const override
	{
		LOG(FATAL) << "Not Implemented";
		
		return nullptr;
  	}
	Operator* CreateOperatorEx(Context ctx, 
				   std::vector < TShape > * in_shape,
        			   std::vector < int >    * in_type) const override;
}; // class LSTMCellProp

#endif // DMLC_USE_CXX11

		} // namespace 
	} // namespace op
} // namespace mxnet