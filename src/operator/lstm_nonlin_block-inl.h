#pragma once

// #include <vector>

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  { CellInput, HiddenState, CellState };
enum class EnumOpOutputs            { HiddenState, CellState };
// NO Need for Temporary Workspace

		} // anonymous namespace

struct LSTMNonLinBlockParam : public dmlc::Parameter < LSTMNonLinBlockParam >
{
	unsigned batch_size, state_size;

	DMLC_DECLARE_PARAMETER(LSTMNonLinBlockParam) {}
};

template < typename xpu, typename DType >
class LSTMNonLinBlockOp : public Operator
{
private:
	LSTMNonLinBlockParam _param;
public:
	explicit LSTMNonLinBlockOp(LSTMNonLinBlockParam param)
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
Operator * CreateOp(LSTMNonLinBlockParam param, int dtype);

#if DMLC_USE_CXX11

class LSTMNonLinBlockProp : public OperatorProperty
{
private:
	LSTMNonLinBlockParam _param;
public:
	LSTMNonLinBlockProp() {}
	explicit LSTMNonLinBlockProp(LSTMNonLinBlockParam param) : _param(param) {}

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
		_param.Init(kwargs);
	}
	std::map < std::string, std::string > GetParams() const override
	{
		return _param.__DICT__();
	}

	bool InferShape(std::vector < TShape > *  in_shape,
	                std::vector < TShape > * out_shape,
	                std::vector < TShape > * aux_shape) const override
	{
		using namespace mshadow;

		CHECK_EQ(in_shape->size(), 3U); // CellInput, HiddenState, CellState

		// query the input shape and perform shape inference
		const TShape & ishape = (*in_shape)[int(EnumOpInputs::CellInput)];

		CHECK_EQ(ishape.ndim(), 2U) << "Input data should be rank-2 tensor of dim "
			"[batch size, 4 * state size].";

		unsigned batch_size = ishape[0];
		unsigned state_size = ishape[1] / 4;

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::HiddenState), 
			Shape2(batch_size, 4 * state_size));
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::  CellState),
			Shape2(batch_size,     state_size));

		out_shape->clear();

		out_shape->push_back((*in_shape)[int(EnumOpInputs::CellState)]); // HiddenState
		out_shape->push_back((*in_shape)[int(EnumOpInputs::CellState)]); //   CellState

		return true;
	}
	bool InferType (std::vector < int > *  in_type,
	                std::vector < int > * out_type,
	                std::vector < int > * aux_type) const override
	{
		CHECK_GE(in_type->size(), 1U);

		int itype = (*in_type)[0]; // query the input data type

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

		out_type->push_back(itype); // HiddenState
		out_type->push_back(itype); //   CellState
		
		return true;
	}

	OperatorProperty * Copy() const override
	{
		return new LSTMNonLinBlockProp(_param);
	}

	std::string TypeString() const override
	{
		return "LSTMNonLinBlock";
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
		return {
			// in_data[int(EnumOpInputs ::  CellState)],
			out_grad[int(EnumOpOutputs::HiddenState)],
			out_grad[int(EnumOpOutputs::  CellState)]};
	}

	/*
	std::vector < std::pair < int, void * > >  ForwardInplaceOption(
		const std::vector < int >    &  in_data, 
		const std::vector < void * > & out_data) const override
	{
		return {std::make_pair(in_data[int(EnumOpInputs ::HiddenState)], 
		                      out_data[int(EnumOpOutputs::HiddenState)]),
			std::make_pair(in_data[int(EnumOpInputs ::  CellState)],
			              out_data[int(EnumOpOutputs::  CellState)])};
	}

	std::vector < std::pair < int, void * > > BackwardInplaceOption(
		const std::vector < int >    & out_grad,
		const std::vector < int >    &  in_data,
		const std::vector < int >    & out_data,
		const std::vector < void * > &  in_grad) const override
	{
		return {std::make_pair(out_grad[int(EnumOpOutputs::HiddenState)],
		                        in_grad[int(EnumOpInputs ::HiddenState)]),
			std::make_pair(out_grad[int(EnumOpOutputs::  CellState)],
			                in_grad[int(EnumOpInputs ::  CellState)])};
	}
	 */

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

	Operator * CreateOperator  (Context ctx) const override
	{
		LOG(FATAL) << "Not Implemented";
		
		return nullptr;
  	}
	Operator * CreateOperatorEx(Context ctx, 
				    std::vector < TShape > * in_shape,
        			    std::vector < int >    * in_type) const override;
}; // class LSTMNonLinBlockProp

#endif // DMLC_USE_CXX11

	} // namespace op
} // namespace mxnet