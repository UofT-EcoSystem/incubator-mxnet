#pragma once

// #include <vector>

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  { Input, StateH, StateC };
enum class EnumOpOutputs { StateHOut, StateCOut,
                           InputFM, ForgetFM,
			   IActvFM, OutputFM };
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

template < typename xpu >
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
		return { "input", "state_h", "state_c" };
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return { "state_h_out", "state_c_out",
		         "input_fm", "forget_fm",
			 "iactv_fm", "output_fm" };
	}
	int NumOutputs() const override 
	{
		return 6;
  	}
	int NumVisibleOutputs() const override 
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

		CHECK_EQ(in_shape->size(), 3U); // input, state_h, state_c

		const TShape & ishape = (*in_shape)[int(EnumOpInputs::Input)];

		CHECK_EQ(ishape.ndim(), 2U) << "Input data should be rank-2 tensor of dim "
			"[batch size, 4 * state size].";

		unsigned batch_size = ishape[0];
		unsigned state_size = ishape[1] / 4;

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::StateH), 
			Shape2(batch_size, 4 * state_size));
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::StateC),
			Shape2(batch_size,     state_size));

		out_shape->clear();

		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // state_h_out
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // state_c_out
		
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // input_fm
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // forget_fm
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // iactv_fm
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // output_fm

		return true;
	}
	bool InferType (std::vector < int > *  in_type,
	                std::vector < int > * out_type,
	                std::vector < int > * aux_type) const override
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
					(*in_type)[i] << " at " << ListArguments()[i];
			}
		}

		out_type->clear();

		out_type->push_back(itype); // state_h_out
		out_type->push_back(itype); // state_c_out

		out_type->push_back(itype); // input_fm
		out_type->push_back(itype); // forget_fm
		out_type->push_back(itype); // iactv_fm
		out_type->push_back(itype); // output_fm
		
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
		return { in_data[int(EnumOpInputs ::StateC)],
			out_data[int(EnumOpOutputs::InputFM)],
			out_data[int(EnumOpOutputs::ForgetFM)],
			out_data[int(EnumOpOutputs::IActvFM)],
			out_data[int(EnumOpOutputs::OutputFM)],
			out_grad[int(EnumOpOutputs::StateHOut)],
			out_grad[int(EnumOpOutputs::StateCOut)] };
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
