#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs { Input, StateH, StateC, 
                          I2HWeight, I2HBias,
			  H2HWeight, H2HBias };
enum class EnumOpOutputs { StateHOut, StateCOut };
enum class EnumOpWorkspace { TempSpace };

		} // anonymous namespace

struct EcoLSTMCellParam : public dmlc::Parameter < EcoLSTMCell >
{
	unsigned batch_size, input_size, state_size;

	DMLC_DECLARE_PARAMETER(EcoLSTMCellParam) {}
}

template < typename xpu, typename DType >
class EcoLSTMCellOp : public Operator
{
private:
	EcoLSTMCellParam _param;
public:
	explicit EcoLSTMCellOp(EcoLSTMCellParam param)
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
Operator * CreateOp(LSMTCellParam param, int dtype);

#if DMLC_USE_CXX11

class EcoLSTMCellProp : public OperatorProperty
{
private:
	EcoLSTMCellParam _param;
public:
	EcoLSTMCellProp() {}
	explicit EcoLSTMCellProp(EcoLSTMCellParam param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		return { "input", "state_h", "state_c",
		         "i2h_weight", "i2h_bias",
			 "h2h_weight", "h2h_bias" };
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return { "state_h_out", "state_c_out" };
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

		// input, state_h, state_c
		// i2h_weight, i2h_bias
		// h2h_weight, h2h_bias
		CHECK_EQ(in_shape->size(), 7); 

		const TShape & ishape = (*in_shape)[int(EnumOpInputs::Input )];
		const TShape & hshape = (*in_shape)[int(EnumOpInputs::StateH)];

		CHECK_EQ(ishape.ndim(), 7U) << "Input data should be rank-2 tensor of dim "
			"[batch size, input size]"
		
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
			Shape2(4 * stae_size, state_size));
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2HBias),
			Shape1(4 * state_size));
		
		out_shape->clear();

		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]); // state_h_out
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateC)]); // state_c_out

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
		
		return true;
	}

	OperatorProperty * Copy() const override
	{
		return new EcoLSTMCellProp(_param);
	}

	std::string TypeString() const override
	{
		return "EcoLSTMCell";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		return { in_data[EnumOpInputs ::Input],
		         in_data[EnumOpInputs ::StateH],
			 in_data[EnumOpInputs ::StateC],
			out_data[EnumOpOutputs::StateHOut],
			out_data[EnumOpOutputs::StateCOut],
			out_grad[EnumOpOutputs::StateHOut],
			out_grad[EnumOpOutputs::StateCOut]};
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
}; // class EcoLSTMCellProp

#endif // DMLC_USE_CXX11

	} // namespace op
} // namespace mxnet