#pragma once 

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
        namespace op {
                namespace {

enum class EnumOpInputs  { Output, Mean, InvVar, Gamma, Beta };
enum class EnumOpOutputs { Data };

                }  // anonymous namespace

struct BatchNormInvParam : public dmlc::Parameter < BatchNormInvParam >
{
        DMLC_DECLARE_PARAMETER(BatchNormInvParam);
};

template < typename xpu, typename DType >
class BatchNormInvOp : public Operator
{
private:
        BatchNormInvParam _param;
public:
	explicit BatchNormInvOp(BatchNormInvParam param)
	{
		// empty
	}
	virtual void  Forward(const OpContext & ctx,
	                      const std::vector < TBlob > &  in_data,
			      const std::vector < OpReqType > &  req,
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
Operator * CreateOp(BatchNormInvParam param, int dtype);

#if DMLC_USE_CXX11

class BatchNormInvProp : public OperatorProperty
{
private:
        BatchNormInvParam _param;
public:
	BatchNormInvProp() {}
	explicit
	BatchNormInvProp(BatchNormInvParam param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		return { "output", "mean", "inv_var", "gamma", "beta" };
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return { "data" };
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

		// output, mean, inv_var, gamma, beta
		CHECK_EQ(in_shape->size(), 5); 

		const TShape & oshape = (*in_shape)[int(EnumOpInputs::Output)];
		
		unsigned batch_size = oshape[1];

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Mean),
			Shape1(batch_size));
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::InvVar),
			Shape1(batch_size));
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
		
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]); // input_fm
		out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]); // forget_fm

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
		return { in_data[int(EnumOpInputs ::Input)],
		         in_data[int(EnumOpInputs ::StateH)],
			 in_data[int(EnumOpInputs ::StateC)],
			 in_data[int(EnumOpInputs ::I2HWeight)],
			 in_data[int(EnumOpInputs ::H2HWeight)],
			out_data[int(EnumOpOutputs::StateHOut)],
			out_data[int(EnumOpOutputs::StateCOut)],
			out_data[int(EnumOpOutputs::InputFM)],
			out_data[int(EnumOpOutputs::ForgetFM)],
			out_grad[int(EnumOpOutputs::StateHOut)],
			out_grad[int(EnumOpOutputs::StateCOut)] };
	}

	std::vector < ResourceRequest >  ForwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		return { ResourceRequest::kTempSpace };
	}
	std::vector < ResourceRequest > BackwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		return { ResourceRequest::kTempSpace };
	}

	Operator * CreateOperator  (Context ctx) const override
	{
		LOG(FATAL) << "Not Implemented";

		return nullptr;
	}
	Operator * CreateOperatorEx(Context ctx,
	                            std::vector < TShape > * in_shape,
				    std::vector < int >    * in_type) const override;
};

#endif // DMLC_USE_CXX11

        }  // namespace op
}  // namespace mxnet
