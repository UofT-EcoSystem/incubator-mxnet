#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "../operator_common.h"

namespace mxnet {
	namespace op {
		namespace v2 {
			namespace {

enum class EnumOpInputs  { Data, SequenceLength };
enum class EnumOpOutputs { Output };

			} // namespace anonymous

struct SequenceReverseV2Param : public dmlc::Parameter < SequenceReverseV2Param >
{
	unsigned seq_length, batch_size, state_size;

	bool use_sequence_length;

	DMLC_DECLARE_PARAMETER(SequenceReverseV2Param)
	{
		DMLC_DECLARE_FIELD(use_sequence_length).set_default(false)
			.describe("Whether to provide an additional vector indicating "
			          "the sequence length for each batch.");
	}
};

template < typename xpu, typename DType >
class SequenceReverseV2Op : public Operator
{
private:
	SequenceReverseV2Param _param;
public:
	explicit SequenceReverseV2Op(SequenceReverseV2Param param)
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

template<typename xpu>
Operator * CreateOp(SequenceReverseV2Param param, int dtype);

#if DMLC_USE_CXX11

class SequenceReverseV2Prop : public OperatorProperty
{
private:
	SequenceReverseV2Param _param;
public:
	SequenceReverseV2Prop() {}
	explicit SequenceReverseV2Prop(SequenceReverseV2Param param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		if (_param.use_sequence_length)
		{
			return { "data", "sequence_length" };
		}
		else 
		{
			return { "data" };
		}
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return { "output" };
	}
	int NumOutputs() const override
	{
		return 1;
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

		if (_param.use_sequence_length)
		{
			CHECK_EQ(in_shape->size(), 2U); // data, sequence_length
		}
		else
		{
			CHECK_EQ(in_shape->size(), 1U); // data
		}

		// query the input shape and perform shape inference
		const TShape & data_shape = (*in_shape)[int(EnumOpInputs::Data)];

		CHECK_EQ(data_shape.ndim(), 3U) << "Input Data should be rank-3 tensor of dim"
			"[seq length, batch size, state size].";

		if (_param.use_sequence_length)
		{
			unsigned batch_size = data_shape[1];

			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::SequenceLength),
				Shape1(batch_size));
		}
		
		out_shape->clear();

		out_shape->push_back(data_shape); // output

		return true;
	}
	bool InferType (std::vector < int > *  in_type,
	                std::vector < int > * out_type,
			std::vector < int > * aux_type) const override 
	{
		CHECK_GE(in_type->size(), 1U);

		int data_type = (*in_type)[0];

		CHECK_NE(data_type, -1) << "First input must have specified type.";

		for (std::size_t i = 1; i < in_type->size(); ++i)
		{
			if ((*in_type)[i] == -1) 
			{
				(*in_type)[i] = data_type;
			}
			else
			{
				CHECK_EQ((*in_type)[i], data_type) << "This layer requires uniform type. " << 
					"Expected " << data_type << " v.s. given " << 
					(*in_type)[i] << " at " << ListArguments()[i];
			}
		}

		out_type->clear();

		out_type->push_back(data_type); // output
		
		return true;
	}

	OperatorProperty * Copy() const override
	{
		return new SequenceReverseV2Prop(_param);
	}

	std::string TypeString() const override
	{
		return "SequenceReverseV2";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		if (_param.use_sequence_length)
			return {  in_data[int(EnumOpInputs ::SequenceLength)],
				 out_grad[int(EnumOpOutputs::Output)] };
		else
			return { out_grad[int(EnumOpOutputs::Output)] };
	}

	// @TODO Check what are the effects of using inplace options.

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

		}  // namespace v2
	}  // namespace op
}  // namespace mxnet
