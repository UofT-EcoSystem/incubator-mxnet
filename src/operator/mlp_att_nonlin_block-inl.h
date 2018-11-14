#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  {SrcHidden, QueryHidden};
enum class EnumOpOutputs {AttHidden};
// NO Need for Temporary Workspace

		} // namespace 

struct MlpAttNonLinBlockParam : public dmlc::Parameter < MlpAttNonLinBlockParam >
{
	// All parameters do not require explicit declaration.
	// The reason is because they can be effectively inferred from the shape of the input data.
	unsigned batch_size, seq_len, state_size;

	DMLC_DECLARE_PARAMETER(MlpAttNonLinBlockParam) {}
};

template < typename xpu, typename DType >
class MlpAttNonLinBlockOp : public Operator
{
private:
	MlpAttNonLinBlockParam _param;
public:
	explicit MlpAttNonLinBlockOp(MlpAttNonLinBlockParam param)
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
Operator * CreateOp(MlpAttNonLinBlockParam param, int dtype);

#if DMLC_USE_CXX11

class MlpAttNonLinBlockProp : public OperatorProperty
{
private:
	MlpAttNonLinBlockParam _param;
public:
	MlpAttNonLinBlockProp() {}
	explicit MlpAttNonLinBlockProp(MlpAttNonLinBlockParam param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		return {"AttHidden", "QueryHidden"};
	}
	std::vector < std::string > ListOutputs  () const override 
	{
		return {"AttHidden"};
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

		CHECK_EQ(in_shape->size(), 2U);

		// query the input shape and perform shape inference
		const TShape & src_hidden_shape = (*in_shape)[int(EnumOpInputs::SrcHidden)];

		CHECK_EQ(src_hidden_shape.ndim(), 3U) << "Source Hidden should be rank-3 tensor of dim"
			"[batch size, seq len, state size].";

		unsigned batch_size = src_hidden_shape[0];
		unsigned seq_len    = src_hidden_shape[1];
		unsigned state_size = src_hidden_shape[2];

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::QueryHidden),
			Shape2(batch_size, state_size));
		
		out_shape->clear();

		out_shape->push_back((*in_shape)[int(EnumOpInputs::QueryHidden)]); // AttHidden

		return true;
	}
	bool InferType (std::vector < int > *  in_type,
	                std::vector < int > * out_type,
			std::vector < int > * aux_type) const override 
	{
		CHECK_GE(in_type->size(), 1U);

		int src_hidden_type = (*in_type)[0];

		CHECK_NE(src_hidden_type, -1) << "First input must have specified type.";

		for (std::size_t i = 1; i < in_type->size(); ++i)
		{
			if ((*in_type)[i] == -1) 
			{
				(*in_type)[i] = src_hidden_type;
			}
			else
			{
				CHECK_EQ((*in_type)[i], src_hidden_type) << "This layer requires uniform type. " << 
					"Expected " << src_hidden_type << " v.s. given " << 
					(*in_type)[i] << " at " << ListArguments()[i];
			}
		}

		out_type->clear();

		out_type->push_back(src_hidden_type); // AttHidden
		
		return true;
	}

	OperatorProperty * Copy() const override 
	{
		return new MlpAttNonLinBlockProp(_param);
	}

	std::string TypeString() const override
	{
		return "MlpAttNonLinBlock";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		return { out_grad[int(EnumOpOutputs::AttHidden)] };
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
}; // class MlpAttNonLinBlockProp

#endif // DMLC_USE_CXX11

	} // namespace op
} // namespace mxnet