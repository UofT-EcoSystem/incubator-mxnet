#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs    { SrcHidden, QryHidden, H2SWeight };
enum class EnumOpOutputs   { AttScores };
enum class EnumOpWorkspace { AttHidden, AttHiddenGrad };

		} // namespace 

struct MlpAttNonLinBlockParam : public dmlc::Parameter < MlpAttNonLinBlockParam >
{
	unsigned batch_size, seq_length, state_size;

	bool layer_norm;

	DMLC_DECLARE_PARAMETER(MlpAttNonLinBlockParam)
	{
		DMLC_DECLARE_FIELD(layer_norm).set_default(false)
			.describe("Whether to perform layer normalization after "
			          "broadcast add query to source hidden state.");
	}
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
		return {"SrcHidden", "QryHidden"};
	}
	std::vector < std::string > ListOutputs  () const override 
	{
		return {"AttScores"};
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

		CHECK_EQ(in_shape->size(), 3U); // SrcHidden, QryHidden, H2SWeight

		// query the input shape and perform shape inference
		const TShape & src_hidden_shape = (*in_shape)[int(EnumOpInputs::SrcHidden)];

		CHECK_EQ(src_hidden_shape.ndim(), 3U) << "Source Hidden should be rank-3 tensor of dim"
			"[batch size, seq length, state size].";

		unsigned batch_size = src_hidden_shape[0];
		// unsigned seq_length = src_hidden_shape[1];
		unsigned state_size = src_hidden_shape[2];

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::QryHidden),
			Shape2(batch_size, state_size));
		// According to the documentation provided by MXNet,
		// `FullyConnected` layers are implemented in $Y = XW^T$, where 
		// $X$ is of dimension $B\times H$ and $W$ is of dimension $H'\times H$.
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2SWeight),
			Shape2(         1, state_size));
		
		out_shape->clear();

		TShape att_scores_shape = src_hidden_shape;

		att_scores_shape[2] = 1; // [batch_size x seq_length x 1]

		out_shape->push_back(att_scores_shape); // AttScores

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

		out_type->push_back(src_hidden_type); // AttScores
		
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
		return {  in_data[int(EnumOpInputs ::SrcHidden)],
			  in_data[int(EnumOpInputs ::QryHidden)],
			  in_data[int(EnumOpInputs ::H2SWeight)],
			 out_grad[int(EnumOpOutputs::AttScores)] };
	}

	std::vector < ResourceRequest >  ForwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		// return {};
		return { ResourceRequest::kTempSpace }; // EnumOpWorkspace::AttHidden
	}
	std::vector < ResourceRequest > BackwardResource(
		const std::vector < TShape > & in_shape) const override
	{
		// return {};
		return { ResourceRequest::kTempSpace,   // EnumOpWorkspace::AttHidden
		         ResourceRequest::kTempSpace }; // EnumOpWorkspace::AttHiddenGrad
	}

	Operator * CreateOperator  (Context ctx) const override
	{
		LOG(FATAL) << "Not Implemented";
		
		return nullptr;
  	}
	Operator * CreateOperatorEx(Context ctx, 
				    std::vector < TShape > * in_shape,
        			    std::vector < int >    * in_type) const override;
}; // class MlpAttNonLinBlockProp

#endif // DMLC_USE_CXX11

	} // namespace op
} // namespace mxnet