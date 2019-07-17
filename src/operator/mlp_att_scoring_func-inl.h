#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs    { QryHidden, SrcHidden, 
                             H2SWeight, 
			     Gamma, Beta };
enum class EnumOpOutputs   { AttScores, AttHiddenMean, AttHiddenSTD };
enum class EnumOpWorkspace { TempSpace };

		} // namespace 

struct MLPAttScoringFuncParam : public dmlc::Parameter < MLPAttScoringFuncParam >
{
	unsigned batch_size, seq_length, state_size;
	bool layer_norm; float eps

	DMLC_DECLARE_PARAMETER(MLPAttScoringFuncParam)
	{
		DMLC_DECLARE_FIELD(layer_norm).set_default(false)
			.describe("Whether to perform Layer Normalization after "
			          "broadcast adding Query to Source hidden state.");
		DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
			.describe("An `epsilon` parameter to prevent division by zero");
	}
};

template < typename xpu, typename DType >
class MLPAttScoringFuncOp : public Operator
{
private:
	MLPAttScoringFuncParam _param;
public:
	explicit MLPAttScoringFuncOp(MLPAttScoringFuncParam param)
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
Operator * CreateOp(MLPAttScoringFuncParam param, int dtype);

#if DMLC_USE_CXX11

class MLPAttScoringFuncProp : public OperatorProperty
{
private:
	MLPAttScoringFuncParam _param;
public:
	MLPAttScoringFuncProp() {}
	explicit
	MLPAttScoringFuncProp(MLPAttScoringFuncParam param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		if (_param.layer_norm)
			return { "qry_hidden", "src_hidden",
			         "h2s_weight",
				 "gamma", "beta" };
		else
			return { "qry_hidden", "src_hidden", 
		        	 "h2s_weight" };
	}
	std::vector < std::string > ListOutputs  () const override 
	{
		if (_param.layer_norm)
			return { "att_scores", "att_hidden_mean",
			               	       "att_hidden_std" };
		else
			return { "att_scores" };

	}
	int NumVisibleOutputs() const override 
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

		if (_param.layer_norm)
			CHECK_EQ(in_shape->size(), 3U); // qry_hidden, src_hidden, h2s_weight
		else
			CHECK_EQ(in_shape->size(), 5U); // gamma, beta

		// query the input shape and perform shape inference
		const TShape & src_hidden_shape = (*in_shape)[int(EnumOpInputs::SrcHidden)];

		CHECK_EQ(src_hidden_shape.ndim(), 3U) << "Source Hidden should be rank-3 tensor of dim "
			"[batch size, seq length, state size].";

		unsigned batch_size = src_hidden_shape[0];
		unsigned state_size = src_hidden_shape[2];

		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::QryHidden),
			Shape2(batch_size, state_size));
		SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2SWeight), Shape2(1, state_size));
		if (_param.layer_norm)
		{
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Gamma), Shape1(state_size));
			SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Beta),  Shape1(state_size));
		}
		out_shape->clear();

		TShape att_scores_shape = src_hidden_shape;

		att_scores_shape[2] = 1; // [batch_size x seq_length x 1]
		
		out_shape->push_back(att_scores_shape); // att_scores
		if (_param.layer_norm)
		{
			out_shape->push_back(Shape1(state_size)); // att_hidden_mean
			out_shape->push_back(Shape1(state_size)); // att_hidden_std
		}

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

		out_type->push_back(src_hidden_type); // att_scores
		if (_param.layer_norm)
		{
			out_shape->push_back(src_hidden_type); // att_hidden_mean
			out_shape->push_back(src_hidden_type); // att_hidden_std
		}
		
		return true;
	}

	OperatorProperty * Copy() const override 
	{
		return new MLPAttScoringFuncProp(_param);
	}

	std::string TypeString() const override
	{
		return "MLPAttScoringFunc";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		if (_param.layer_norm)
			return { out_grad[int(EnumOpOutputs::AttScores)],
			          in_data[int(EnumOpInputs ::QryHidden)],
				  in_data[int(EnumOpInputs ::SrcHidden)],
				  in_data[int(EnumOpInputs ::H2SWeight)],
				  in_data[int(EnumOpInputs ::Gamma)],
				  in_data[int(EnumOpInputs ::Beta)]
				//  out_data[int(EnumOpOutputs::AttHiddenMean)],
				//  out_data[int(EnumOpOutputs::AttHiddenSTD)]
				};
		else
			return { out_grad[int(EnumOpOutputs::AttScores)],
				  in_data[int(EnumOpInputs ::QryHidden)],
				  in_data[int(EnumOpInputs ::SrcHidden)],
				  in_data[int(EnumOpInputs ::H2SWeight)] };
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
}; // class MLPAttScoringFuncProp

#endif // DMLC_USE_CXX11

	} // namespace op
} // namespace mxnet
