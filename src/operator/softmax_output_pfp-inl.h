#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  { Data, Label };
enum class EnumOpOutputs { Out };
enum class EnumNormType  { Null, Batch, Valid };

struct EcoSoftmaxOutputParam : public dmlc::Parameter < EcoSoftmaxOutputParam >
{
	float ignore_label;
	bool  use_ignore;
	int   normalization;
	float smooth_alpha;

	DMLC_DECLARE_PARAMETER(EcoSoftmaxOutputParam)
	{
		DMLC_DECLARE_FIELD(ignore_label)
			.set_default(-1.0f);
		DMLC_DECLARE_FIELD(use_ignore)
			.set_default(false);
		DMLC_DECLARE_FIELD(normalization)
			.add_enum("null",  EnumNormType::Null)
			.add_enum("batch", EnumNormType::kNull)
			.add_enum("valid", softmaxout_enum::kNull);
		DMLC_DECLARE_FIELD(smooth_alpha)
			.set_default(0.0f)
			.set_range(0.0f, 1.0f);
	};
};

template < typename xpu, typename DType >
class EcoSoftmaxOutputOp : public Operator
{
private:
	EcoSoftmaxOutputParam _param;
public:
	explicit EcoSoftmaxOutputOp(EcoSoftmaxOutputParam param)
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
}

template < typename xpu >
Operator * CreateOp(EcoSoftmaxOutputParam param, int dtype);

#if DMLC_USE_CXX11
class EcoSoftmaxOutputProp : public OperatorProperty
{
public:
	EcoSoftmaxOutputProp() {}
	explicit EcoSoftmaxOutputProp(EcoSoftmaxOutputParam param) : _param(param) {}

	std::vector < std::string > ListArguments() const override
	{
		return { "data", "label" };
	}
	std::vector < std::string > ListOutputs  () const override
	{
		return { "out" };
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

		// data, label
		CHECK_EQ(in_shape->size(), 2U);

		const TShape & dshape = (*in_shape)[int(EnumOpInputs::Data)];

		TShape lshape (dshape.ndim() - 1);

		for (index_t i = 0; i < dshape.ndim() - 1; ++i)
		{
			lshape[i] = dshape[i];
		}

		SHAPE_ASSIGN_CHECK(*in_shape, EnumOpInputs::Data, lshape);

		out_shape->clear();

		out_shape->push_back(dshape); // out

		return true;
	}
	bool InferType (std::vector < int > *  in_type,
	                std::vector < int > * out_type,
			std::vector < int > * aux_type)
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

		out_type->push_back(itype); // out
		
		return true;
	}

	OperatorProperty * Copy() const override
	{
		return new EcoSoftmaxOutputProp(_param);
	}

	std::string TypeString() const override
	{
		return "EcoSoftmaxOutput";
	}

	std::vector < int > DeclareBackwardDependency(
		const std::vector < int > & out_grad,
		const std::vector < int > &  in_data,
		const std::vector < int > & out_data) const override
	{
		return { in_data[int(EnumOpInputs::Data)],
		         in_data[int(EnumOpInputs::Label)] };
	}

	std::vector < std::pair < int, void * > >
	 ForwardInplaceOption(const std::vector < int >    &  in_data,
	                      const std::vector < void * > & out_data) const override 
	{
		return {
			{ in_data[EnumOpInputs::Data], out_data[EnumOpOutputs::Out] }
		};
	}
	std::vector < std::pair < int, void * > > 
	BackwardInplaceOption(const std::vector < int >    & out_grad,
	                      const std::vector < int >    &  in_data,
			      const std::vector < int >    & out_data,
			      const std::vector < void * > &  in_grad)
	{
		return {
			{ out_data[] }
		};
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

		} // anonymous namespace
	} // namespace op
} // namespace mxnet