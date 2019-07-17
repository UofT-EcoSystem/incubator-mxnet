#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
        namespace op {
                namespace {

enum class EnumOpInputs  { Data, Gamma, Beta };
enum class EnumOpOutputs { Output, Mean, STD };
enum class EnumOpWorkspace { TempSpacew };
                }  // namespace op

struct LayerNormParam : public dmlc::Parameter < LayerNormParam >
{
        std::size_t total_size, state_size; float eps;

        DMLC_DECLARE_PARAMETER(LayerNormParam)
        {
                DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
                        .describe("An `epsilon` parameter to prevent "
                                  "division by zero");
        }
};

template < typename xpu, typename DType >
class LayerNormOp : public Operator
{
private:
        LayerNormParam _param;
public:
        explicit LayerNormOp(LayerNormParam param)
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
Operator * CreateOp(LayerNormParam param, int dtype);

#if DMLC_USE_CXX11

class LayerNormProp : public OperatorProperty
{
private:
        LayerNormParam _param;
public:
        LayerNormProp() {}
        explicit 
        LayerNormProp(LayerNormParam param) : _param(param) {}

        std::vector < std::string > ListArguments() const override
        {
                return { "data", "gamma", "beta" };
        }
        std::vector < std::string > ListOutputs  () const override
        {
                return { "output", "mean", "std" };
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

                CHECK_EQ(in_shape->size(), 3U); // data, gamma, beta

                const TShape & dshape = (*in_shape)[int(EnumOpInputs::Data)];

                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Gamma), Shape1(dshape[dshape.ndim() - 1]));
                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::Beta),  Shape1(dshape[dshape.ndim() - 1]));

                out_shape->clear();

                TShape mean_std_shape = (*in_shape)[int(EnumOpInputs::Data)];

                mean_std_shape[(*in_shape)[int(EnumOpInputs::Data)].ndim() - 1] = 1;

                out_shape->push_back((*in_shape)[int(EnumOpInputs::Data)]);  // output
                out_shape->push_back(mean_std_shape); // mean
                out_shape->push_back(mean_std_shape); // std

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

		out_type->push_back(itype); // output
		out_type->push_back(itype); // mean
		out_type->push_back(itype); // std
		
		return true;
        }

        OperatorProperty * Copy() const override
	{
		return new LayerNormProp(_param);
	}

	std::string TypeString() const override
	{
		return "LayerNorm";
	}

        std::vector < int > DeclareBackwardDependency(
                const std::vector < int > & out_grad,
                const std::vector < int > &  in_data,
                const std::vector < int > & out_data) const override
        {
                return { out_grad[int(EnumOpOutputs::Output)],
                          in_data[int(EnumOpInputs ::Data)],
                          in_data[int(EnumOpInputs ::Gamma)],
                         out_data[int(EnumOpOutputs::Mean)],
                         out_data[int(EnumOpOutputs::STD)] };
        }

        // std::vector < ResourceRequest >  ForwardResource(
        //         const std::vector < TShape > & in_shape) const override
        // {
        //         return { ResourceRequest::kTempSpace };
        // }
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

#endif  // DMLC_USE_CXX11

        }  // namespace op 
}  // namespace mxnet
