#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
        namespace op {
                namespace {

enum class EnumOpInputs  { Input, StateH, I2HWeight, I2HBias, 
                                          H2HWeight, H2HBias };
enum class EnumOpOutputs { StateHOut, FeatureMapResetGate,
                                      FeatureMapH2H, 
                                      FeatureMapUpdateGate };
enum class EnumOpWorkspace { TempSpace };

                }  // namespace anonymous

struct InvisGRUCellParam : public dmlc::Parameter < InvisGRUCellParam >
{
        unsigned batch_size, input_size, state_size;

        DMLC_DECLARE_PARAMETER(InvisGRUCellParam) {}
};

template < typename xpu, typename DType >
class InvisGRUCellOp : public Operator
{
public:
        InvisGRUCellOp(InvisGRUCellParam param)
        {
                // param
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
                const std::vector < TBlob > & aux_grad)
        {
                // empty
        }
};  // class InvisGRUCellOp

template < typename xpu >
Operator * CreateOp(InvisGRUCellParam param, int dtype);

#if DMLC_USE_CXX11

class InvisGRUCellProp : public OperatorProperty
{
private:
        InvisGRUCellParam _param;
public:
        InvisGRUCellProp() {}
        explicit InvisGRUCellProp(InvisGRUCellParam param) : _param(param) {}

        std::vector < std::string > ListArguments() const override
        {
                return { "input", "state_h", "i2h_weight", "i2h_bias",
                                             "h2h_weight", "h2h_bias"};
        }
        std::vector < std::string > ListOutputs()   const override
        {
                return { "state_h_out", "feature_map_reset_gate",
                                        "feature_map_h2h",
                                        "feature_map_update_gate" };
        }

        int NumOutputs() const override 
        {
                return 3;
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

                // input, state_h, i2h_weight, i2h_bias
                //                 h2h_weight, h2h_bias
                CHECK_EQ(in_shape->size(), 7);

                const TShape & ishape = (*in_shape)[int(EnumOpInputs::Input)];
                const TShape & hshape = (*in_shape)[int(EnumOpInputs::StateH)];

                CHECK_EQ(ishape.ndim(), 2U) <<   "Input data should be rank-2 tensor of dim "
			"[batch size, input size]";
		CHECK_EQ(hshape.ndim(), 2U) << "Hidden state should be rank-2 tensor of dim "
			"[batch size, state size]";

                unsigned batch_size = ishape[0];
                unsigned input_size = ishape[1];
                unsigned state_size = hshape[1];

                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::StateH),
                        Shape2(batch_size, state_size));
                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::I2HWeight),
                        Shape2(3 * state_size, input_size));
                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::I2HBias),
                        Shape1(3 * state_size));
                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2HWeight),
                        Shape2(3 * state_size, state_size));
                SHAPE_ASSIGN_CHECK(*in_shape, int(EnumOpInputs::H2HBias),
                        Shape1(3 * state_size));
                
                out_shape->clear();

                out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]);  // state_h_out
                out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]);  // feature_map_reset_gate
                out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]);  // feature_map_h2h
                out_shape->push_back((*in_shape)[int(EnumOpInputs::StateH)]);  // feature_map_update_gate

                return true;
        }
        bool InferType(std::vector < int > *  in_type,
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

		out_type->push_back(itype);  // state_h_out
		out_type->push_back(itype);  // feature_map_reset_gate
                out_type->push_back(itype);  // feature_map_h2h
		out_type->push_back(itype);  // feature_map_update_gate
		
		return true;
        }

        OperatorProperty * Copy() const override
        {
                return new InvisGRUCellProp(_param);
        }

        std::string TypeString() const override
        {
                return "InvisGRUCell";
        }

        std::vector < int > DeclareBackwardDependency(
                const std::vector < int > & out_grad,
                const std::vector < int > &  in_data,
                const std::vector < int > & out_data) const override
        {
                return {  in_data[int(EnumOpInputs ::Input)],
                          in_data[int(EnumOpInputs ::StateH)],
                          in_data[int(EnumOpInputs ::I2HWeight)],
                          in_data[int(EnumOpInputs ::H2HWeight)],
                         out_data[int(EnumOpOutputs::StateHOut)],
                         out_data[int(EnumOpOutputs::FeatureMapResetGate)],
                         out_data[int(EnumOpOutputs::FeatureMapH2H)],
                         out_data[int(EnumOpOutputs::FeatureMapUpdateGate)],
                         out_grad[int(EnumOpOutputs::StateHOut)]};
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

        Operator * CreateOperator(Context ctx) const override
        {
                LOG(FATAL) << "Not Implemented";

                return nullptr;
        }
        Operator * CreateOperatorEx(Context ctx,
                std::vector < TShape > * in_shape,
                std::vector < int >    * in_type) const override;
};  // class InvisGRUCellProp

#endif  // DMLC_USE_CXX11
        }  // namespace op
}  // namespace mxnet
