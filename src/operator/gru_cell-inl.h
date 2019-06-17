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
private:
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
}

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
        std::map < std::string > GetParams() const override
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
                out_shape->push_back((*in_shape)[int])

                return true;
        }
};

#endif  // DMLC_USE_CXX11
        }  // namespace op
}  // namespace mxnet
