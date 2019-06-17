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
enum class EnumOpOutputs { StateHOut,  };
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
        std::map < std::string >
};

#endif  // DMLC_USE_CXX11
        }  // namespace op
}  // namespace mxnet
