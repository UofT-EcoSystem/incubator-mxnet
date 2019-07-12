#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
        namespace op {
                namespace {

enum class EnumOpInputs    { data };
enum class EnumOpOutputs   { output };
enum class EnumOpWorkspace { TempSpace };

                }  // namespace anonymous

struct EcoReduceSumParam : public dmlc::Parameter < EcoReduceSumParam >
{
        int axis; bool keepdims;

        DMLC_DECLARE_PARAMETER(EcoReduceSumParam) {}
};

template < typename xpu, typename DType >
class EcoReduceSumOp : public Operator
{
private:
        EcoReduceSumParam _param;
public:
        explicit EcoReduceSumParam(EcoReduceSumParam param)
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
Operator * CreateOp(EcoReduceSumParam param, int dtype);

#if DMLC_USE_CXX11

class EcoReduceSumProp : public OperatorProperty
{
private:
        EcoReduceSumParam _param;
public:
        EcoReduceSumProp() {}
        explicit
        EcoReduceSumProp(EcoReduceSumParam param): _param(param) {}
};

#endif  // DMLC_USE_CXX11

        }  // namespace op
}  // namespace mxnet
