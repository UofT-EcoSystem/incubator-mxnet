#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
        namespace op {
                namespace {

enum class EnumOpInputs    { Data };
enum class EnumOpOutputs   { Output };
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

        std::vector < std::string > ListArguments() const override
        {
                return { "data" };
        }
        std::vector < std::string > ListOutputs  () const override
        {
                return { "output" };
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

                CHECK_EQ(in_shape->size(), 1);
                const TShape & ishape = (*in_shape)[int(EnumOpInputs::Data)];
                const int reduce_axis = (_param.axis + ishape.ndim()) % 
                                                       ishape.ndim();
                out_shape->clear();

                TShape oshape = _param.keepdims ?
                        TShape(ishape.ndim()) : 
                        TShape(ishape.ndim() - 1);
                for (uint32_t idim_idx = 0, 
                              odim_idx = 0;
                              idim_idx < ishape.ndim();
                            ++idim_idx)
                {
                        if (idim_idx == reduce_axis)
                        {
                                if (keep_dim)
                                {
                                        oshape[odim_idx++] = 1;
                                }
                                continue;
                        }
                        oshape[odim_idx++] = ishape[idim_idx];
                }
                out_shape->push_back(oshape);

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

		out_type->push_back(itype); // output
		
		return true;
        }
};

#endif  // DMLC_USE_CXX11

        }  // namespace op
}  // namespace mxnet
