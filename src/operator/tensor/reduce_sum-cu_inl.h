#pragma once

#include "reduce_sum-inl.h"

namespace mxnet {
        namespace op {

// #if defined(__CUDACC__)

/**
 * Forward Pass of Reduce Sum
 */

template < typename DType >
class CUEcoReduceSumOp : public Operator
{
private:
        EcoReduceSumParam _param;

        bool _initialized = false;
public:
        explicit CUEcoReduceSumOp(EcoReduceSumParam param)
        {
                _param = param;
        }
        ~CUEcoReduceSumOp() {}
public:
        virtual void  Forward(const OpContext & ctx,
                              const std::vector < TBlob > &  in_data,
                              const std::vector < OpReqType > &  req,
                              const std::vector < TBlob > & out_data,
                              const std::vector < TBlob > & aux_args)
        {
                using namespace mshadow;

                std::size_t  in_expected = 1,
                            out_expected = 1;

                CHECK_EQ( in_data.size(),  in_expected); // data
                CHECK_EQ(out_data.size(), out_expected); // output

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();
        }
};


// #endif  // defined(__CUDACC__)

        }  // namespace op 
}  // namespace mxnet
