#pragma once

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include "operator_common.h"

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  {AttHidden, QueryHidden};
enum class EnumOpOutputs {AttHidden};
// NO Need for Temporary Workspace

		} // namespace 

struct MlpAttNonlinBlockParam : public dmlc::Parameter < MlpAttNonlinBlockParam >
{
	// All parameters do not require explicit declaration.
	// The reason is because they can be effectively inferred from the shape of the input data.
	unsigned batch_size, state_size;

	DMLC_DECLARE_PARAMETER(MlpAttNonlinBlockParam) {}
}

template < typename xpu, typename DType >
class MlpAttNonlinBlockOp : public Operator
{
private:
	MlpAttNonlinBlockParam _param;
public:
	explicit MlpAttNonlinBlockOp(MlpAttNonlinBlockParam param)
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
Operator * CreateOp(LSTMNonLinBlockParam param, int dtype);

	} // namespace op
} // namespace mxnet