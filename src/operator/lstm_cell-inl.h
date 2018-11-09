#pragma once

// #include <dmlc/logging.h>
#include <dmlc/parameter.h>

// #include <mxnet/storage.h>
#include <mxnet/operator.h>

namespace mxnet {
	namespace op {
		namespace {

enum class EnumOpInputs  {CellInput, HiddenState, CellState};
enum class EnumOpOutputs            {HiddenState, CellState};
// NO Need for Workspace for Reserve Space

struct LSTMCellParam : public dmlc::Parameter < LSTMCellParam >
{
	std::uint32_t state_size;
	// The `state_size` does not require any declaration, and the reason is becasue 
	// it can be inferred from the shape of the input tensor.
};

template < typename xpu, typename DType >
class LSTMCellOp : public Operator
{
private:
	LSTMCellParam _param;
public:
	explicit LSTMCellOp(LSTMCellParam param)
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
Operator * CreateOp(LSTMCellParam param, int dtype);

#if DMLC_USE_CXX11

class LSTMCellProp : public OperatorProperty
{
public:
	std::vector<std::string> ListArguments() const override
	{
		return {"data", "parameters", "state"};
	}
	

  std::vector<std::string> ListOutputs() const override {
    std::vector<std::string> outputs = {"output"};
    if (!param_.state_outputs)
      return outputs;
    else
      outputs.push_back("state");
    if (param_.mode == rnn_enum::kLstm)
      outputs.push_back("state_cell");
    return outputs;
  }

  int NumOutputs() const override {
    int mode_num = (param_.mode == rnn_enum::kLstm) ? 2 : 1;
    int num_outputs = param_.state_outputs ? (mode_num + 1) : 1;
    return num_outputs;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (param_.mode == rnn_enum::kLstm) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, parameters, state, cell_state]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, parameters, state]";
    }
    const TShape &dshape = (*in_shape)[rnn_enum::kData];
    if (dshape.ndim() ==  0) return false;
    CHECK_EQ(dshape.ndim(), 3U) \
        << "Input data should be rank-3 tensor of dim [sequence length, batch size, input size]";
    // data: [sequence len, batch, input dimension]
    int batch_size = dshape[1];
    int input_size = dshape[2];
    int numDirections = param_.bidirectional ? 2 : 1;
    int total_layers = numDirections * param_.num_layers;  // double for bidirectional
    SHAPE_ASSIGN_CHECK(*in_shape,
                       rnn_enum::kState,
                       Shape3(total_layers, batch_size, param_.state_size));
    if (param_.mode == rnn_enum::kLstm)
      SHAPE_ASSIGN_CHECK(*in_shape,
                        rnn_enum::kStateCell,
                        Shape3(total_layers, batch_size, param_.state_size));

    // calculate parameter vector length
    int param_size = rnn_param_size(param_.num_layers,
                                    input_size,
                                    param_.state_size,
                                    param_.bidirectional,
                                    param_.mode);
    SHAPE_ASSIGN_CHECK(*in_shape, rnn_enum::kParams, Shape1(param_size));

    out_shape->clear();
    // output: [sequence len, batch, output size]
    TShape oshape = dshape;
    oshape[2] = numDirections * param_.state_size;
    out_shape->push_back(oshape);
    if (!param_.state_outputs) {
      return true;
    } else {
      // outStateShape: [layer_num, batch, state size]
      TShape outStateShape = dshape;
      outStateShape[0] = total_layers;
      outStateShape[1] = batch_size;
      outStateShape[2] = param_.state_size;
      out_shape->push_back(outStateShape);
      // Deal with lstm cell state
      if (param_.mode == rnn_enum::kLstm)
        out_shape->push_back(outStateShape);
      return true;
    }
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    if (!param_.state_outputs) {
      return true;
    } else {
      out_type->push_back(dtype);
      // Deal with lstm cell state
      if (param_.mode == rnn_enum::kLstm)
        out_type->push_back(dtype);
      return true;
    }
  }

  OperatorProperty* Copy() const override {
    auto ptr = new RNNProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "RNN";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    std::vector<int> dep = {in_data[rnn_enum::kData], in_data[rnn_enum::kParams],
        in_data[rnn_enum::kState], out_data[rnn_enum::kOut], out_grad[rnn_enum::kOut]};

    if (param_.state_outputs) {
      dep.push_back(out_data[rnn_enum::kStateOut]);
      dep.push_back(out_grad[rnn_enum::kStateOut]);
    }

    if (param_.mode == rnn_enum::kLstm) {
      dep.push_back(in_data[rnn_enum::kStateCell]);
      if (param_.state_outputs) {
        dep.push_back(out_data[rnn_enum::kStateCellOut]);
        dep.push_back(out_grad[rnn_enum::kStateCellOut]);
      }
    }
    return dep;
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  RNNParam param_;
};  // class RNNProp

#endif // DMLC_USE_CXX11

		} // namespace 
	} // namespace op
} // namespace mxnet