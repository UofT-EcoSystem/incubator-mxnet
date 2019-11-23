/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu
*/

#ifndef MXNET_OPERATOR_DROPOUT_INL_H_
#define MXNET_OPERATOR_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "./operator_common.h"
#include "./mshadow_op.h"

#if defined(USE_MKL) && defined(_OPENMP)
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // USE_MKL && _OPENMP

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
enum DropoutOpMode {kTraining, kAlways};
}  // namespace dropout

namespace mxnet {
namespace op {

#if defined(USE_MKL) && defined(_OPENMP)
static void bernoulli_generate(int n, double p, int* r) {
  int seed = 17 + rand() % 4096;  // NOLINT(runtime/threadsafe_fn)
  int nthr = omp_get_max_threads();
# pragma omp parallel num_threads(nthr)
  {
    const int ithr = omp_get_thread_num();
    const int avg_amount = (n + nthr - 1) / nthr;
    const int my_offset = ithr * avg_amount;
    const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
    if (my_amount > 0) {
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MCG31, seed);
      vslSkipAheadStream(stream, my_offset);
      viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount,
        r + my_offset, p);
      vslDeleteStream(&stream);
    }
  }
}
#endif  // USE_MKL && _OPENMP

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", dropout::kTraining)
    .add_enum("always", dropout::kAlways)
    .set_default(dropout::kTraining)
    .describe("Whether to only turn on dropout during training or to also turn on for inference.");
  }
};  // struct DropoutParam

#if defined(__CUDACC__)

__global__ void binarize_mask(
    const float    * const __restrict__ mask,
          unsigned * const __restrict__ binarized_mask,
    const unsigned mask_size) {
  const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (g_threadIdx >= mask_size) {
    return;
  }
  // unsigned bit_mask = 1 << ((g_threadIdx / 32 + 1) * 32 - g_threadIdx - 1);
  // if (mask[g_threadIdx]) {
  //   binarized_mask[g_threadIdx / 32] |=   bit_mask;
  // } else {
  //   binarized_mask[g_threadIdx / 32] &= (~bit_mask);
  // }

  // __shared__ bool smem_binarized_mask [32];

  // smem_binarized_mask[g_threadIdx % 32] = mask[g_threadIdx] != 0;
  // if (threadIdx.x == 0) {
  //   unsigned accumulated_mask = 0;

  //   for (unsigned i = 0; i < 32; ++i) {
  //     accumulated_mask += (1 << (31 - i)) * smem_binarized_mask[i];
  //   }

  //   // printf("%u\n", accumulated_mask);
  //   binarized_mask[g_threadIdx / 32] = accumulated_mask;
  // }
  unsigned accumulated_mask = static_cast<unsigned>(
      __ballot_sync(0xffffffff, 
        mask[g_threadIdx] != 0));

  if (threadIdx.x == 0) {
    binarized_mask[g_threadIdx / 32] = accumulated_mask;
  }
}

__global__ void unbinarize_mask(
          float    * const __restrict__ mask,
    const unsigned * const __restrict__ binarized_mask,
    const float p,
    const unsigned mask_size) {
  const unsigned g_threadIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (g_threadIdx >= mask_size) {
    return;
  }
  // unsigned bit_mask = 1 << ((g_threadIdx / 32 + 1) * 32 - g_threadIdx - 1);
  unsigned bit_mask = 1 << (g_threadIdx - (g_threadIdx / 32) * 32);
  unsigned bit = binarized_mask[g_threadIdx / 32] & bit_mask;
  if (bit) {
    mask[g_threadIdx] = 1.0 / p;
  } else {
    mask[g_threadIdx] = 0;
  }
}

#endif  // defined(__CUDACC__)

static bool logged_binarized_dropout = false;

template<typename xpu, typename DType>
class DropoutOp : public Operator {
 public:
  explicit DropoutOp(DropoutParam param) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = param.mode;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 2U);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1, DType> data = in_data[dropout::kData].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> out = out_data[dropout::kOut].FlatTo1D<xpu, DType>(s);
    if (ctx.is_train || mode_ == dropout::kAlways) {

      if (!logged_binarized_dropout) {
        LOG(INFO) << "MXNet uses binarized dropout? : "
                  << std::boolalpha
                  << bool(dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0))
                  << std::noboolalpha;
        logged_binarized_dropout = true;
      }

      if (dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0)) {

      CHECK_EQ(ctx.requested.size(), 2);
      CHECK_GE(out_data[dropout::kMask].shape_.Size() * 32, 
                in_data[dropout::kData].shape_.Size());
      Tensor<xpu, 1, DType> mask = ctx.requested[1]
          .get_space_typed<xpu, 1, DType>(Shape1(data.shape_.Size()), s);
      Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
      mask = tcast<DType>(F<mshadow_op::threshold>(
             prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_));
      Tensor<xpu, 1, DType> binarized_mask = out_data[dropout::kMask].get<xpu, 1, DType>(s);
#if defined(__CUDACC__)
      binarize_mask <<< (data.shape_.Size() - 1) / 32 + 1, 32, 0, 
                        Stream<gpu>::GetStream(s) >>> 
          (mask.dptr_, reinterpret_cast<unsigned *>(binarized_mask.dptr_), data.shape_.Size());
#endif  // defined(__CUDACC__)
      Assign(out, req[dropout::kOut], data * mask);

      } else {

      Tensor<xpu, 1, DType> mask = out_data[dropout::kMask].FlatTo1D<xpu, DType>(s);
#if !defined(__CUDACC__) && defined(USE_MKL) && defined(_OPENMP)
      DType* outptr = out.dptr_;
      DType* dataptr = data.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);
      int count = mask.shape_.Size();
      bernoulli_generate(count, this->pkeep_, maskptr);
  #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        outptr[i] = dataptr[i] * maskptr[i] * (1.0f / pkeep_);
      }
#else
      Random<xpu> *prnd = ctx.requested[dropout::kRandom].get_random<xpu, real_t>(s);
      mask = tcast<DType>(F<mshadow_op::threshold>(
             prnd->uniform(mask.shape_), pkeep_) * (1.0f / pkeep_));
      Assign(out, req[dropout::kOut], data * mask);
#endif  // USE_MKL && _OPENMP

      }
    } else {
      Assign(out, req[dropout::kOut], F<mshadow_op::identity>(data));
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_grad.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1, DType> grad = out_grad[dropout::kOut].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> gdata = in_grad[dropout::kData].FlatTo1D<xpu, DType>(s);
    if (ctx.is_train || mode_ == dropout::kAlways) {

      if (dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0)) {

      CHECK_EQ(ctx.requested.size(), 1);
      Tensor<xpu, 1, DType> mask = ctx.requested[0]
          .get_space_typed<xpu, 1, DType>(Shape1(gdata.shape_.Size()), s);
      Tensor<xpu, 1, DType> binarized_mask = out_data[dropout::kMask].get<xpu, 1, DType>(s);
#if defined(__CUDACC__)
      unbinarize_mask <<< (gdata.shape_.Size() - 1) / 32 + 1, 32, 0, 
                          Stream<gpu>::GetStream(s) >>> 
          (mask.dptr_, reinterpret_cast<unsigned *>(binarized_mask.dptr_), 
           this->pkeep_, gdata.shape_.Size());
#endif  // defined(__CUDACC__)
      Assign(gdata, req[dropout::kData], grad * mask);

      } else {

      Tensor<xpu, 1, DType> mask = out_data[dropout::kMask].FlatTo1D<xpu, DType>(s);
#if !defined(__CUDACC__) && defined(USE_MKL) && defined(_OPENMP)
      DType* ingradptr = gdata.dptr_;
      DType* outgradptr = grad.dptr_;
      int* maskptr = reinterpret_cast<int*>(mask.dptr_);

      int count = mask.shape_.Size();

      #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        ingradptr[i] = outgradptr[i] * maskptr[i] * (1.0f / pkeep_);
      }
#else  // USE_MKL && _OPENMP
      Assign(gdata, req[dropout::kData], grad * mask);
#endif  // USE_MKL && _OPENMP
      }
    } else {
      Assign(gdata, req[dropout::kData], F<mshadow_op::identity>(grad));
    }
  }

 private:
  real_t pkeep_;
  int mode_;
};  // class DropoutOp


template<typename xpu>
Operator *CreateOp(DropoutParam param, int dtype);

#if DMLC_USE_CXX11
class DropoutProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 1U);
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    if (dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0)) {
      out_shape->push_back(Shape1((dshape.Size() - 1) / 32 + 1));
    } else {
      out_shape->push_back(dshape);
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = in_type->at(0);

    if (dtype == -1) {
      LOG(FATAL) << "input type to dropout is not specified.";
      return false;
    }

    size_t nout = this->ListOutputs().size();
    out_type->clear();
    for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new DropoutProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Dropout";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[dropout::kOut], out_data[dropout::kMask]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[dropout::kOut], in_grad[dropout::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[dropout::kData], out_data[dropout::kOut]}};
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    if (dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0)) {
      return {ResourceRequest::kRandom,
              ResourceRequest::kTempSpace};
    } else {
      return {ResourceRequest::kRandom};
    }
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    if (dmlc::GetEnv("USE_BINARIZED_DROPOUT", 0)) {
      return {ResourceRequest::kTempSpace};
    } else {
      return {};
    }
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mask"};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  DropoutParam param_;
};  // class DropoutProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_DROPOUT_INL_H_
