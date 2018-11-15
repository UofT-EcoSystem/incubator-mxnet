#pragma once

#include <vector>
#include <cooperative_groups.h>

#include <mxnet/storage.h>

#include "mlp_att_nonlin_block-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

template < typename RealType >
static __forceinline__ __device__ void __cu_reduce_sum(
	volatile RealType * const __restrict__ svmem_reduced_sum, 
	         RealType                      local_value)
{
	namespace cg = cooperative_groups;

	RealType local_sum;

	cg::thread_block this_cta = cg::this_thread_block();

	svmem_reduced_sum[threadIdx.x] = local_value;
	cg::sync(this_cta); // up to this point, the content of shared memory has been initialized with local variable
	// =========================================================================================
	// do reduction in shared mem
	if ((blockDim.x >= 512) && (threadIdx.x < 256)) 
		svmem_reduced_sum[threadIdx.x] = local_sum = local_sum + svmem_reduced_sum[threadIdx.x + 256];
	cg::sync(this_cta);
    	if ((blockDim.x >= 256) && (threadIdx.x < 128))
		svmem_reduced_sum[threadIdx.x] = local_sum = local_sum + svmem_reduced_sum[threadIdx.x + 128];
	cg::sync(this_cta);
	if ((blockDim.x >= 128) && (threadIdx.x <  64))
		svmem_reduced_sum[threadIdx.x] = local_sum = local_sum + svmem_reduced_sum[threadIdx.x +  64];
	cg::sync(this_cta);
	// =========================================================================================
	// further reduction will be done within a wrap, therefore no cta-level synchronization is needed
	if (threadIdx.x < 32)
	{
		cg::coalesced_group active_threads = cg::coalesced_threads();

		if (blockDim.x >= 64) local_sum += svmem_reduced_sum[threadIdx.x + 32];

#pragma unroll
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			local_sum += active_threads.shfl_down(local_sum, offset);
		}
	}
	if (threadIdx.x == 0) svmem_reduced_sum[0] = local_sum;
	cg::sync(this_cta);
}

/**
 * Forward Pass of the MLP Attention Layer Nonlinear Block
 * This kernel shall be launched using the parameter <<< (B, T), H >>>
 * @param1 src_hidden [B x T x H]:  (Input) Source Hidden State
 * @param2 qry_hidden [B     x H]:  (Input)  Query Hidden State
 * @param3 att_hidden [B x T x H]: (Output) Attention Hidden State
 * @param4 layer_norm: (Parameter) Whether to perform Layer Normalization
 */
template < typename RealType >
static __global__ void _cuda_fused_mlp_att_nonlin_block__forward(
	const RealType * const __restrict__ src_hidden,
	const RealType * const __restrict__ qry_hidden,
	      RealType * const __restrict__ att_hidden, const bool layer_norm)
{
	/*
            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.broadcast_add(lhs=attention_hidden_lhs, rhs=query_hidden,
                                                    name="%squery_plus_input" % self.prefix)
	 */
	const unsigned g_threadIdx = blockIdx.y *  gridDim.x *  blockDim.x + 
	                                          blockIdx.x *  blockDim.x + 
						               threadIdx.x;

	RealType att_hidden_reg = src_hidden[g_threadIdx] + 
	                          qry_hidden[blockIdx.y * blockDim.x + threadIdx.x];
	
	/*
            if self._ln is not None:
                attention_hidden = self._ln.normalize(attention_hidden)
	 */

	extern __shared__ volatile RealType svmem_reduced_sum[];

	if (layer_norm)
	{
		__cu_reduce_sum(svmem_reduced_sum,  att_hidden_reg);
		RealType exp_X = svmem_reduced_sum[0] / blockDim.x; // EXP[X]
		__cu_reduce_sum(svmem_reduced_sum, (att_hidden_reg - exp_X) *
		                                   (att_hidden_reg - exp_X));
		RealType var_X = svmem_reduced_sum[0] / blockDim.x; // VAR[X]
	
		// perform layer normalization
		att_hidden_reg = att_hidden_reg - exp_X;
		att_hidden_reg = att_hidden_reg / sqrt(var_X);
	}

	/*
            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%shidden" % self.prefix)
	 */
	att_hidden[g_threadIdx] = tanh(att_hidden_reg);
}

template < typename DType >
class CUMlpAttNonLinBlockOp : public Operator
{
private:
	MlpAttNonLinBlockParam _param;

	bool _initialized = false;

	// ReserveSpace
	Storage::Handle _reserved_space;
public:
	explicit CUMlpAttNonLinBlockOp(MlpAttNonLinBlockParam param)
	{
		_param = param;
	}
	~CUMlpAttNonLinBlockOp() {}
private:
	void _Init(mshadow::Stream < gpu > * cuda_stream,
	           const std::vector < TBlob > &  in_data,
		   const std::vector < TBlob > & out_data)
	{
		using namespace mshadow;

		CHECK_EQ(_initialized, false);

		Tensor < gpu, 3, DType > src_hidden = in_data[int(EnumOpInputs::SrcHidden)].
			get < gpu, 3, DType > (cuda_stream);

		// infer the parameters from the source hidden state
		_param.batch_size = src_hidden.shape_[0];
		_param.seq_length = src_hidden.shape_[1];
		_param.state_size = src_hidden.shape_[2];

		_initialized = true;
	}
public:
	virtual void  Forward(const OpContext & ctx,
	                      const std::vector < TBlob > &  in_data,
			      const std::vector < OpReqType > &  req,
			      const std::vector < TBlob > & out_data,
			      const std::vector < TBlob > & aux_args)
	{
		using namespace mshadow;

		std::size_t in_expected = 3, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected); // SrcHidden, QryHidden, H2SWeight
		CHECK_EQ(out_data.size(), out_expected); // AttScores

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 3, DType > src_hidden =  in_data[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > qry_hidden =  in_data[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2s_weight =  in_data[int(EnumOpInputs ::H2SWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > att_scores = out_data[int(EnumOpOutputs::AttScores)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(src_hidden.CheckContiguous(), true);
		CHECK_EQ(qry_hidden.CheckContiguous(), true);
		CHECK_EQ(h2s_weight.CheckContiguous(), true);
		CHECK_EQ(att_scores.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}
		/*
		if (ctx.is_train)
		{
			// TODO: Allocate the reserved space for training.
		}
		 */
		// TODO: Forward kernel goes here.
	}

	virtual void Backward(const OpContext & ctx,
	                      const std::vector < TBlob > & out_grad,
			      const std::vector < TBlob > &  in_data,
			      const std::vector < TBlob > & out_data,
			      const std::vector < OpReqType > &  req,
			      const std::vector < TBlob > &  in_grad,
			      const std::vector < TBlob > & aux_args)
	{
		using namespace mshadow;

		std::size_t in_expected = 3, out_expected = 1;

		CHECK_EQ( in_data.size(),  in_expected);
		CHECK_EQ( in_grad.size(),  in_expected);
		CHECK_EQ(     req.size(),  in_expected);
		CHECK_EQ(out_data.size(), out_expected);
		CHECK_EQ(out_grad.size(), out_expected);

		// TODO: Double check that the gradient computations and make sure that they are or are NOT accumulative.
		CHECK_NE(req[int(EnumOpInputs::SrcHidden)], kAddTo) << "AddTo is not supported for " "source hidden state.";
		CHECK_NE(req[int(EnumOpInputs::QryHidden)], kAddTo) << "AddTo is not supported for "  "query hidden state.";
		CHECK_NE(req[int(EnumOpInputs::H2SWeight)], kAddTo) << "AddTo is not supported for " "hidden-to-score weight.";
		
		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 3, DType > src_hidden_grad =  in_grad[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > qry_hidden_grad =  in_grad[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2s_weight_grad =  in_grad[int(EnumOpInputs ::H2SWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > att_scores_grad = out_grad[int(EnumOpOutputs::AttScores)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(src_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(qry_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(h2s_weight_grad.CheckContiguous(), true);
		CHECK_EQ(att_scores_grad.CheckContiguous(), true);

		// TODO: Backward kernel goes here.
	}
}; // class CUMlpAttNonLinBlockOp

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet