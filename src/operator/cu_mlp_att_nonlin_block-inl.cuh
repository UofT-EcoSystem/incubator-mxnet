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

	RealType local_sum = 0;

	cg::thread_block this_cta = cg::this_thread_block();

	svmem_reduced_sum[threadIdx.x] = local_value;
	cg::sync(this_cta); // up to this point, the content of shared memory has been initialized with local variable
	// =========================================================================================
	// do reduction in shared memory
	if ((blockDim.x > 512) && (threadIdx.x < 512))
		svmem_reduced_sum[threadIdx.x] = local_sum += (threadIdx.x + 512) >= blockDim.x ? 0 : svmem_reduced_sum[threadIdx.x + 512];
	cg::sync(this_cta);
	if ((blockDim.x > 256) && (threadIdx.x < 256)) 
		svmem_reduced_sum[threadIdx.x] = local_sum += (threadIdx.x + 256) >= blockDim.x ? 0 : svmem_reduced_sum[threadIdx.x + 256];
	cg::sync(this_cta);
    	if ((blockDim.x > 128) && (threadIdx.x < 128))
		svmem_reduced_sum[threadIdx.x] = local_sum += (threadIdx.x + 128) >= blockDim.x ? 0 : svmem_reduced_sum[threadIdx.x + 128];
	cg::sync(this_cta);
	if ((blockDim.x >  64) && (threadIdx.x <  64))
		svmem_reduced_sum[threadIdx.x] = local_sum += (threadIdx.x +  64) >= blockDim.x ? 0 : svmem_reduced_sum[threadIdx.x +  64];
	cg::sync(this_cta);
	// =========================================================================================
	// further reduction will be done within a wrap, therefore no cta-level synchronization is needed
	if (threadIdx.x < 32)
	{
		cg::coalesced_group active_threads = cg::coalesced_threads();

		if (blockDim.x > 32) local_sum += (threadIdx.x + 32) >= blockDim.x ? 0 : svmem_reduced_sum[threadIdx.x + 32];

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
 * This kernel shall be launched using the parameter <<< (B, T), H, H * sizeof(RealType) >>>
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
	// declaring dynamic shared memory in templated CUDA kernel is tricky
	// please refer to this StackOverflow thread: 
	// https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
	extern __shared__ volatile unsigned char svmem[];

	volatile RealType * svmem_reduced_sum = reinterpret_cast < volatile RealType * > (svmem);

	if (layer_norm)
	{
		__cu_reduce_sum (svmem_reduced_sum,  att_hidden_reg);
		RealType exp_X = svmem_reduced_sum[0] / blockDim.x; // EXP[X]
		__cu_reduce_sum (svmem_reduced_sum, (att_hidden_reg - exp_X) *
		                                    (att_hidden_reg - exp_X));
		RealType var_X = svmem_reduced_sum[0] / blockDim.x; // VAR[X]
	
		// perform layer normalization
		att_hidden_reg = att_hidden_reg - exp_X;
#define VARIANCE_EPSILON 0.000001 // avoid the case when the variance is exactly 0
		att_hidden_reg = att_hidden_reg / sqrt(var_X + VARIANCE_EPSILON);
#undef  VARIANCE_EPSILON
	}

	/*
            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%shidden" % self.prefix)
	 */
	att_hidden[g_threadIdx] = tanh(att_hidden_reg);
}

// FullyConnected Layer Y = XW^T
// @param1 X [batch_size x input_dim]: (Input) Input Variable `X`
// @param2 W [num_hidden x input_dim]: (Input) Weight Parameter `W`
// @param3 O [batch_size x num_hidden]: (Output) Output Variable `Y`
// @param4 batch_size: (Parameter) Batch Size
// @param5 input_dim : (Parameter) Input Dimension
// @param6 num_hidden: (Parameter) Number of Hidden Units
template < typename RealType >
static inline void FullyConnected(cublasHandle_t cublas_handle,
	const RealType * const __restrict__ X,
	const RealType * const __restrict__ W,
	      RealType * const __restrict__ O,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden);

template <>
inline void FullyConnected < float  > (cublasHandle_t cublas_handle,
	const float  * const __restrict__ X,
	const float  * const __restrict__ W,
	      float  * const __restrict__ O,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden)
{
	float  alpha = 1.0, beta = 0.0;

	CUBLAS_CALL(cublasSgemm(cublas_handle, // cublas Handle
	                        CUBLAS_OP_T, // W.T
				CUBLAS_OP_N, // X
				num_hidden,  // Y.shape[1]
				batch_size,  // Y.shape[0]
				input_dim,   // W.shape[1]
				&alpha, W, input_dim, X, input_dim,
				& beta, O, num_hidden));
}

template <>
inline void FullyConnected < double > (cublasHandle_t cublas_handle,
	const double * const __restrict__ X,
	const double * const __restrict__ W,
	      double * const __restrict__ O,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden)
{
	double alpha = 1.0, beta = 0.0;

	CUBLAS_CALL(cublasDgemm(cublas_handle, // cublas Handle
	                        CUBLAS_OP_T, // W.T
				CUBLAS_OP_N, // X
				num_hidden,  // Y.shape[1]
				batch_size,  // Y.shape[0]
				input_dim,   // W.shape[1]
				&alpha, W, input_dim, X, input_dim,
				& beta, O, num_hidden));
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
			// TODO: Allocate training reserved space here.
		}
		 */
		// obtain the requested workspace
		Tensor < gpu, 3, DType > att_hidden = ctx.requested[int(EnumOpWorkspace::Workspace)]
			.get_space_typed < gpu, 3, DType > (
				Shape3(_param.batch_size, 
				       _param.seq_length,
				       _param.state_size), cuda_stream);

		_cuda_fused_mlp_att_nonlin_block__forward < DType >
			<<<
				dim3(_param.batch_size,
				     _param.seq_length),
				_param.state_size,
				_param.state_size * sizeof(DType),
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				src_hidden.dptr_,
				qry_hidden.dptr_,
				att_hidden.dptr_,
				_param.layer_norm
			);
		
		/*
            # (batch_size, seq_len, 1)
            attention_scores = mx.sym.FullyConnected(data=attention_hidden,
                                                     weight=self.att_h2s_weight,
                                                     num_hidden=1,
                                                     no_bias=True,
                                                     flatten=False,
                                                     name="%sraw_att_score_fc" % self.prefix)
		 */
		CHECK_EQ(cuda_stream->blas_handle_ownership_, Stream < gpu > ::OwnHandle) << 
			"Must initialize the cuBLAS handle in CUDA stream.";
		
		FullyConnected(Stream < gpu > ::GetBlasHandle(cuda_stream),
		               att_hidden.dptr_,
		               h2s_weight.dptr_,
			       att_scores.dptr_,
				_param.batch_size * _param.seq_length, 
				_param.state_size, 1);
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