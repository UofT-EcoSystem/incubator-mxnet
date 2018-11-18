#pragma once

#include <vector>
#include <cooperative_groups.h>

#include <mxnet/storage.h>

#include "mlp_att_scoring_func-inl.h"

namespace mxnet {
	namespace op {

#if defined(__CUDACC__)

/**
 * Forward Pass of the MLP Attention Layer Scoring Function
 * This kernel shall be launched using the parameter <<< (B, T), H, H * sizeof(RealType), cuda_stream >>>
 * @param1 qry_hidden [B     x H]:  (Input)  Query Hidden State
 * @param2 src_hidden [B x T x H]:  (Input) Source Hidden State
 * @param3 att_hidden [B x T x H]: (Output) Attention Hidden State
 * @param4 att_hidden_exp [B x T x H]: (Output) EXP_H[Attention Hidden State]
 * @param5 att_hidden_var [B x T x H]: (Output) VAR_H[Attention Hidden State]
 * The workspace reserved for EXP and VAR will ONLY be used during backward propagation.
 * @param6 layer_norm: (Parameter) Whether to perform Layer Normalization
 */
template < typename RealType >
static __global__ void _cuda_fused_mlp_att_scoring_func_forward(
	const RealType * const __restrict__ qry_hidden,
	const RealType * const __restrict__ src_hidden,
	      RealType * const __restrict__ att_hidden, 
	      RealType * const __restrict__ att_hidden_exp,
	      RealType * const __restrict__ att_hidden_var, 
	const bool layer_norm);

/**
 * Backward Pass of the MLP Attention Layer Scoring Function
 * This kernel shall be launched using the parameter <<< (B, T), H, H * sizeof(RealType), cuda_stream >>>
 * @param1 qry_hidden      [B     x H]:  (Input)  Query Hidden State
 * @param2 qry_hidden_grad [B     x H]: (Output)  Query Hidden State Gradient
 * @param3 src_hidden      [B x T x H]:  (Input) Source Hidden State
 * @param4 src_hidden_grad [B x T x H]: (Output) Source Hidden State Gradient
 * @param5 att_hidden      [B x T x H]:  (Input) Attention Hidden State
 * @param6 att_hidden_grad [B x T x H]:  (Input) Attention Hidden State Gradient
 * @param7 att_hidden_exp  [B x T]: (Input) EXP_H[Attention Hidden State]
 * @param8 att_hidden_var  [B x T]: (Input) VAR_H[Attention Hidden State]
 * @param9 layer_norm: (Parameter) Whether Layer Normalization is performed in the Forward Pass
 */
template < typename RealType >
static __global__ void _cuda_fused_mlp_att_scoring_func_backward(
	const RealType * const __restrict__ qry_hidden,
	      RealType * const __restrict__ qry_hidden_grad,
	const RealType * const __restrict__ src_hidden,
	      RealType * const __restrict__ src_hidden_grad,
	const RealType * const __restrict__ att_hidden,
	const RealType * const __restrict__ att_hidden_grad,
	const RealType * const __restrict__ att_hidden_exp,
	const RealType * const __restrict__ att_hidden_var,
	const bool layer_norm);

// FullyConnected Layer Y = X W^T Forward Pass
// @param1 X [batch_size x input_dim] :  (Input) Input  Variable  `X`
// @param2 W [num_hidden x input_dim] :  (Input) Weight Parameter `W`
// @param3 Y [batch_size x num_hidden]: (Output) Output Variable  `Y`
// @param4 batch_size: (Parameter) Batch Size
// @param5 input_dim : (Parameter) Input Dimension
// @param6 num_hidden: (Parameter) Number of Hidden Units
template < typename RealType >
static inline void FullyConnectedFW(cublasHandle_t cublas_handle,
	const RealType * const __restrict__ X,
	const RealType * const __restrict__ W,
	      RealType * const __restrict__ Y,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden);

// FullyConnected Layer Y = XW^T Backward Pass on Weight (dW = dY^T X)
// @param1  X [batch_size x input_dim] :  (Input)  Input Variable  `X`
// @param2 dW [num_hidden x input_dim] : (Output) Weight Parameter Gradient `dW`
// @param3 dY [batch_size x num_hidden]:  (Input) Output Gradient `dY`
// @param4 batch_size: (Parameter) Batch Size
// @param5 input_dim : (Parameter) Input Dimension
// @param6 num_hidden: (Parameter) Number of Hidden Units
template < typename RealType >
static inline void FullyConnectedBWWeight(cublasHandle_t cublas_handle,
	const RealType * const __restrict__  X,
	      RealType * const __restrict__ dW,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req, const unsigned batch_size, 
	const unsigned input_dim, const unsigned num_hidden);

// FullyConnected Layer Y = XW^T Backward Pass on Data (dX = dY W)
// @param1 dX [batch_size x input_dim] : (Output)  Input Gradient `dX`
// @param2  W [num_hidden x input_dim] :  (Input) Weight Parameter `W`
// @param3 dY [batch_size x num_hidden]:  (Input) Output Gradient `dY`
// @param4 batch_size: (Parameter) Batch Size
// @param5 input_dim : (Parameter) Input Dimension
// @param6 num_hidden: (Parameter) Number of Hidden Units
template < typename RealType >
static inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      RealType * const __restrict__ dX,
	const RealType * const __restrict__  W,
	const RealType * const __restrict__ dY,
	const OpReqType grad_req, const unsigned batch_size, 
	const unsigned input_dim, const unsigned num_hidden);

template < typename DType >
class CUMlpAttScoringFuncOp : public Operator
{
private:
	MlpAttScoringFuncParam _param;

	bool _initialized = false;

	// ReserveSpace
	Storage::Handle _reserved_space;
public:
	explicit CUMlpAttScoringFuncOp(MlpAttScoringFuncParam param)
	{
		_param = param;
	}
	~CUMlpAttScoringFuncOp() {}
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

		CHECK_EQ( in_data.size(),  in_expected); // QryHidden, SrcHidden, H2SWeight
		CHECK_EQ(out_data.size(), out_expected); // AttScores

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		Tensor < gpu, 2, DType > qry_hidden =  in_data[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > src_hidden =  in_data[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2s_weight =  in_data[int(EnumOpInputs ::H2SWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > att_scores = out_data[int(EnumOpOutputs::AttScores)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(qry_hidden.CheckContiguous(), true);
		CHECK_EQ(src_hidden.CheckContiguous(), true);
		CHECK_EQ(h2s_weight.CheckContiguous(), true);
		CHECK_EQ(att_scores.CheckContiguous(), true);

		if (!_initialized)
		{
			_Init(cuda_stream, in_data, out_data);
		}
		// obtain the requested workspace
		/*
		Tensor < gpu, 3, DType > att_hidden = ctx.requested[int(EnumOpWorkspace::AttHidden)]
			.get_space_typed < gpu, 3, DType > (
				Shape3(_param.batch_size, 
				       _param.seq_length,
				       _param.state_size), cuda_stream);
		 */
		Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
			.get_space_typed < gpu, 1, DType > (Shape1(
				_param.batch_size * _param.seq_length * _param.state_size), cuda_stream);
		
		DType * ptr_att_hidden = workspace.dptr_;

		_cuda_fused_mlp_att_scoring_func_forward < DType >
			<<<
				dim3(_param.seq_length, 
				     _param.batch_size),
				_param.state_size,
				_param.state_size * sizeof(DType),
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				qry_hidden.dptr_,
				src_hidden.dptr_,
				ptr_att_hidden,
				nullptr, nullptr,
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
		
		FullyConnectedFW(Stream < gpu > ::GetBlasHandle(cuda_stream),
		                 ptr_att_hidden,
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

		// The gradient request type is automatically determined by the `graph_executor`.
		// For variables that are exclusive to *this* operator, its gradient request type will be `WriteTo` (or `WriteInplace`).
		// Otherwise, its gradient request type will be `AddTo`.
		// In our case, `SrcHidden` and `H2SWeight` are both shared across multiple time steps,
		// so their gradient request type must be `AddTo`.
		// In contrast, each MLP attention operator has exclusive access to `QryHidden`,
		// so   its gradient request type must be either `WriteTo` or `WriteInplace`.
		// CHECK_NE(req[int(EnumOpInputs::QryHidden)], kAddTo) << // Note that the condition here is NOT_EQUAL.
		// 	"The gradient request for "  "query hidden" " must NOT be AddTo.";
		// CHECK_EQ(req[int(EnumOpInputs::SrcHidden)], kAddTo) << 
		// 	"The gradient request for " "source hidden" " must be AddTo.";
		// CHECK_EQ(req[int(EnumOpInputs::H2SWeight)], kAddTo) << 
		// 	"The gradient request for " "hidden-to-score weight" "must be AddTo.";

		Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

		// get the input data in the forward pass
		Tensor < gpu, 2, DType > qry_hidden      =  in_data[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > src_hidden      =  in_data[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2s_weight      =  in_data[int(EnumOpInputs ::H2SWeight)]
			.get < gpu, 2, DType > (cuda_stream);

		Tensor < gpu, 2, DType > qry_hidden_grad =  in_grad[int(EnumOpInputs ::QryHidden)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > src_hidden_grad =  in_grad[int(EnumOpInputs ::SrcHidden)]
			.get < gpu, 3, DType > (cuda_stream);
		Tensor < gpu, 2, DType > h2s_weight_grad =  in_grad[int(EnumOpInputs ::H2SWeight)]
			.get < gpu, 2, DType > (cuda_stream);
		Tensor < gpu, 3, DType > att_scores_grad = out_grad[int(EnumOpOutputs::AttScores)]
			.get < gpu, 3, DType > (cuda_stream);

		CHECK_EQ(qry_hidden     .CheckContiguous(), true);
		CHECK_EQ(src_hidden     .CheckContiguous(), true);
		CHECK_EQ(h2s_weight     .CheckContiguous(), true);
		CHECK_EQ(qry_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(src_hidden_grad.CheckContiguous(), true);
		CHECK_EQ(h2s_weight_grad.CheckContiguous(), true);
		CHECK_EQ(att_scores_grad.CheckContiguous(), true);

		// obtain the requested workspace
		Tensor < gpu, 1, DType > workspace = ctx.requested[int(EnumOpWorkspace::TempSpace)]
			.get_space_typed < gpu, 1, DType > (Shape1(
				2 * _param.batch_size * _param.seq_length * _param.state_size + 
				_param.layer_norm ? 2 * _param.batch_size * _param.seq_length : 0), cuda_stream);
		
		DType * ptr_att_hidden      = workspace.dptr_;
		DType * ptr_att_hidden_grad = workspace.dptr_ + 
			1 * _param.batch_size * _param.seq_length * _param.state_size;
		DType * ptr_att_hidden_exp  = 
			_param.layer_norm ? 
				workspace.dptr_ + 2 * _param.batch_size * _param.seq_length * _param.state_size : nullptr;
		DType * ptr_att_hidden_var  = 
			_param.layer_norm ? 
				workspace.dptr_ + 2 * _param.batch_size * _param.seq_length * _param.state_size + 
				                  1 * _param.batch_size * _param.seq_length : nullptr;

		// !Important: Replay the forward pass computation.
		_cuda_fused_mlp_att_scoring_func_forward < DType >
			<<<
				dim3(_param.seq_length,
				     _param.batch_size),
				_param.state_size,
				_param.state_size * sizeof(DType),
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				qry_hidden.dptr_,
				src_hidden.dptr_,
				ptr_att_hidden,
				ptr_att_hidden_exp,
				ptr_att_hidden_var,
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
		
		FullyConnectedBWWeight(Stream < gpu > ::GetBlasHandle(cuda_stream),
				       ptr_att_hidden,
				       h2s_weight_grad.dptr_,
				       att_scores_grad.dptr_,
				       req[int(EnumOpInputs::H2SWeight)],
				       _param.batch_size * _param.seq_length,
				       _param.state_size, 1);
		FullyConnectedBWData  (Stream < gpu > ::GetBlasHandle(cuda_stream),
		                       ptr_att_hidden_grad,
				       h2s_weight     .dptr_,
				       att_scores_grad.dptr_,
				       OpReqType::kWriteTo,
				       _param.batch_size * _param.seq_length,
				       _param.state_size, 1);

		_cuda_fused_mlp_att_scoring_func_backward
			<<<
				dim3(_param.seq_length,
				     _param.batch_size),
				_param.state_size, 
				_param.state_size * sizeof(DType), 
				Stream < gpu > ::GetStream(cuda_stream)
			>>>
			(
				qry_hidden.dptr_,
				qry_hidden_grad.dptr_,
				src_hidden.dptr_,
				src_hidden_grad.dptr_,
				ptr_att_hidden,
				ptr_att_hidden_grad,
				ptr_att_hidden_exp,
				ptr_att_hidden_var,
				_param.layer_norm
			);
	}
}; // class CUMlpAttScoringFuncOp

/**
 * Perform sum reduction across *this* thread block.
 * @param1 local_value: Local Value that is stored in Register
 * @param2 gmem_reduced_sum: [Optional] Global Memory for writting back the Result
 * @return the Result of Reduction across *this* Thread Block
 */
template < typename RealType >
static __forceinline__ __device__ RealType __cu_reduce_sum( 
	RealType                      local_value,
	RealType * const __restrict__ gmem_reduced_sum = nullptr)
{
	namespace cg = cooperative_groups;
	cg::thread_block this_cta = cg::this_thread_block();

	// declaring dynamic shared memory in templated CUDA kernel is tricky
	// please refer to this StackOverflow thread: 
	// https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
	extern __shared__ volatile unsigned char svmem[];
	volatile RealType * svmem_reduced_sum = 
		reinterpret_cast < volatile RealType * > (svmem);

	svmem_reduced_sum[threadIdx.x] = local_value;

	cg::sync(this_cta); // up to this point, the content of shared memory has been initialized with local variable
	// =========================================================================================
	// do reduction in shared memory
	if ((blockDim.x > 512) && (threadIdx.x < 512))
		svmem_reduced_sum[threadIdx.x] = 
			local_value += (threadIdx.x + 512) >= blockDim.x ? 
				0 : svmem_reduced_sum[threadIdx.x + 512];
	cg::sync(this_cta);
	if ((blockDim.x > 256) && (threadIdx.x < 256)) 
		svmem_reduced_sum[threadIdx.x] = 
			local_value += (threadIdx.x + 256) >= blockDim.x ? 
				0 : svmem_reduced_sum[threadIdx.x + 256];
	cg::sync(this_cta);
    	if ((blockDim.x > 128) && (threadIdx.x < 128))
		svmem_reduced_sum[threadIdx.x] = 
			local_value += (threadIdx.x + 128) >= blockDim.x ? 
				0 : svmem_reduced_sum[threadIdx.x + 128];
	cg::sync(this_cta);
	if ((blockDim.x >  64) && (threadIdx.x <  64))
		svmem_reduced_sum[threadIdx.x] = 
			local_value += (threadIdx.x +  64) >= blockDim.x ? 
				0 : svmem_reduced_sum[threadIdx.x +  64];
	cg::sync(this_cta);
	// =========================================================================================
	// further reduction will be done within a wrap, therefore no cta-level synchronization is needed
	if (threadIdx.x < 32)
	{
		cg::coalesced_group active_threads = cg::coalesced_threads();

		if (blockDim.x > 32) 
			svmem_reduced_sum[threadIdx.x] = 
				local_value += (threadIdx.x + 32) >= blockDim.x ? 
					0 : svmem_reduced_sum[threadIdx.x + 32];

#pragma unroll
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			svmem_reduced_sum[threadIdx.x] = 
				local_value += (threadIdx.x + offset) >= blockDim.x ? 
					0 : svmem_reduced_sum[threadIdx.x + offset];
		}
	}
	// also write the reduction result to global memory if it is provided
	if (gmem_reduced_sum != nullptr)
	{
		if (threadIdx.x == 0)
			gmem_reduced_sum[blockIdx.y *  gridDim.x + 
		        	                      blockIdx.x] = local_value;
		cg::sync(this_cta);
	}
	return svmem_reduced_sum[0];
}

template < typename RealType >
__global__ void _cuda_fused_mlp_att_scoring_func_forward(
	const RealType * const __restrict__ qry_hidden,
	const RealType * const __restrict__ src_hidden,
	      RealType * const __restrict__ att_hidden, 
	      RealType * const __restrict__ att_hidden_exp,
	      RealType * const __restrict__ att_hidden_var,
	const bool layer_norm)
{
	/*
            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.broadcast_add(lhs=attention_hidden_lhs, rhs=query_hidden,
                                                    name="%squery_plus_input" % self.prefix)
	 */
	const unsigned g_threadIdx = blockIdx.y *  gridDim.x *  blockDim.x + 
	                                          blockIdx.x *  blockDim.x + 
						               threadIdx.x;

	RealType att_hidden_reg = qry_hidden[blockIdx.y * blockDim.x + threadIdx.x] + 
	                          src_hidden[g_threadIdx];
	
	/*
            if self._ln is not None:
                attention_hidden = self._ln.normalize(attention_hidden)
	 */
	
	if (layer_norm)
	{
		RealType exp_X = __cu_reduce_sum( att_hidden_reg / blockDim.x,
		                                  att_hidden_exp);
		RealType var_X = __cu_reduce_sum((att_hidden_reg - exp_X) *
		                                 (att_hidden_reg - exp_X) / blockDim.x,
		                                  att_hidden_var);
		// perform layer normalization
		att_hidden_reg = att_hidden_reg - exp_X;
#define VARIANCE_EPSILON 0.000001 // avoid the case when the variance is exactly 0
		att_hidden_reg = att_hidden_reg / sqrt(var_X + VARIANCE_EPSILON);
	}

	/*
            # (batch_size, seq_len, attention_num_hidden)
            attention_hidden = mx.sym.Activation(attention_hidden, act_type="tanh",
                                                 name="%shidden" % self.prefix)
	 */
	att_hidden[g_threadIdx] = tanh(att_hidden_reg);
}

template < typename RealType >
__global__ void _cuda_fused_mlp_att_scoring_func_backward(
	const RealType * const __restrict__ qry_hidden,
	      RealType * const __restrict__ qry_hidden_grad,
	const RealType * const __restrict__ src_hidden,
	      RealType * const __restrict__ src_hidden_grad,
	const RealType * const __restrict__ att_hidden,
	const RealType * const __restrict__ att_hidden_grad,
	const RealType * const __restrict__ att_hidden_exp,
	const RealType * const __restrict__ att_hidden_var,
	const bool layer_norm)
{
	const unsigned g_threadIdx = blockIdx.y *  gridDim.x *  blockDim.x + 
	                                          blockIdx.x *  blockDim.x + 
						               threadIdx.x;
	
	// att_hidden[g_threadIdx] = tanh(att_hidden_reg);
	RealType att_hidden_reg      = att_hidden     [g_threadIdx];
	RealType att_hidden_grad_reg = att_hidden_grad[g_threadIdx] * 
		(1 - att_hidden_reg * 
		     att_hidden_reg);

	/*
            if self._ln is not None:
                attention_hidden = self._ln.normalize(attention_hidden)
	 */
	if (layer_norm)
	{
		// read the expected value and variance from the reserved workspace
		RealType att_hidden_exp_reg = att_hidden_exp[blockIdx.y * gridDim.x + blockIdx.x];
		RealType att_hidden_var_reg = att_hidden_var[blockIdx.y * gridDim.x + blockIdx.x];
		// 1 / sqrt(var + epsilon)
		RealType rsqrt_var_plus_epsilon = 1.0 / sqrt(att_hidden_var_reg + VARIANCE_EPSILON);		
#undef  VARIANCE_EPSILON
		// x - mu
		RealType att_hidden_minus_exp = qry_hidden[blockIdx.y * blockDim.x + threadIdx.x] + 
		                                src_hidden[g_threadIdx] - att_hidden_exp_reg;

		RealType att_hidden_exp_grad = __cu_reduce_sum(- att_hidden_grad_reg * 
								 rsqrt_var_plus_epsilon);
		RealType att_hidden_var_grad = __cu_reduce_sum(- att_hidden_grad_reg * 
								 rsqrt_var_plus_epsilon * 
								 rsqrt_var_plus_epsilon *
								 rsqrt_var_plus_epsilon *
								 att_hidden_minus_exp / 2.0);
		att_hidden_grad_reg = att_hidden_grad_reg * rsqrt_var_plus_epsilon + 
		                      att_hidden_exp_grad / blockDim.x + 
				      att_hidden_var_grad / blockDim.x * 2 * att_hidden_minus_exp;
	}

	/*
	RealType att_hidden_reg = src_hidden[g_threadIdx] + 
	                          qry_hidden[blockIdx.y * blockDim.x + threadIdx.x];
	 */

	src_hidden_grad[g_threadIdx] += att_hidden_grad_reg;
	atomicAdd(&qry_hidden_grad[blockIdx.y * blockDim.x + threadIdx.x], att_hidden_grad_reg);
}

template <>
inline void FullyConnectedFW < float > (cublasHandle_t cublas_handle,
	const float * const __restrict__ X,
	const float * const __restrict__ W,
	      float * const __restrict__ Y,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden)
{
	float alpha = 1.0, beta = 0.0;

	CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle
	                        CUBLAS_OP_T, // W.T
				CUBLAS_OP_N, // X
				num_hidden,  // Y.shape[1]
				batch_size,  // Y.shape[0]
				input_dim,   // W.shape[1]
				&alpha, W, input_dim, X, input_dim,
				& beta, Y, num_hidden));
}

template <>
inline void FullyConnectedFW < double > (cublasHandle_t cublas_handle,
	const double * const __restrict__ X,
	const double * const __restrict__ W,
	      double * const __restrict__ Y,
	const unsigned batch_size, const unsigned input_dim, const unsigned num_hidden)
{
	double alpha = 1.0, beta = 0.0;

	CUBLAS_CALL(cublasDgemm(cublas_handle, // cuBLAS Handle
	                        CUBLAS_OP_T, // W.T
				CUBLAS_OP_N, // X
				num_hidden,  // Y.shape[1]
				batch_size,  // Y.shape[0]
				input_dim,   // W.shape[1]
				&alpha, W, input_dim, X, input_dim,
				& beta, Y, num_hidden));
}

template <>
inline void FullyConnectedBWWeight < float >  (cublasHandle_t cublas_handle,
	const float * const __restrict__  X,
	      float * const __restrict__ dW,
	const float * const __restrict__ dY,
	const OpReqType grad_req, const unsigned batch_size,
	const unsigned input_dim, const unsigned num_hidden)
{
	float alpha = 1.0, beta = float(grad_req == kAddTo);

	CUBLAS_CALL(cublasSgemm(cublas_handle, // cuBLAS Handle
	                        CUBLAS_OP_N, //  X
				CUBLAS_OP_T, // dY^T
				input_dim,   // dW.shape[1]
				num_hidden,  // dW.shape[0]
				batch_size,  //  X.shape[0]
	                        &alpha,  X, input_dim, dY, num_hidden,
				& beta, dW, input_dim));
}

template <>
inline void FullyConnectedBWWeight < double > (cublasHandle_t cublas_handle,
	const double * const __restrict__  X,
	      double * const __restrict__ dW,
	const double * const __restrict__ dY,
	const OpReqType grad_req, const unsigned batch_size,
	const unsigned input_dim, const unsigned num_hidden)
{
	double alpha = 1.0, beta = double(grad_req == kAddTo);

	CUBLAS_CALL(cublasDgemm(cublas_handle, // cuBLAS Handle
	                        CUBLAS_OP_N, //  X
				CUBLAS_OP_T, // dY^T
				input_dim,   // dW.shape[1]
				num_hidden,  // dW.shape[0]
				batch_size,  //  X.shape[0]
	                        &alpha,  X, input_dim, dY, num_hidden,
				& beta, dW, input_dim));
}

template <>
inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      float * const __restrict__ dX,
	const float * const __restrict__  W,
	const float * const __restrict__ dY, 
	const OpReqType grad_req, const unsigned batch_size,
	const unsigned input_dim, const unsigned num_hidden)
{
	float alpha = 1.0, beta = float(grad_req == kAddTo);

	CUBLAS_CALL(cublasSgemm(cublas_handle, //cuBLAS Handle
	                        CUBLAS_OP_N, //  W
				CUBLAS_OP_N, // dY
				input_dim,   // dX.shape[1]
				batch_size,  // dX.shape[0]
				num_hidden,  //  W.shape[0]
				&alpha,  W, input_dim, dY, num_hidden,
				& beta, dX, input_dim));
}

template <>
inline void FullyConnectedBWData  (cublasHandle_t cublas_handle,
	      double * const __restrict__ dX,
	const double * const __restrict__  W, 
	const double * const __restrict__ dY,
	const OpReqType grad_req, const unsigned batch_size,
	const unsigned input_dim, const unsigned num_hidden)
{
	double alpha = 1.0, beta = double(grad_req == kAddTo);

	CUBLAS_CALL(cublasDgemm(cublas_handle, //cuBLAS Handle
	                        CUBLAS_OP_N, //  W
				CUBLAS_OP_N, // dY
				input_dim,   // dX.shape[1]
				batch_size,  // dX.shape[0]
				num_hidden,  //  W.shape[0]
				&alpha,  W, input_dim, dY, num_hidden,
				& beta, dX, input_dim));
}

#endif // __CUDACC__

	} // namespace op
} // namespace mxnet