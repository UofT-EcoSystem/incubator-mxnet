#pragma once

#include "layer_norm-inl.h"
#include "layer_norm-kernel.h"

namespace mxnet {
        namespace op {

template < typename DType >
class CULayerNormOp : public Operator
{
private:
        LayerNormParam _param;
        bool _initialized = false;
public:
        explicit CULayerNormOp(LayerNormParam param)
        {
                _param = param;
        }
private:
        void _Init(mshadow::Stream < gpu > * cuda_stream,
                   const std::vector < TBlob > &  in_data,
                   const std::vector < TBlob > & out_data)
        {
                using namespace mshadow;

                CHECK_EQ(_initialized, false);

                _param.state_size = in_data[int(EnumOpInputs::Data)].shape_
                                           [in_data[int(EnumOpInputs::Data)].shape_.ndim() - 1];
                _param.batch_size = in_data[int(EnumOpInputs::Data)].shape_.Size() / _param.state_size;

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

                std::size_t in_expected = 3, out_expected = 3;

                CHECK_EQ( in_data.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected);

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                TBlob data   =  in_data[int(EnumOpInputs ::Data)];
                TBlob gamma  =  in_data[int(EnumOpInputs ::Gamma)];
                TBlob beta   =  in_data[int(EnumOpInputs ::Beta)];
                TBlob output = out_data[int(EnumOpOutputs::Output)];
                TBlob mean   = out_data[int(EnumOpOutputs::Mean)];
                TBlob std    = out_data[int(EnumOpOutputs::STD)];

                CHECK_EQ(data  .CheckContiguous(), true);
                CHECK_EQ(gamma .CheckContiguous(), true);
                CHECK_EQ(beta  .CheckContiguous(), true);
                CHECK_EQ(output.CheckContiguous(), true);
                CHECK_EQ(mean  .CheckContiguous(), true);
                CHECK_EQ(std   .CheckContiguous(), true);

                if (!_initialized)
                {
                        _Init(cuda_stream, in_data, out_data);
                }

                unsigned blockDim_y;

                if (_param.state_size <= 128)
                {
                        blockDim_y = 1;
                }
                else if (_param.state_size <= 512)
                {
                        blockDim_y = 2;
                }
                else 
                {
                        blockDim_y = 4;
                }

                LayerNormFusedForwardKernelContig < double, DType, int > <<<
                        _param.batch_size,
                        dim3(32, blockDim_y),
                        blockDim_y > 1 ?  blockDim_y      * 32 * sizeof(double) + 
                                         (blockDim_y / 2) * 32 * sizeof(int) 
                                       :  0,
                        Stream < gpu > ::GetStream(cuda_stream)
                        >>> (
                        _param.batch_size,
                        _param.state_size,
                        _param.eps,
                        data  .dptr < DType > (),
                        gamma .dptr < DType > (),
                        beta  .dptr < DType > (),
                        output.dptr < DType > (),
                        mean  .dptr < DType > (),
                        std   .dptr < DType > ());
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

                std::size_t in_expected = 3, out_expected = 3, visible_out_expected = 1;

                CHECK_EQ( in_data.size(),  in_expected);  // data, gamma, beta
                CHECK_EQ( in_grad.size(),  in_expected);
                CHECK_EQ(     req.size(),  in_expected);
                CHECK_EQ(out_data.size(), out_expected);  // output, mean, std
                CHECK_EQ(out_grad.size(), visible_out_expected);

                Stream < gpu > * cuda_stream = ctx.get_stream < gpu > ();

                TBlob data        =  in_data[int(EnumOpInputs ::Data)];
                TBlob gamma       =  in_data[int(EnumOpInputs ::Gamma)];
                TBlob mean        = out_data[int(EnumOpOutputs::Mean)];
                TBlob std         = out_data[int(EnumOpOutputs::STD)];
                TBlob data_grad   =  in_grad[int(EnumOpInputs ::Data)];
                TBlob gamma_grad  =  in_grad[int(EnumOpInputs ::Gamma)];
                TBlob beta_grad   =  in_grad[int(EnumOpInputs ::Beta)];
                TBlob output_grad = out_grad[int(EnumOpOutputs::Output)];

                CHECK_EQ(data       .CheckContiguous(), true);
                CHECK_EQ(gamma      .CheckContiguous(), true);
                CHECK_EQ(mean       .CheckContiguous(), true);
                CHECK_EQ(std        .CheckContiguous(), true);
                CHECK_EQ(data_grad  .CheckContiguous(), true);
                CHECK_EQ(gamma_grad .CheckContiguous(), true);
                CHECK_EQ(beta_grad  .CheckContiguous(), true);
                CHECK_EQ(output_grad.CheckContiguous(), true);

                OpReqType data_grad_req  = req[int(EnumOpInputs::Data)];
                OpReqType gamma_grad_req = req[int(EnumOpInputs::Gamma)];
                OpReqType beta_grad_req  = req[int(EnumOpInputs::Beta)];

                CHECK_NE(data_grad_req,  kWriteInplace);
                CHECK_NE(gamma_grad_req, kWriteInplace);
                CHECK_NE(beta_grad_req,  kWriteInplace);

                dim3 part_grad_block_dim, 
                     part_grad_grid_dim, 
                     gb_block_dim, gb_grid_dim;
                int npart;

                GetGammaBetaGradKernelParams(_param.batch_size,
                                             _param.state_size,
                                             &part_grad_block_dim,
                                             &part_grad_grid_dim,
                                             &gb_block_dim,
                                             &gb_grid_dim, &npart);
                Tensor < gpu, 1, double > workspace = ctx.requested[0]
                        .get_space_typed < gpu, 1, double > (
                                Shape1(2 * npart * _param.state_size), cuda_stream);
                double * part_gamma_grad_ptr = workspace.dptr_;
                double * part_beta_grad_ptr  = workspace.dptr_ + npart * _param.state_size;

                const int nshared_K1 = 2 * (part_grad_block_dim.x + 1) * 
                                            part_grad_block_dim.y * sizeof(double);
                const int nshared_K2 = 2 * gb_block_dim.x * gb_block_dim.y * sizeof(double);
                
                LayerNormFusedBackwardKernel_PartGammaBeta <<<
                        part_grad_grid_dim,
                        part_grad_block_dim,
                        nshared_K1,
                        Stream < gpu > ::GetStream(cuda_stream)
                        >>> (
                        _param.batch_size,
                        _param.state_size,
                        data       .dptr < DType > (),
                        output_grad.dptr < DType > (),
                        mean       .dptr < DType > (),
                        std        .dptr < DType > (),
                        part_gamma_grad_ptr,
                        part_beta_grad_ptr);
                if (gamma_grad_req == kAddTo && beta_grad_req != kAddTo)
                {
                        LayerNormFusedBackwardKernel_GammaBeta < true, false > <<<
                                gb_grid_dim,
                                gb_block_dim, 
                                nshared_K2, 
                                Stream < gpu > ::GetStream(cuda_stream) 
                                >>> (
                                _param.batch_size,
                                _param.state_size,
                                npart,
                                part_gamma_grad_ptr,
                                part_beta_grad_ptr,
                                gamma_grad.dptr < DType > (), 
                                beta_grad .dptr < DType > ());
                }
                else if (gamma_grad_req != kAddTo && beta_grad_req == kAddTo)
                {
                        LayerNormFusedBackwardKernel_GammaBeta < false, true > <<<
                                gb_grid_dim,
                                gb_block_dim, 
                                nshared_K2, 
                                Stream < gpu > ::GetStream(cuda_stream) 
                                >>> (
                                _param.batch_size,
                                _param.state_size,
                                npart,
                                part_gamma_grad_ptr,
                                part_beta_grad_ptr,
                                gamma_grad.dptr < DType > (), 
                                beta_grad .dptr < DType > ());
                }
                else if (gamma_grad_req == kAddTo && beta_grad_req == kAddTo)
                {
                        LayerNormFusedBackwardKernel_GammaBeta < true, true > <<<
                                gb_grid_dim,
                                gb_block_dim, 
                                nshared_K2, 
                                Stream < gpu > ::GetStream(cuda_stream) 
                                >>> (
                                _param.batch_size,
                                _param.state_size,
                                npart,
                                part_gamma_grad_ptr,
                                part_beta_grad_ptr,
                                gamma_grad.dptr < DType > (), 
                                beta_grad .dptr < DType > ());
                }
                else
                {
                        LayerNormFusedBackwardKernel_GammaBeta < false, false > <<<
                                gb_grid_dim,
                                gb_block_dim, 
                                nshared_K2, 
                                Stream < gpu > ::GetStream(cuda_stream) 
                                >>> (
                                _param.batch_size,
                                _param.state_size,
                                npart,
                                part_gamma_grad_ptr,
                                part_beta_grad_ptr,
                                gamma_grad.dptr < DType > (), 
                                beta_grad .dptr < DType > ());
                }

                unsigned blockDim_y;

                if (_param.state_size <= 32)
                {
                        blockDim_y = 1;
                }
                else if (_param.state_size <= 128)
                {
                        blockDim_y = 2;
                }
                else if (_param.state_size <= 512)
                {
                        blockDim_y = 4;
                }
                else 
                {
                        blockDim_y = 8;
                }

                if (data_grad_req == kAddTo)
                {
                        LayerNormFusedBackwardKernel_Data < 4, true, double > <<< 
                                _param.batch_size,
                                dim3(32, blockDim_y),
                                blockDim_y > 1 ? blockDim_y * 32 * sizeof(double) : 0,
                                Stream < gpu > ::GetStream(cuda_stream) >>> (
                                _param.batch_size,
                                _param.state_size,
                                data.dptr < DType > (),
                                output_grad.dptr < DType > (),
                                mean       .dptr < DType > (),
                                std        .dptr < DType > (),
                                gamma      .dptr < DType > (),
                                data_grad  .dptr < DType > ());
                }
                else
                {
                        LayerNormFusedBackwardKernel_Data < 4, false, double > <<< 
                                _param.batch_size,
                                dim3(32, blockDim_y),
                                blockDim_y > 1 ? blockDim_y * 32 * sizeof(double) : 0,
                                Stream < gpu > ::GetStream(cuda_stream) >>> (
                                _param.batch_size,
                                _param.state_size,
                                data.dptr < DType > (),
                                output_grad.dptr < DType > (),
                                mean       .dptr < DType > (),
                                std        .dptr < DType > (),
                                gamma      .dptr < DType > (),
                                data_grad  .dptr < DType > ());
                }
        }
};

        } // namespace op 
}  // namespace mxnet
