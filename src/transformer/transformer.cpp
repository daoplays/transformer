#include "transformer.h"
#include "encoder_layer.h"

MatrixXf transformer_t::forward(const MatrixXf& X)
{
    // X is the input sequence, shape: [seq_len, d_model]
    MatrixXf output = X;
    // Pass input through each encoder layer
    for (encoder_layer_t& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

void transformer_t::set_layer_weights(const int layer_idx, const MatrixXf& self_attn_qkv_weight,  const VectorXf& self_attn_qkv_bias, 
                           const MatrixXf& self_attn_out_proj_weight, const VectorXf& self_attn_out_proj_bias,
                           const VectorXf& norm1_gamma, const VectorXf& norm1_beta, const MatrixXf& ff_linear1_weight,
                           const VectorXf& ff_linear1_bias, const MatrixXf& ff_linear2_weight, const VectorXf& ff_linear2_bias,
                           const VectorXf& norm2_gamma, const VectorXf& norm2_beta)
    {
        layers[layer_idx].set_weights(self_attn_qkv_weight, self_attn_qkv_bias, self_attn_out_proj_weight, self_attn_out_proj_bias, norm1_gamma, norm1_beta,
                                      ff_linear1_weight, ff_linear1_bias, ff_linear2_weight, ff_linear2_bias, norm2_gamma, norm2_beta);
    }