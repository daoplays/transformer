#pragma once

#include <vector>
#include "../eigen_config.h"
#include "encoder_layer.h"

// transformer_t class
// This stacks multiple Encoder Layers
class transformer_t {
private:

    std::vector<encoder_layer_t> layers;

public:

    transformer_t(int num_layers, int d_model, int num_heads, int d_ff)
    {
        // Create multiple encoder layers
        for (int i = 0; i < num_layers; i++) {
            layers.emplace_back(d_model, num_heads, d_ff);
        }
    }

    MatrixXf forward(const MatrixXf& X);

    void set_layer_weights(const int layer_idx, const MatrixXf& self_attn_q_weight, const MatrixXf& self_attn_k_weight,
                           const MatrixXf& self_attn_v_weight, const VectorXf& self_attn_q_bias, const VectorXf& self_attn_k_bias,
                           const VectorXf& self_attn_v_bias, const MatrixXf& self_attn_out_proj_weight, const VectorXf& self_attn_out_proj_bias,
                           const VectorXf& norm1_gamma, const VectorXf& norm1_beta, const MatrixXf& ff_linear1_weight,
                           const VectorXf& ff_linear1_bias, const MatrixXf& ff_linear2_weight, const VectorXf& ff_linear2_bias,
                           const VectorXf& norm2_gamma, const VectorXf& norm2_beta)
    {
        layers[layer_idx].set_weights(self_attn_q_weight, self_attn_k_weight, self_attn_v_weight, self_attn_q_bias, self_attn_k_bias,
                                      self_attn_v_bias, self_attn_out_proj_weight, self_attn_out_proj_bias, norm1_gamma, norm1_beta,
                                      ff_linear1_weight, ff_linear1_bias, ff_linear2_weight, ff_linear2_bias, norm2_gamma, norm2_beta);
    }
};
