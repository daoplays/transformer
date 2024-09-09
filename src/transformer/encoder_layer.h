#include "../eigen_config.h"
#include "feed_forward.h"
#include "multi_head_attention.h"
#include "norm_layer.h"

// Encoder Layer class
// This combines Multi-Head Attention and Feed-Forward Network
class encoder_layer_t {
private:

    int d_model;
    multi_head_attention_t self_attn;
    feed_forward_t ff;
    norm_layer_t norm1;
    norm_layer_t norm2;

public:

    encoder_layer_t(int d_model, int num_heads, int d_ff)
        : d_model(d_model), self_attn(d_model, num_heads), ff(d_model, d_ff), norm1(d_model), norm2(d_model)
    {
    }

    MatrixXf forward(const MatrixXf& X);

    void set_weights(const MatrixXf& self_attn_q_weight, const MatrixXf& self_attn_k_weight, const MatrixXf& self_attn_v_weight,
                     const VectorXf& self_attn_q_bias, const VectorXf& self_attn_k_bias, const VectorXf& self_attn_v_bias,
                     const MatrixXf& self_attn_out_proj_weight, const VectorXf& self_attn_out_proj_bias, const VectorXf& norm1_gamma,
                     const VectorXf& norm1_beta, const MatrixXf& ff_linear1_weight, const VectorXf& ff_linear1_bias,
                     const MatrixXf& ff_linear2_weight, const VectorXf& ff_linear2_bias, const VectorXf& norm2_gamma, const VectorXf& norm2_beta)
    {
        // Set weights for self-attention
        self_attn.set_weights(self_attn_q_weight, self_attn_k_weight, self_attn_v_weight, self_attn_q_bias, self_attn_k_bias, self_attn_v_bias,
                              self_attn_out_proj_weight, self_attn_out_proj_bias);

        // Set gamma and beta for first layer norm
        norm1.setGammaBeta(norm1_gamma, norm1_beta);

        // Set weights for feed-forward network
        ff.set_weights(ff_linear1_weight, ff_linear2_weight, ff_linear1_bias, ff_linear2_bias);

        // Set gamma and beta for second layer norm
        norm2.setGammaBeta(norm2_gamma, norm2_beta);
    }
};