#include "../eigen_config.h"
#include "feed_forward.h"
#include "multi_head_attention.h"
#include "norm_layer.h"

// Encoder Layer class
// This combines Multi-Head Attention and Feed-Forward Network
class encoder_layer_t
{
private:
    int d_model;
    multi_head_attention_t self_attn;
    feed_forward_t ff;
    norm_layer_t norm1;
    norm_layer_t norm2;

public:
    encoder_layer_t(int d_model, int num_heads, int d_ff)
        : d_model(d_model), self_attn(d_model, num_heads), ff(d_model, d_ff), norm1(d_model), norm2(d_model) {}

    MatrixXf forward(const MatrixXf &X);
};