#include "../eigen_config.h"
#include "feed_forward.h"
#include "multi_head_attention.h"

// Encoder Layer class
// This combines Multi-Head Attention and Feed-Forward Network
class encoder_layer_t
{
private:
    multi_head_attention_t self_attn;
    feed_forward_t ff;
    int d_model;

public:
    encoder_layer_t(int d_model, int num_heads, int d_ff)
        : self_attn(d_model, num_heads), ff(d_model, d_ff), d_model(d_model) {}

    MatrixXf forward(const MatrixXf &X);
};