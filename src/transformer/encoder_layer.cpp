#include "encoder_layer.h"

MatrixXf encoder_layer_t::forward(const MatrixXf &X)
{
    // Self-attention
    MatrixXf attn_output = self_attn.forward(X);

    // Add & Norm
    MatrixXf norm1_output = norm1.forward(X + attn_output);

    // Feed-forward
    MatrixXf ff_output = ff.forward(norm1_output);
    // Add & Norm
    MatrixXf norm2_output = norm2.forward(norm1_output + ff_output);

    return norm2_output;
}