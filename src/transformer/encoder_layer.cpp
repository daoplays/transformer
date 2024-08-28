#include "encoder_layer.h"

MatrixXf encoder_layer_t::forward(const MatrixXf &X)
{
    // Self-attention
    MatrixXf attn_output = self_attn.forward(X);
    // Add & Norm (simplified, just adding for now)
    MatrixXf norm1 = X + attn_output;

    // Feed-forward
    MatrixXf ff_output = ff.forward(norm1);
    // Add & Norm (simplified, just adding for now)
    MatrixXf norm2 = norm1 + ff_output;

    return norm2;
}