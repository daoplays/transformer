#include "decoder_layer.h"

MatrixXf decoder_layer_t::forward(const MatrixXf& X)
{
    // Layer Norm 1
    MatrixXf norm1_output = norm1.forward(X);

    // Self-attention
    MatrixXf attn_output = self_attn.forward(norm1_output);

    // Residual connection 1
    MatrixXf residual1 = X + attn_output;

    // Layer Norm 2
    MatrixXf norm2_output = norm2.forward(residual1);

    // Feed-forward
    MatrixXf ff_output = ff.forward(norm2_output);

    // Residual connection 2
    MatrixXf residual2 = residual1 + ff_output;

    return residual2;
}