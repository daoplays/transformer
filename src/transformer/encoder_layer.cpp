#include "encoder_layer.h"

MatrixXf encoder_layer_t::forward(const MatrixXf& X)
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

MatrixXf encoder_layer_t::forward2(const MatrixXf& X)
{
    // Layer Norm 1
    MatrixXf norm1_output = norm1.forward(X);

    // Self-attention
    MatrixXf attn_output = self_attn.forward2(norm1_output);

    // Residual connection 1
    MatrixXf residual1 = X + attn_output;

    // Layer Norm 2
    MatrixXf norm2_output = norm2.forward(residual1);

    // Feed-forward
    MatrixXf ff_output = ff.forward2(norm2_output);

    // Residual connection 2
    MatrixXf residual2 = residual1 + ff_output;

    return residual2;
}