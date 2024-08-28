#include "multi_head_attention.h"

MatrixXf multi_head_attention_t::forward(const MatrixXf &X)
{
    // X is the input sequence, shape: [seq_len, d_model]

    // Create Query, Key, and Value matrices
    // These projections allow the model to focus on different aspects of the input
    MatrixXf Q = X * W_q; // Query: what we're looking for
    MatrixXf K = X * W_k; // Key: what we match against
    MatrixXf V = X * W_v; // Value: what we retrieve

    // Compute attention scores
    // This step allows each position to attend to all other positions
    MatrixXf scores = Q * K.transpose() / std::sqrt(d_model);

    // Apply softmax to get attention weights
    // This converts scores to probabilities, allowing for a weighted sum
    MatrixXf attention_weights = scores.array().exp();
    attention_weights = attention_weights.array().colwise() / attention_weights.rowwise().sum().array();

    // Apply attention
    // This weighted sum allows the model to focus on relevant parts of the input
    MatrixXf output = attention_weights * V;
    return output * W_o; // Final projection
}