#include "attention.h"
#include <iostream>

MatrixXf attention_t::forward(const MatrixXf &X)
{
    // X is the input sequence, shape: [seq_len, d_model]

    // Create Query, Key, and Value matrices
    // These projections allow the model to focus on different aspects of the input 
    MatrixXf Q = (X * query_weights.transpose()).rowwise() + query_bias.transpose(); // Query: what we're looking for
    MatrixXf K = (X * key_weights.transpose()).rowwise() + key_bias.transpose(); // Key: what we match against 
    MatrixXf V = (X * value_weights.transpose()).rowwise() + value_bias.transpose(); // Value: what we retrieve

    // Compute attention scores
    // This step allows each position to attend to all other positions
    MatrixXf scores = Q * K.transpose() / std::sqrt(d_model);

    // Apply softmax to get attention weights
    // This converts scores to probabilities, allowing for a weighted sum
    MatrixXf attention_weights(scores.rows(), scores.cols());
    for (int i = 0; i < scores.rows(); ++i) {
        attention_weights.row(i) = softmax(scores.row(i).transpose()).transpose();
    }
    // Apply attention
    // This weighted sum allows the model to focus on relevant parts of the input
    return attention_weights * V;
}