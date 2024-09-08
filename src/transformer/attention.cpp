#include "attention.h"
#include <iostream>

MatrixXf attention_t::forward(const MatrixXf &Q, const MatrixXf &K, const MatrixXf &V)
{

    int d_model = Q.cols();
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