#include "attention.h"
#include <iostream>

MatrixXf attention_t::forward(const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V)
{

    int d_model = Q.cols();
    // Compute attention scores
    // This step allows each position to attend to all other positions
    MatrixXf scores = Q * K.transpose() / std::sqrt(d_model);

    // Create and apply causal mask
    for (int i = 0; i < scores.rows(); ++i) {
        for (int j = i + 1; j < scores.rows(); ++j) {
            scores(i, j) = -std::numeric_limits<float>::infinity();
        }
    }

    //std::cout << "scores: " << scores.rows() << " " << scores.cols() << std::endl;
    //s/td::cout << scores.row(0).head(10) << std::endl;

    // Apply softmax to get attention weights
    // This converts scores to probabilities, allowing for a weighted sum
    MatrixXf attention_weights(scores.rows(), scores.cols());
    for (int i = 0; i < scores.rows(); ++i) {
        attention_weights.row(i) = softmax(scores.row(i).transpose()).transpose();
    }

    //std::cout << "return" << std::endl;
    // Apply attention
    // This weighted sum allows the model to focus on relevant parts of the input
    return attention_weights * V;
}