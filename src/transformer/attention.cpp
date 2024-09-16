#include "attention.h"
#include <iostream>

MatrixXf attention_t::forward(const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V, bool causal)
{

    int d_model = Q.cols();
    // Compute attention scores
    // This step allows each position to attend to all other positions

    // X * C * X^T, where C = Wq * Wk^T.
    //You're correct that if it were Wq * Wq^T, it would be analogous to a covariance matrix.
    //The attention mechanism using Wq * Wk^T can indeed be seen as a generalized or asymmetric version of this.


    //Interpretation:

    //In a covariance matrix (Wq * Wq^T), you're measuring how each dimension varies with every other dimension in the same space.
    //In attention (Wq * Wk^T), you're measuring how each dimension in the "query space" relates to each dimension in the "key space".
    MatrixXf scores = Q * K.transpose() / std::sqrt(d_model);

    if (causal) {
        // Create and apply causal mask
        for (int i = 0; i < scores.rows(); ++i) {
            for (int j = i + 1; j < scores.rows(); ++j) {
                scores(i, j) = -std::numeric_limits<float>::infinity();
            }
        }
    }

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