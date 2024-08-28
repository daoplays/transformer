#pragma once

#include "../eigen_config.h"
#include <vector>
#include <cmath>
#include <random>
#include "utils.h"

// Feed-Forward Network class
// This adds non-linearity and increases the model's capacity
class feed_forward_t
{
private:
    MatrixXf W1, W2;
    int d_model, d_ff;

public:
    feed_forward_t(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff)
    {
       allocate_and_initialize(W1, d_model, d_ff);
       allocate_and_initialize(W2, d_ff, d_model);
    }

    MatrixXf forward(const MatrixXf &X);

    void set_weights(const Eigen::MatrixXf& new_W1, const Eigen::MatrixXf& new_W2) {
        // Check if the dimensions of the new weights match the expected dimensions
        if (new_W1.rows() != d_model || new_W1.cols() != d_ff ||
            new_W2.rows() != d_ff || new_W2.cols() != d_model) {
            throw std::invalid_argument("New weights have incorrect dimensions");
        }

        // If dimensions are correct, set the new weights
        W1 = new_W1;
        W2 = new_W2;
    }
};