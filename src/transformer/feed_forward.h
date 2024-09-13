#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../eigen_config.h"
#include "utils.h"

// Feed-Forward Network class
// This adds non-linearity and increases the model's capacity
class feed_forward_t {
private:

    int d_model, d_ff;
    MatrixXf W1, W2;
    VectorXf b1, b2;

public:

    feed_forward_t(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff)
    {
        allocate_and_initialize(W1, d_ff, d_model);
        allocate_and_initialize(W2, d_model, d_ff);

        b1 = Eigen::VectorXf::Zero(d_ff);
        b2 = Eigen::VectorXf::Zero(d_model);
    }

    MatrixXf forward(const MatrixXf& X);
    MatrixXf forward2(const MatrixXf& X);

    void set_weights(const Eigen::MatrixXf& new_W1, const Eigen::MatrixXf& new_W2, const Eigen::VectorXf& new_b1, const Eigen::VectorXf& new_b2)
    {
        // Check if the dimensions of the new weights match the expected dimensions
        if (new_W1.rows() != d_ff || new_W1.cols() != d_model || new_W2.rows() != d_model || new_W2.cols() != d_ff || new_b1.size() != d_ff ||
            new_b2.size() != d_model) {
            std::cout << "new_W1: " << new_W1.rows() << " " << new_W1.cols() << std::endl;
            std::cout << "new_W2: " << new_W2.rows() << " " << new_W2.cols() << std::endl;
            std::cout << "new_b1: " << new_b1.size() << std::endl;
            std::cout << "new_b2: " << new_b2.size() << std::endl;
            std::cout << "d_ff: " << d_ff << std::endl;
            std::cout << "d_model: " << d_model << std::endl;
            die("New weights or biases have incorrect dimensions");
        }

        // If dimensions are correct, set the new weights
        W1 = new_W1;
        W2 = new_W2;
        b1 = new_b1;
        b2 = new_b2;
    }
};