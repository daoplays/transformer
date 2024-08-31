#pragma once

#include "../eigen_config.h"
#include <random>
#include "../utils.h"
#include <iostream>

// Multi-Head Attention class
// This is the core of the transformer architecture
class attention_t
{
private:
    int d_model, num_heads;
    MatrixXf query_weights, key_weights, value_weights;
    VectorXf query_bias, key_bias, value_bias;

public:
    attention_t(int d_model) : d_model(d_model)
    {
        // Initialize weights
        allocate_and_initialize(query_weights, d_model, d_model);
        allocate_and_initialize(key_weights, d_model, d_model);
        allocate_and_initialize(value_weights, d_model, d_model);

        // Initialize biases to zero
        query_bias = VectorXf::Zero(d_model);
        key_bias = VectorXf::Zero(d_model);
        value_bias = VectorXf::Zero(d_model);

    }

    MatrixXf forward(const MatrixXf &X);

    void set_weights(const MatrixXf& q, const MatrixXf& k, const MatrixXf& v,
                     const VectorXf& q_bias, const VectorXf& k_bias, const VectorXf& v_bias) {
        query_weights = q;
        key_weights = k;
        value_weights = v;
        query_bias = q_bias;
        key_bias = k_bias;
        value_bias = v_bias;
    }
};