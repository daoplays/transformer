#pragma once

#include "../eigen_config.h"
#include <random>
#include "../utils.h"

// Multi-Head Attention class
// This is the core of the transformer architecture
class attention_t
{
private:
    int d_model, num_heads;
    MatrixXf query_weights, key_weights, value_weights;

public:
    attention_t(int d_model) : d_model(d_model)
    {
        // Initialize weights
        allocate_and_initialize(query_weights, d_model, d_model);
        allocate_and_initialize(key_weights, d_model, d_model);
        allocate_and_initialize(value_weights, d_model, d_model);
    }

    MatrixXf forward(const MatrixXf &X);

    void set_weights(const MatrixXf& q, const MatrixXf& k, const MatrixXf& v) {
        query_weights = q;
        key_weights = k;
        value_weights = v;
    }
};