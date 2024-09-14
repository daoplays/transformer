#pragma once

#include <iostream>
#include <random>
#include "../eigen_config.h"
#include "../utils.h"

// Multi-Head Attention class
// This is the core of the transformer architecture
class attention_t {
public:

    MatrixXf forward(const MatrixXf& Q, const MatrixXf& K, const MatrixXf& V, bool causal = true);
};