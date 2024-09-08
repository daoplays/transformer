#pragma once

#include "../eigen_config.h"
#include <random>
#include "../utils.h"
#include <iostream>

// Multi-Head Attention class
// This is the core of the transformer architecture
class attention_t
{
public:
   
    MatrixXf forward(const MatrixXf &Q, const MatrixXf &K, const MatrixXf &V);
};