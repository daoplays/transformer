#pragma once

#include "../eigen_config.h"
#include <vector>
#include "encoder_layer.h"

// transformer_t class
// This stacks multiple Encoder Layers
class transformer_t
{
private:
    std::vector<encoder_layer_t> layers;

public:
    transformer_t(int num_layers, int d_model, int num_heads, int d_ff)
    {
        // Create multiple encoder layers
        for (int i = 0; i < num_layers; i++)
        {
            layers.emplace_back(d_model, num_heads, d_ff);
        }
    }

    MatrixXf forward(const MatrixXf &X);
};
