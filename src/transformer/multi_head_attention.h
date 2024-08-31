#pragma once
#include "attention.h" // Include the file containing the attention_t class
#include "../utils.h"
#include <vector>
#include <cassert>

class multi_head_attention_t {
private:
    int d_model, num_heads, d_k;
    std::vector<attention_t> attention_heads;
    MatrixXf output_projection;

public:
    multi_head_attention_t(int d_model, int num_heads) : d_model(d_model), num_heads(num_heads) {

        if (d_model % num_heads != 0){
            die("d_model must be a multiple of num_heads");
        }
        
        d_k = d_model / num_heads;
        
        // Create multiple attention heads
        for (int i = 0; i < num_heads; ++i) {
            attention_heads.emplace_back(d_k);
        }

        // Initialize output projection
        allocate_and_initialize(output_projection, d_model, d_model);
    }

    MatrixXf forward(const MatrixXf &X);

    void set_weights(const std::vector<std::array<MatrixXf, 3>>& head_weights,const std::vector<std::array<MatrixXf, 3>>& head_biases, const MatrixXf& out_proj) {
        for (size_t i = 0; i < attention_heads.size(); ++i) {
            attention_heads[i].set_weights(head_weights[i][0], head_weights[i][1], head_weights[i][2], head_biases[i][0], head_biases[i][1], head_biases[i][2]);
        }
        output_projection = out_proj;
    }
};