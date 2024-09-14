#pragma once
#include "eigen_config.h"

struct gpt2_layer_t {
    // Attention weights
    Eigen::MatrixXf attn_c_attn_weight;
    Eigen::VectorXf attn_c_attn_bias;
    Eigen::MatrixXf attn_c_proj_weight;
    Eigen::VectorXf attn_c_proj_bias;

    // MLP weights
    Eigen::MatrixXf mlp_c_fc_weight;
    Eigen::VectorXf mlp_c_fc_bias;
    Eigen::MatrixXf mlp_c_proj_weight;
    Eigen::VectorXf mlp_c_proj_bias;

    // Layer normalization weights
    Eigen::VectorXf ln_1_weight;
    Eigen::VectorXf ln_1_bias;
    Eigen::VectorXf ln_2_weight;
    Eigen::VectorXf ln_2_bias;
};

struct gpt2_weights_t {
    Eigen::MatrixXf token_embedding;
    Eigen::MatrixXf position_embedding;

    std::vector<gpt2_layer_t> layers;
    Eigen::VectorXf ln_f_weight;  
    Eigen::VectorXf ln_f_bias;   
};