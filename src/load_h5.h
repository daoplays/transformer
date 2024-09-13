#pragma once
#include <H5Cpp.h>
#include "eigen_config.h"
#include "types/basic_types.h"

struct GPT2Layer {
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

struct GPT2_Weights {
    Eigen::MatrixXf token_embedding;
    Eigen::MatrixXf position_embedding;

    std::vector<GPT2Layer> layers;
    Eigen::VectorXf ln_f_weight;  // Final layer normalization weight
    Eigen::VectorXf ln_f_bias;    // Final layer normalization bias
};

Eigen::MatrixXf read_matrix(const H5::H5File& file, const string_t& dataset_name);
GPT2_Weights load_gpt2_weights(const string_t& h5_file_path);
