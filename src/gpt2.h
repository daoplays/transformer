#pragma once
#include "eigen_config.h"
#include "load_h5.h"
#include "tokenizer.h"
#include "transformer/norm_layer.h"
#include "transformer/transformer.h"

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

class gpt2_t {
private:

    static constexpr int max_seq_len = 1024;
    static constexpr int d_model = 768;
    static constexpr int num_heads = 12;
    static constexpr int d_ff = 3072;
    static constexpr int num_layers = 12;
    static constexpr int vocab_size = 50257;

    transformer_t transformer;
    tokenizer_t tokenizer;
    norm_layer_t final_norm_layer;
    gpt2_weights_t weights;

public:

    gpt2_t()
        : transformer(num_layers, d_model, num_heads, d_ff), tokenizer("gpt2/vocab.json", "gpt2/merges.txt"), final_norm_layer(d_model, 1e-5) {

          };

    void init();

    Eigen::MatrixXf forward(string_t input_string);

    gpt2_weights_t get_weights() { return weights; }

    string_t get_next_max_like_token(MatrixXf& logits);
};

gpt2_weights_t load_gpt2_weights(const string_t& h5_file_path);
