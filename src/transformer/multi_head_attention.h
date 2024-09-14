#pragma once
#include <cassert>
#include <vector>
#include "../utils.h"
#include "attention.h"  // Include the file containing the attention_t class

class multi_head_attention_t {
private:

    int d_model, num_heads, d_k;
    attention_t attention_head;
    MatrixXf query_weights, key_weights, value_weights;
    VectorXf query_bias, key_bias, value_bias;
    MatrixXf output_projection;
    VectorXf output_bias;
    float scale_factor;

    MatrixXf qkv_weights;
    VectorXf qkv_bias;

public:

    multi_head_attention_t(int d_model, int num_heads) : d_model(d_model), num_heads(num_heads)
    {

        if (d_model % num_heads != 0) {
            die("d_model must be a multiple of num_heads");
        }

        d_k = d_model / num_heads;

        // Initialize matrices
        allocate_and_initialize(query_weights, d_model, d_model);
        allocate_and_initialize(key_weights, d_model, d_model);
        allocate_and_initialize(value_weights, d_model, d_model);

        allocate_and_initialize(output_projection, d_model, d_model);

        query_bias = Eigen::VectorXf::Zero(d_model);
        key_bias = Eigen::VectorXf::Zero(d_model);
        value_bias = Eigen::VectorXf::Zero(d_model);

        output_bias = Eigen::VectorXf::Zero(d_model);
        scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k));

        allocate_and_initialize(qkv_weights, d_model, 3 * d_model);
        qkv_bias = Eigen::VectorXf::Zero(3 * d_model);

    }

    MatrixXf forward(const MatrixXf& X);

    void set_weights(const MatrixXf& q_weights, const MatrixXf& k_weights, const MatrixXf& v_weights, const VectorXf& q_bias, const VectorXf& k_bias,
                     const VectorXf& v_bias, const MatrixXf& out_proj, const VectorXf& out_bias)
    {
        query_weights = q_weights;
        key_weights = k_weights;
        value_weights = v_weights;
        query_bias = q_bias;
        key_bias = k_bias;
        value_bias = v_bias;
        output_projection = out_proj;
        output_bias = out_bias;

        // Sanity checks
        assert(query_weights.rows() == d_model && query_weights.cols() == d_model);
        assert(key_weights.rows() == d_model && key_weights.cols() == d_model);
        assert(value_weights.rows() == d_model && value_weights.cols() == d_model);
        assert(query_bias.size() == d_model);
        assert(key_bias.size() == d_model);
        assert(value_bias.size() == d_model);
        assert(output_projection.rows() == d_model && output_projection.cols() == d_model);
        assert(output_bias.size() == d_model);
    }

    void set_weights2(const MatrixXf& _qkv_weights,  const VectorXf& _qkv_bias, const MatrixXf& out_proj, const VectorXf& out_bias)
    {
        qkv_weights = _qkv_weights;
        qkv_bias = _qkv_bias;
       
        output_projection = out_proj;
        output_bias = out_bias;

        // Sanity checks
       
    }
};