#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include "../src/eigen_config.h"
#include "../src/transformer/multi_head_attention.h"
#include "test_utils.h"
#include "../src/gpt2.h"

TEST_CASE("Attention Forward Pass", "[attention]")
{
    int d_model = 512;
    int seq_length = 10;

    // Create attention layer
    attention_t attn;

    // Load weights and biases
    Eigen::MatrixXf query_weights = readMatrixFromFile("tests/test_data/attention/query_weights.txt", d_model, d_model);
    Eigen::VectorXf query_bias = readVectorFromFile("tests/test_data/attention/query_bias.txt");
    Eigen::MatrixXf key_weights = readMatrixFromFile("tests/test_data/attention/key_weights.txt", d_model, d_model);
    Eigen::VectorXf key_bias = readVectorFromFile("tests/test_data/attention/key_bias.txt");
    Eigen::MatrixXf value_weights = readMatrixFromFile("tests/test_data/attention/value_weights.txt", d_model, d_model);
    Eigen::VectorXf value_bias = readVectorFromFile("tests/test_data/attention/value_bias.txt");

    // Load input
    Eigen::MatrixXf input = readMatrixFromFile("tests/test_data/attention/input.txt", seq_length, d_model);

    // Construct Q, K, V matrices
    Eigen::MatrixXf Q = (input * query_weights.transpose()).rowwise() + query_bias.transpose();
    Eigen::MatrixXf K = (input * key_weights.transpose()).rowwise() + key_bias.transpose();
    Eigen::MatrixXf V = (input * value_weights.transpose()).rowwise() + value_bias.transpose();

    // Perform forward pass with Q, K, V
    Eigen::MatrixXf output = attn.forward(Q, K, V, false);

    // Load expected output
    Eigen::MatrixXf expected_output = readMatrixFromFile("tests/test_data/attention/output.txt", seq_length, d_model);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output));
}

TEST_CASE("Multi-Head Attention matches PyTorch output", "[multi_head_attention]")
{
    int d_model = 768;
    int num_heads = 12;
    int seq_length = 10;

    gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");


    // Create multi-head attention layer
    multi_head_attention_t mha(d_model, num_heads);

    // Set weights and biases
    mha.set_weights2(gpt_weights.layers[0].attn_c_attn_weight, gpt_weights.layers[0].attn_c_attn_bias,
                    gpt_weights.layers[0].attn_c_proj_weight, gpt_weights.layers[0].attn_c_proj_bias);

    // Load input
    Eigen::MatrixXf input = readMatrixFromFile("tests/test_data/multi_head_attention/mha_input.txt", seq_length, d_model);

    // Perform forward pass
    Eigen::MatrixXf output = mha.forward(input);

    // Load expected output
    Eigen::MatrixXf expected_output = readMatrixFromFile("tests/test_data/multi_head_attention/mha_output.txt", seq_length, d_model);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output, 1e-4));
}