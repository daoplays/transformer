#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <random>
#include "../src/eigen_config.h"
#include "../src/transformer/multi_head_attention.h"
#include "test_utils.h"

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
    Eigen::MatrixXf output = attn.forward(Q, K, V);

    // Load expected output
    Eigen::MatrixXf expected_output = readMatrixFromFile("tests/test_data/attention/output.txt", seq_length, d_model);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output));
}

TEST_CASE("Multi-Head Attention matches PyTorch output", "[multi_head_attention]")
{
    int d_model = 512;
    int num_heads = 8;
    int seq_length = 10;

    // Create multi-head attention layer
    multi_head_attention_t mha(d_model, num_heads);

    // Load weights and biases
    Eigen::MatrixXf in_proj_weight = readMatrixFromFile("tests/test_data/multi_head_attention/mha_in_proj_weight.txt", 3 * d_model, d_model);
    Eigen::VectorXf in_proj_bias = readVectorFromFile("tests/test_data/multi_head_attention/mha_in_proj_bias.txt");
    Eigen::MatrixXf out_proj_weight = readMatrixFromFile("tests/test_data/multi_head_attention/mha_out_proj_weight.txt", d_model, d_model);
    Eigen::VectorXf out_proj_bias = readVectorFromFile("tests/test_data/multi_head_attention/mha_out_proj_bias.txt");

    // Split in_proj_weight and in_proj_bias into q, k, v components
    Eigen::MatrixXf query_weights = in_proj_weight.block(0, 0, d_model, d_model);
    Eigen::MatrixXf key_weights = in_proj_weight.block(d_model, 0, d_model, d_model);
    Eigen::MatrixXf value_weights = in_proj_weight.block(2 * d_model, 0, d_model, d_model);

    Eigen::VectorXf query_bias = in_proj_bias.segment(0, d_model);
    Eigen::VectorXf key_bias = in_proj_bias.segment(d_model, d_model);
    Eigen::VectorXf value_bias = in_proj_bias.segment(2 * d_model, d_model);

    // Set weights and biases
    mha.set_weights(query_weights, key_weights, value_weights, query_bias, key_bias, value_bias, out_proj_weight, out_proj_bias);

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