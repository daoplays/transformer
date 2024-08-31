#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../src/transformer/multi_head_attention.h"
#include "../src/eigen_config.h"
#include <random>
#include "test_utils.h"



TEST_CASE("Attention Forward Pass", "[attention]") {
    int d_model = 512;
    int seq_length = 10;

    // Create attention layer
    attention_t attn(d_model);

    // Load weights and biases
    Eigen::MatrixXf query_weights = readMatrixFromFile("tests/test_data/attention/query_weights.txt", d_model, d_model);
    Eigen::VectorXf query_bias = readVectorFromFile("tests/test_data/attention/query_bias.txt");
    Eigen::MatrixXf key_weights = readMatrixFromFile("tests/test_data/attention/key_weights.txt", d_model, d_model);
    Eigen::VectorXf key_bias = readVectorFromFile("tests/test_data/attention/key_bias.txt");
    Eigen::MatrixXf value_weights = readMatrixFromFile("tests/test_data/attention/value_weights.txt", d_model, d_model);
    Eigen::VectorXf value_bias = readVectorFromFile("tests/test_data/attention/value_bias.txt");

    // Set weights and biases
    attn.set_weights(query_weights, key_weights, value_weights, query_bias, key_bias, value_bias);

    // Load input
    Eigen::MatrixXf input = readMatrixFromFile("tests/test_data/attention/input.txt", seq_length, d_model);

    // Perform forward pass
    Eigen::MatrixXf output = attn.forward(input);

    // Load expected output
    Eigen::MatrixXf expected_output = readMatrixFromFile("tests/test_data/attention/output.txt", seq_length, d_model);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output));


}