#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <vector>
#include "../src/eigen_config.h"
#include "../src/transformer/encoder_layer.h"
#include "test_utils.h"

TEST_CASE("Encoder Layer Forward Pass", "[encoder_layer]")
{
    int d_model = 512;
    int nhead = 8;
    int dim_feedforward = 2048;
    int seq_length = 10;

    // Create encoder layer
    encoder_layer_t encoder_layer(d_model, nhead, dim_feedforward);

    // Load weights and biases
    MatrixXf self_attn_in_proj_weight = readMatrixFromFile("tests/test_data/encoder/self_attn_in_proj_weight.txt", 3 * d_model, d_model);
    VectorXf self_attn_in_proj_bias = readVectorFromFile("tests/test_data/encoder/self_attn_in_proj_bias.txt");
    MatrixXf self_attn_out_proj_weight = readMatrixFromFile("tests/test_data/encoder/self_attn_out_proj_weight.txt", d_model, d_model);
    VectorXf self_attn_out_proj_bias = readVectorFromFile("tests/test_data/encoder/self_attn_out_proj_bias.txt");

    VectorXf norm1_gamma = readVectorFromFile("tests/test_data/encoder/norm1_weight.txt");  // PyTorch's "weight" is gamma
    VectorXf norm1_beta = readVectorFromFile("tests/test_data/encoder/norm1_bias.txt");     // PyTorch's "bias" is beta
    VectorXf norm2_gamma = readVectorFromFile("tests/test_data/encoder/norm2_weight.txt");
    VectorXf norm2_beta = readVectorFromFile("tests/test_data/encoder/norm2_bias.txt");

    MatrixXf ff_linear1_weight = readMatrixFromFile("tests/test_data/encoder/ff_linear1_weight.txt", dim_feedforward, d_model);
    VectorXf ff_linear1_bias = readVectorFromFile("tests/test_data/encoder/ff_linear1_bias.txt");
    MatrixXf ff_linear2_weight = readMatrixFromFile("tests/test_data/encoder/ff_linear2_weight.txt", d_model, dim_feedforward);
    VectorXf ff_linear2_bias = readVectorFromFile("tests/test_data/encoder/ff_linear2_bias.txt");

    // Set weights and biases
    encoder_layer.set_weights(self_attn_in_proj_weight.block(0, 0, d_model, d_model), self_attn_in_proj_weight.block(d_model, 0, d_model, d_model),
                              self_attn_in_proj_weight.block(2 * d_model, 0, d_model, d_model), self_attn_in_proj_bias.segment(0, d_model),
                              self_attn_in_proj_bias.segment(d_model, d_model), self_attn_in_proj_bias.segment(2 * d_model, d_model),
                              self_attn_out_proj_weight, self_attn_out_proj_bias, norm1_gamma, norm1_beta, ff_linear1_weight, ff_linear1_bias,
                              ff_linear2_weight, ff_linear2_bias, norm2_gamma, norm2_beta);

    // Load input
    MatrixXf input = readMatrixFromFile("tests/test_data/encoder/encoder_input.txt", seq_length, d_model);

    // Perform forward pass
    MatrixXf output = encoder_layer.forward(input);

    MatrixXf pytorch_final_output = readMatrixFromFile("tests/test_data/encoder/final_output.txt", seq_length, d_model);

    REQUIRE(output.rows() == pytorch_final_output.rows());
    REQUIRE(output.cols() == pytorch_final_output.cols());

    REQUIRE(matrices_approx_equal(output, pytorch_final_output, 1e-4));
}