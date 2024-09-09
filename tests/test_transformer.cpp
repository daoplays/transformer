#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <vector>
#include "../src/eigen_config.h"
#include "../src/transformer/transformer.h"
#include "test_utils.h"

TEST_CASE("Transformer Forward Pass", "[transformer]")
{
    const int d_model = 512;
    const int nhead = 8;
    const int dim_feedforward = 2048;
    const int seq_length = 10;
    const int num_layers = 3;

    // Create encoder
    transformer_t transformer(num_layers, d_model, nhead, dim_feedforward);

    // Load weights and biases
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::string layer_prefix = "tests/test_data/transformer/layer_" + std::to_string(layer_idx) + "_";
        MatrixXf self_attn_in_proj_weight = readMatrixFromFile(layer_prefix + "self_attn_in_proj_weight.txt", 3 * d_model, d_model);
        VectorXf self_attn_in_proj_bias = readVectorFromFile(layer_prefix + "self_attn_in_proj_bias.txt");
        MatrixXf self_attn_out_proj_weight = readMatrixFromFile(layer_prefix + "self_attn_out_proj_weight.txt", d_model, d_model);
        VectorXf self_attn_out_proj_bias = readVectorFromFile(layer_prefix + "self_attn_out_proj_bias.txt");

        VectorXf norm1_gamma = readVectorFromFile(layer_prefix + "norm1_weight.txt");
        VectorXf norm1_beta = readVectorFromFile(layer_prefix + "norm1_bias.txt");
        VectorXf norm2_gamma = readVectorFromFile(layer_prefix + "norm2_weight.txt");
        VectorXf norm2_beta = readVectorFromFile(layer_prefix + "norm2_bias.txt");

        MatrixXf ff_linear1_weight = readMatrixFromFile(layer_prefix + "ff_linear1_weight.txt", dim_feedforward, d_model);
        VectorXf ff_linear1_bias = readVectorFromFile(layer_prefix + "ff_linear1_bias.txt");
        MatrixXf ff_linear2_weight = readMatrixFromFile(layer_prefix + "ff_linear2_weight.txt", d_model, dim_feedforward);
        VectorXf ff_linear2_bias = readVectorFromFile(layer_prefix + "ff_linear2_bias.txt");

        // Set weights and biases
        transformer.set_layer_weights(layer_idx, self_attn_in_proj_weight.block(0, 0, d_model, d_model),
                                      self_attn_in_proj_weight.block(d_model, 0, d_model, d_model),
                                      self_attn_in_proj_weight.block(2 * d_model, 0, d_model, d_model), self_attn_in_proj_bias.segment(0, d_model),
                                      self_attn_in_proj_bias.segment(d_model, d_model), self_attn_in_proj_bias.segment(2 * d_model, d_model),
                                      self_attn_out_proj_weight, self_attn_out_proj_bias, norm1_gamma, norm1_beta, ff_linear1_weight, ff_linear1_bias,
                                      ff_linear2_weight, ff_linear2_bias, norm2_gamma, norm2_beta);
    }

    // Load input
    MatrixXf input = readMatrixFromFile("tests/test_data/transformer/transformer_input.txt", seq_length, d_model);

    // Perform forward pass
    MatrixXf output = transformer.forward(input);

    // Load expected output
    MatrixXf expected_output = readMatrixFromFile("tests/test_data/transformer/transformer_output.txt", seq_length, d_model);

    // Compare outputs
    REQUIRE(matrices_approx_equal(output, expected_output, 1e-4));
}