#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <vector>
#include "../src/eigen_config.h"
#include "../src/transformer/transformer.h"
#include "test_utils.h"
#include "../src/gpt2.h"

TEST_CASE("Transformer Forward Pass", "[transformer]")
{
    int d_model = 768;
    int num_heads = 12;
    int d_ff = 3072;
    const int seq_length = 10;
    const int num_layers = 12;

    gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");


    // Create encoder
    transformer_t transformer(num_layers, d_model, num_heads, d_ff);

    // Load weights and biases
    for (int i = 0; i < num_layers; ++i) {


    // Set weights and biases
    transformer.set_layer_weights(i, gpt_weights.layers[i].attn_c_attn_weight, gpt_weights.layers[i].attn_c_attn_bias,
                gpt_weights.layers[i].attn_c_proj_weight, gpt_weights.layers[i].attn_c_proj_bias,
                gpt_weights.layers[i].ln_1_weight, gpt_weights.layers[i].ln_1_bias,
                gpt_weights.layers[i].mlp_c_fc_weight.transpose(), gpt_weights.layers[i].mlp_c_fc_bias,
                gpt_weights.layers[i].mlp_c_proj_weight.transpose(), gpt_weights.layers[i].mlp_c_proj_bias,
                gpt_weights.layers[i].ln_2_weight, gpt_weights.layers[i].ln_2_bias);
    }

    // Load input
    MatrixXf input = readMatrixFromFile("tests/test_data/transformer/transformer_input.txt", seq_length, d_model);

    // Perform forward pass
    MatrixXf output = transformer.forward(input);

    // Load expected output
    MatrixXf expected_output = readMatrixFromFile("tests/test_data/transformer/transformer_output.txt", seq_length, d_model);

    // Compare outputs
    REQUIRE(matrices_approx_equal(output, expected_output, 1e-1));
}