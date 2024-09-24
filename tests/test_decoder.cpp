#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fstream>
#include <vector>
#include "../src/eigen_config.h"
#include "../src/gpt2.h"
#include "../src/transformer/decoder_layer.h"
#include "test_utils.h"

TEST_CASE("Decoder Layer Forward Pass", "[encoder_layer]")
{
    int d_model = 768;
    int num_heads = 12;
    int seq_length = 10;
    int d_ff = 3072;

    gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");

    // Create encoder layer
    decoder_layer_t decoder_layer(d_model, num_heads, d_ff);

    // Set weights and biases
    decoder_layer.set_weights(gpt_weights.layers[0].attn_c_attn_weight, gpt_weights.layers[0].attn_c_attn_bias,
                              gpt_weights.layers[0].attn_c_proj_weight, gpt_weights.layers[0].attn_c_proj_bias, gpt_weights.layers[0].ln_1_weight,
                              gpt_weights.layers[0].ln_1_bias, gpt_weights.layers[0].mlp_c_fc_weight.transpose(), gpt_weights.layers[0].mlp_c_fc_bias,
                              gpt_weights.layers[0].mlp_c_proj_weight.transpose(), gpt_weights.layers[0].mlp_c_proj_bias,
                              gpt_weights.layers[0].ln_2_weight, gpt_weights.layers[0].ln_2_bias);

    // Load input
    MatrixXf input = readMatrixFromFile("tests/test_data/encoder/encoder_input.txt", seq_length, d_model);

    // Perform forward pass
    MatrixXf output = decoder_layer.forward(input);

    MatrixXf pytorch_final_output = readMatrixFromFile("tests/test_data/encoder/encoder_output.txt", seq_length, d_model);

    REQUIRE(output.rows() == pytorch_final_output.rows());
    REQUIRE(output.cols() == pytorch_final_output.cols());

    REQUIRE(matrices_approx_equal(output, pytorch_final_output, 1e-1));
}