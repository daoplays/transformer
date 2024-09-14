#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include "../src/transformer/norm_layer.h"  // Include your LayerNorm class definition here
#include "test_utils.h"
#include "../src/load_h5.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("LayerNorm forward pass", "[layernorm]")
{

        int d_model = 768;
        int seq_length = 10;

        gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");

        norm_layer_t layer_norm(d_model, 1e-5);
        layer_norm.setGammaBeta(gpt_weights.layers[0].ln_1_weight, gpt_weights.layers[0].ln_1_bias);

        MatrixXf input = readMatrixFromFile("tests/test_data/layer_norm/layer_norm_input.txt", seq_length, d_model);

        MatrixXf norm_output = layer_norm.forward(input);

        MatrixXf expected_output = readMatrixFromFile("tests/test_data/layer_norm/layer_norm_output.txt", seq_length, d_model);


        REQUIRE(norm_output.rows() == expected_output.rows());
        REQUIRE(norm_output.cols() == expected_output.cols());
        REQUIRE(matrices_approx_equal(norm_output, expected_output, 1e-4));

}