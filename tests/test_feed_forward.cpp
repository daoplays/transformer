#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include "../src/transformer/feed_forward.h"  // Assume this contains your feed_forward_t class
#include <iostream>
#include "test_utils.h"
#include "../src/load_h5.h"

// Custom approx_equal function
bool approx_equal(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) < epsilon;
}

TEST_CASE("ReLU function", "[relu]") {
    feed_forward_t ff(10, 20);  // Dimensions don't matter for this test

    SECTION("ReLU of positive number") {
        REQUIRE(relu(5.0f) == 5.0f);
    }

    SECTION("ReLU of negative number") {
        REQUIRE(relu(-3.0f) == 0.0f);
    }

    SECTION("ReLU of zero") {
        REQUIRE(relu(0.0f) == 0.0f);
    }
}

TEST_CASE("Apply ReLU to matrix", "[apply_relu]") {
    feed_forward_t ff(10, 20);  // Dimensions don't matter for this test

    SECTION("Apply ReLU to matrix with positive and negative values") {
        Eigen::MatrixXf input(2, 3);
        input << 1.0f, -2.0f, 3.0f,
                 -4.0f, 5.0f, 0.0f;

        Eigen::MatrixXf expected(2, 3);
        expected << 1.0f, 0.0f, 3.0f,
                    0.0f, 5.0f, 0.0f;

        Eigen::MatrixXf result = apply_relu(input);
        REQUIRE(matrices_approx_equal(result, expected));
    }
}



TEST_CASE("GELU function", "[gelu]") {

    SECTION("GELU of positive number") {
        REQUIRE(approx_equal(gelu(2.0f), 1.9545977116f));
    }

    SECTION("GELU of negative number") {
        REQUIRE(approx_equal(gelu(-2.0f), -0.0454023f));
    }

    SECTION("GELU of zero") {
        REQUIRE(approx_equal(gelu(0.0f), 0.0f));
    }

    SECTION("GELU of large positive number") {
        REQUIRE(approx_equal(gelu(10.0f), 10.0f));
    }

    SECTION("GELU of large negative number") {
        REQUIRE(approx_equal(gelu(-10.0f), 0.0f));
    }
}

TEST_CASE("Feed-Forward matches PyTorch output", "[feed_forward]") {

    int d_model = 768;
    int num_heads = 12;
    int d_ff = 3072;
    int seq_length = 10;

    
    gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");

    // Create and initialize feed_forward_t
    feed_forward_t ff(d_model, d_ff);
    ff.set_weights(gpt_weights.layers[0].mlp_c_fc_weight.transpose(), gpt_weights.layers[0].mlp_c_proj_weight.transpose(),
                    gpt_weights.layers[0].mlp_c_fc_bias, gpt_weights.layers[0].mlp_c_proj_bias);

    MatrixXf input = readMatrixFromFile("tests/test_data/feed_forward/ff_input.txt", seq_length, d_model);
    MatrixXf expected_output = readMatrixFromFile("tests/test_data/feed_forward/ff_output.txt", seq_length, d_model);

    

    // Run feed-forward
    Eigen::MatrixXf output = ff.forward(input);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output, 1e-4));
}