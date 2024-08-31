#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include "../src/transformer/feed_forward.h"  // Assume this contains your feed_forward_t class
#include <iostream>
#include "test_utils.h"


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


TEST_CASE("Feed-Forward matches PyTorch output", "[feed_forward]") {
    int d_model = 512;
    int d_ff = 2048;
    int batch_size = 32;
    int seq_length = 10;

    // Load weights and biases
    Eigen::MatrixXf W1 = readMatrixFromFile("tests/test_data/feed_forward/w1.txt", d_ff, d_model);
    Eigen::VectorXf b1 = readVectorFromFile("tests/test_data/feed_forward/b1.txt");
    Eigen::MatrixXf W2 = readMatrixFromFile("tests/test_data/feed_forward/w2.txt", d_model, d_ff);
    Eigen::VectorXf b2 = readVectorFromFile("tests/test_data/feed_forward/b2.txt");

    // Create and initialize feed_forward_t
    feed_forward_t ff(d_model, d_ff);
    ff.set_weights(W1, W2, b1, b2);

    // Load input and expected output
    Eigen::MatrixXf input = readMatrixFromFile("tests/test_data/feed_forward/input.txt", batch_size * seq_length, d_model);
    Eigen::MatrixXf expected_output = readMatrixFromFile("tests/test_data/feed_forward/output.txt", batch_size * seq_length, d_model);

    // Run feed-forward
    Eigen::MatrixXf output = ff.forward(input);

    // Compare output with expected output
    REQUIRE(output.rows() == expected_output.rows());
    REQUIRE(output.cols() == expected_output.cols());

    REQUIRE(matrices_approx_equal(output, expected_output));
}