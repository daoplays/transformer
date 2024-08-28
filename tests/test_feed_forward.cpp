#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include "../src/transformer/feed_forward.h"  // Assume this contains your feed_forward_t class
#include <iostream>

// Helper function to check if two Eigen matrices are approximately equal
bool matrices_approx_equal(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float epsilon = 1e-5f) {
    return (m1 - m2).cwiseAbs().maxCoeff() < epsilon;
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

TEST_CASE("Feed Forward Network", "[feed_forward]") {
    int d_model = 4;
    int d_ff = 8;
    feed_forward_t ff(d_model, d_ff);

    Eigen::MatrixXf W1(d_model, d_ff);
    W1 << 0.1f,  0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,
         -0.1f, -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f,
          0.2f,  0.3f,  0.4f,  0.5f,  0.6f,  0.7f,  0.8f,  0.9f,
         -0.2f, -0.3f, -0.4f, -0.5f, -0.6f, -0.7f, -0.8f, -0.9f;

    Eigen::MatrixXf W2(d_ff, d_model);
    W2 << 0.1f,  0.2f,  0.3f,  0.4f,
         0.2f,  0.3f,  0.4f,  0.5f,
         0.3f,  0.4f,  0.5f,  0.6f,
         0.4f,  0.5f,  0.6f,  0.7f,
        -0.1f, -0.2f, -0.3f, -0.4f,
        -0.2f, -0.3f, -0.4f, -0.5f,
        -0.3f, -0.4f, -0.5f, -0.6f,
        -0.4f, -0.5f, -0.6f, -0.7f;

    ff.set_weights(W1, W2);

    SECTION("Forward pass with mixed positive and negative inputs") {
        Eigen::MatrixXf input(2, d_model);
        input << 1.0f, -2.0f,  3.0f, -4.0f,
                -5.0f,  6.0f, -7.0f,  8.0f;

        Eigen::MatrixXf result = ff.forward(input);

        std::cout << result << std::endl;

        // Hardcoded expected output
        Eigen::MatrixXf expected(2, d_model);
        expected << -4.0f, -5.6f, -7.2f, -8.8f,
                     0.0f,  0.0f,  0.0f,  0.0f;

        REQUIRE(matrices_approx_equal(result, expected));
    }
}