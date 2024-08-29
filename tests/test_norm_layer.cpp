#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include "../src/transformer/norm_layer.h" // Include your LayerNorm class definition here
#include "test_utils.h"
#include <iostream>

using Catch::Matchers::WithinAbs;

TEST_CASE("LayerNorm forward pass", "[layernorm]") {
    SECTION("2x3 input, epsilon = 1e-5") {
        norm_layer_t layer_norm(3, 1e-5);
        
        // Set predetermined values for gamma and beta
        Eigen::Vector3f gamma(1.0f, 0.5f, 2.0f);
        Eigen::Vector3f beta(0.0f, 1.0f, -1.0f);
        std::cout << "made gamma" << gamma << std::endl;
        layer_norm.setGammaBeta(gamma, beta);
        Eigen::MatrixXf input(2, 3);
        input << 1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f;

        Eigen::MatrixXf output = layer_norm.forward(input);
        std::cout << output << std::endl;

        // Expected output calculated manually
        Eigen::MatrixXf expected_output(2, 3);
        expected_output << -1.2247356859, 1.0000f, 1.44946,
                           -1.2247356859, 1.0000f, 1.44946;


        REQUIRE(output.rows() == 2);
        REQUIRE(output.cols() == 3);

        REQUIRE(matrices_approx_equal(output, expected_output));

    }

    SECTION("3x2 input, epsilon = 1e-5") {
        norm_layer_t layer_norm(2, 1e-5);
        
        // Set predetermined values for gamma and beta
        layer_norm.setGammaBeta(Eigen::Vector2f(1.5f, 0.5f), Eigen::Vector2f(-1.0f, 1.0f));

        Eigen::MatrixXf input(3, 2);
        input << 1.0f, 4.0f,
                 2.0f, 5.0f,
                 3.0f, 6.0f;

        Eigen::MatrixXf output = layer_norm.forward(input);
        std::cout << output << std::endl;

        // Expected output calculated manually
        Eigen::MatrixXf expected_output(3, 2);
        expected_output << -2.5000f, 1.5000f,
                           -2.5000f, 1.5000f,
                           -2.5000f, 1.5000f;

        REQUIRE(output.rows() == 3);
        REQUIRE(output.cols() == 2);
        REQUIRE(matrices_approx_equal(output, expected_output, 1e-4));

       
    }
}