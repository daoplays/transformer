#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../src/transformer/multi_head_attention.h"
#include "../src/eigen_config.h"
#include <random>
#include "test_utils.h"



// Helper function to generate random matrix
MatrixXf generateRandomMatrix(int rows, int cols) {
    return MatrixXf::Random(rows, cols);
}

TEST_CASE("attention_t basic functionality", "[attention]") {
    SECTION("Output shape is correct") {
        int d_k = 64;
        int seq_len = 10;
        attention_t attention(d_k);
        MatrixXf input = generateRandomMatrix(seq_len, d_k);
        
        MatrixXf output = attention.forward(input);
        
        REQUIRE(output.rows() == seq_len);
        REQUIRE(output.cols() == d_k);
    }
}

TEST_CASE("multi_head_attention_t basic functionality", "[multi_head_attention]") {
    SECTION("Output shape is correct") {
        int d_model = 512;
        int num_heads = 8;
        int seq_len = 10;
        multi_head_attention_t mha(d_model, num_heads);
        MatrixXf input = generateRandomMatrix(seq_len, d_model);
        
        MatrixXf output = mha.forward(input);
        
        REQUIRE(output.rows() == seq_len);
        REQUIRE(output.cols() == d_model);
    }
}

TEST_CASE("attention_t deterministic test", "[attention]") {
    SECTION("Predetermined input produces expected output") {
        int d_k = 4;
        attention_t attention(d_k);

        // Set predetermined weights
        MatrixXf q_weights(4, 4);
        q_weights << 0.1, 0.2, -0.1, 0.3,
                     0.0, 0.4, 0.1, -0.2,
                     0.5, -0.3, 0.2, 0.1,
                     -0.2, 0.3, 0.1, 0.0;

        MatrixXf k_weights(4, 4);
        k_weights << 0.2, -0.1, 0.3, 0.1,
                     0.4, 0.1, -0.2, 0.0,
                     -0.3, 0.2, 0.1, 0.5,
                     0.3, 0.1, 0.0, -0.2;

        MatrixXf v_weights(4, 4);
        v_weights << 0.3, 0.1, 0.2, -0.1,
                     0.1, -0.2, 0.0, 0.4,
                     0.2, 0.1, 0.5, -0.3,
                     0.1, 0.0, -0.2, 0.3;

        attention.set_weights(q_weights, k_weights, v_weights);

        // Predetermined input
        MatrixXf input(3, 4);
        input << 1.0, 0.5, -0.3, 0.2,
                 0.7, -0.2, 0.4, -0.1,
                 -0.5, 0.1, 0.8, 0.3;

        MatrixXf output = attention.forward(input);
        
        // Expected output (updated based on PyTorch results with higher precision)
        MatrixXf expected_output(3, 4);
        expected_output << 0.1989499961, 0.0439635770, 0.2091713054, -0.0442822813,
                           0.2094571942, 0.0425704424, 0.2002728360, -0.0323901394,
                           0.2149729638, 0.0417010399, 0.1953565505, -0.0257543842;

        // Check if the output matches the expected output
        REQUIRE(output.rows() == expected_output.rows());
        REQUIRE(output.cols() == expected_output.cols());

        REQUIRE(matrices_approx_equal(output, expected_output));

    }
}


TEST_CASE("multi_head_attention_t deterministic test", "[multi_head_attention]") {
    SECTION("Predetermined input produces expected output") {
        int d_model = 8;
        int num_heads = 2;
        multi_head_attention_t mha(d_model, num_heads);

        // Prepare weights for each head
        std::vector<std::array<MatrixXf, 3>> head_weights(2);

        // Head 1 weights
        head_weights[0][0] = (MatrixXf(4, 4) <<  // Query weights
            0.1, 0.2, -0.1, 0.3,
            0.0, 0.4, 0.1, -0.2,
            0.5, -0.3, 0.2, 0.1,
            -0.2, 0.3, 0.1, 0.0).finished();

        head_weights[0][1] = (MatrixXf(4, 4) <<  // Key weights
            0.2, -0.1, 0.3, 0.1,
            0.4, 0.1, -0.2, 0.0,
            -0.3, 0.2, 0.1, 0.5,
            0.3, 0.1, 0.0, -0.2).finished();

        head_weights[0][2] = (MatrixXf(4, 4) <<  // Value weights
            0.3, 0.1, 0.2, -0.1,
            0.1, -0.2, 0.0, 0.4,
            0.2, 0.1, 0.5, -0.3,
            0.1, 0.0, -0.2, 0.3).finished();

        // Head 2 weights
        head_weights[1][0] = (MatrixXf(4, 4) <<  // Query weights
            -0.1, 0.3, 0.2, 0.0,
            0.4, 0.1, -0.2, 0.3,
            0.2, -0.3, 0.1, 0.4,
            0.1, 0.2, 0.0, -0.1).finished();

        head_weights[1][1] = (MatrixXf(4, 4) <<  // Key weights
            0.3, 0.0, -0.1, 0.2,
            0.1, 0.4, 0.2, -0.3,
            -0.2, 0.1, 0.3, 0.0,
            0.2, -0.1, 0.0, 0.4).finished();

        head_weights[1][2] = (MatrixXf(4, 4) <<  // Value weights
            0.2, -0.1, 0.3, 0.0,
            0.0, 0.3, -0.2, 0.1,
            0.4, 0.1, 0.0, -0.3,
            -0.1, 0.2, 0.4, 0.1).finished();

        // Output projection weights
        MatrixXf out_proj(8, 8);
        out_proj << 0.1, -0.1, 0.2, 0.0, 0.3, -0.2, 0.1, 0.0,
                    0.0, 0.2, -0.1, 0.3, -0.1, 0.1, 0.0, 0.2,
                    0.3, 0.1, 0.0, -0.2, 0.2, 0.0, -0.1, 0.3,
                    -0.2, 0.0, 0.3, 0.1, 0.0, 0.3, 0.2, -0.1,
                    0.2, -0.3, 0.1, 0.0, 0.1, -0.1, 0.3, 0.0,
                    0.1, 0.0, -0.2, 0.3, -0.3, 0.2, 0.0, 0.1,
                    -0.1, 0.2, 0.0, 0.1, 0.0, -0.2, 0.1, 0.3,
                    0.0, 0.1, 0.3, -0.1, 0.2, 0.1, -0.3, 0.0;

        mha.set_weights(head_weights, out_proj);

        // Predetermined input
        MatrixXf input(3, 8);
        input << 1.0, 0.5, -0.3, 0.2, 0.7, -0.2, 0.4, -0.1,
                 0.7, -0.2, 0.4, -0.1, 1.0, 0.5, -0.3, 0.2,
                 -0.5, 0.1, 0.8, 0.3, -0.2, 0.6, 0.0, -0.4;

        MatrixXf output = mha.forward(input);

        // Expected output (based on PyTorch results with higher precision)
        MatrixXf expected_output(3, 8);
        expected_output << 0.1137356885, -0.0167729645, 0.0325491083, -0.0222125931,
                           0.1043725842, -0.0652467459, 0.0294510647, 0.0927920048,
                           0.1095431497, -0.0180263335, 0.0384253444, -0.0194680621,
                           0.1052492433, -0.0616933519, 0.0308990083, 0.0878364291,
                           0.1060610018, -0.0172885898, 0.0431026609, -0.0175334489,
                           0.1081724170, -0.0670506592, 0.0378613590, 0.0912913118;

        // Check if the output matches the expected output
        REQUIRE(matrices_approx_equal(output, expected_output));

        // If you need to debug, you can print the matrices:
        if (!matrices_approx_equal(output, expected_output)) {
            std::cout << "Output:\n" << output << std::endl;
            std::cout << "Expected:\n" << expected_output << std::endl;
        }
    }
}