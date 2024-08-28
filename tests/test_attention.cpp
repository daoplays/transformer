#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../src/transformer/multi_head_attention.h"
#include "../src/eigen_config.h"
#include <random>

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
    
    SECTION("Output values are within expected range") {
        int d_k = 64;
        int seq_len = 10;
        attention_t attention(d_k);
        MatrixXf input = generateRandomMatrix(seq_len, d_k);
        
        MatrixXf output = attention.forward(input);
        
        REQUIRE(output.minCoeff() >= -1.0f);
        REQUIRE(output.maxCoeff() <= 1.0f);
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
    
    SECTION("Output values are within expected range") {
        int d_model = 512;
        int num_heads = 8;
        int seq_len = 10;
        multi_head_attention_t mha(d_model, num_heads);
        MatrixXf input = generateRandomMatrix(seq_len, d_model);
        
        MatrixXf output = mha.forward(input);
        
        REQUIRE(output.minCoeff() >= -10.0f);
        REQUIRE(output.maxCoeff() <= 10.0f);
    }
}

TEST_CASE("attention_t and multi_head_attention_t consistency", "[attention][multi_head_attention]") {
    SECTION("Single head multi-head attention should be similar to regular attention") {
        int d_model = 64;
        int num_heads = 1;
        int seq_len = 10;
        
        attention_t attention(d_model);
        multi_head_attention_t mha(d_model, num_heads);
        
        MatrixXf input = generateRandomMatrix(seq_len, d_model);
        
        MatrixXf attention_output = attention.forward(input);
        MatrixXf mha_output = mha.forward(input);
        
        // Check if the outputs are similar (not exact due to the extra projection in MHA)
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                REQUIRE_THAT(mha_output(i, j), Catch::Matchers::WithinAbs(attention_output(i, j), 1.0));
            }
        }
    }
}

TEST_CASE("multi_head_attention_t with different number of heads", "[multi_head_attention]") {
    SECTION("Different number of heads produce different outputs") {
        int d_model = 512;
        int seq_len = 10;
        
        multi_head_attention_t mha1(d_model, 1);
        multi_head_attention_t mha4(d_model, 4);
        multi_head_attention_t mha8(d_model, 8);
        
        MatrixXf input = generateRandomMatrix(seq_len, d_model);
        
        MatrixXf output1 = mha1.forward(input);
        MatrixXf output4 = mha4.forward(input);
        MatrixXf output8 = mha8.forward(input);
        
        REQUIRE_FALSE((output1 - output4).isZero(1e-6));
        REQUIRE_FALSE((output1 - output8).isZero(1e-6));
        REQUIRE_FALSE((output4 - output8).isZero(1e-6));
    }
}

TEST_CASE("attention_t with predefined input", "[attention]") {
    SECTION("Predefined input produces expected output") {
        int d_k = 4;
        int seq_len = 3;
        attention_t attention(d_k);

        // Manually set the weights
        attention.set_weights(
            (MatrixXf(4,4) << 
             0.1, 0.2, -0.1, 0.3,
             0.0, 0.4, 0.1, -0.2,
             0.5, -0.3, 0.2, 0.1,
             -0.2, 0.3, 0.1, 0.0).finished(),  // Query weights
            (MatrixXf(4,4) << 
             0.2, -0.1, 0.3, 0.1,
             0.4, 0.1, -0.2, 0.0,
             -0.3, 0.2, 0.1, 0.5,
             0.3, 0.1, 0.0, -0.2).finished(),  // Key weights
            (MatrixXf(4,4) << 
             0.3, 0.1, 0.2, -0.1,
             0.1, -0.2, 0.0, 0.4,
             0.2, 0.1, 0.5, -0.3,
             0.1, 0.0, -0.2, 0.3).finished()   // Value weights
        );

        MatrixXf input(3, 4);
        input << 1.0, 0.5, -0.3, 0.2,
                 0.7, -0.2, 0.4, -0.1,
                 -0.5, 0.1, 0.8, 0.3;

        MatrixXf output = attention.forward(input);

        // Expected output (calculated manually or with a reference implementation)
        MatrixXf expected_output(3, 4);
        expected_output << 0.29788, -0.03329, 0.18722, 0.04243,
                           0.30396, -0.02406, 0.16547, 0.02443,
                           0.28025, -0.05376, 0.23136, 0.08273;

        REQUIRE(output.rows() == expected_output.rows());
        REQUIRE(output.cols() == expected_output.cols());

        for (int i = 0; i < output.rows(); ++i) {
            for (int j = 0; j < output.cols(); ++j) {
                REQUIRE_THAT(output(i, j), Catch::Matchers::WithinAbs(expected_output(i, j), 1e-5));
            }
        }
    }
}

TEST_CASE("multi_head_attention_t with predefined input", "[multi_head_attention]") {
    SECTION("Predefined input produces expected output") {
        int d_model = 8;
        int num_heads = 2;
        int seq_len = 3;
        multi_head_attention_t mha(d_model, num_heads);

        // Manually set the weights for each attention head and the output projection
        // This is a simplified example; in practice, you'd set weights for all heads
        mha.set_weights({
            {(MatrixXf(4,4) <<  // Head 1
              0.1, 0.2, -0.1, 0.3,
              0.0, 0.4, 0.1, -0.2,
              0.5, -0.3, 0.2, 0.1,
              -0.2, 0.3, 0.1, 0.0).finished(),
             (MatrixXf(4,4) << 
              0.2, -0.1, 0.3, 0.1,
              0.4, 0.1, -0.2, 0.0,
              -0.3, 0.2, 0.1, 0.5,
              0.3, 0.1, 0.0, -0.2).finished(),
             (MatrixXf(4,4) << 
              0.3, 0.1, 0.2, -0.1,
              0.1, -0.2, 0.0, 0.4,
              0.2, 0.1, 0.5, -0.3,
              0.1, 0.0, -0.2, 0.3).finished()},
            {(MatrixXf(4,4) <<  // Head 2
              -0.1, 0.3, 0.2, 0.0,
              0.4, 0.1, -0.2, 0.3,
              0.2, -0.3, 0.1, 0.4,
              0.1, 0.2, 0.0, -0.1).finished(),
             (MatrixXf(4,4) << 
              0.3, 0.0, -0.1, 0.2,
              0.1, 0.4, 0.2, -0.3,
              -0.2, 0.1, 0.3, 0.0,
              0.2, -0.1, 0.0, 0.4).finished(),
             (MatrixXf(4,4) << 
              0.2, -0.1, 0.3, 0.0,
              0.0, 0.3, -0.2, 0.1,
              0.4, 0.1, 0.0, -0.3,
              -0.1, 0.2, 0.4, 0.1).finished()}
        },
        (MatrixXf(8,8) <<  // Output projection
         0.1, -0.1, 0.2, 0.0, 0.3, -0.2, 0.1, 0.0,
         0.0, 0.2, -0.1, 0.3, -0.1, 0.1, 0.0, 0.2,
         0.3, 0.1, 0.0, -0.2, 0.2, 0.0, -0.1, 0.3,
         -0.2, 0.0, 0.3, 0.1, 0.0, 0.3, 0.2, -0.1,
         0.2, -0.3, 0.1, 0.0, 0.1, -0.1, 0.3, 0.0,
         0.1, 0.0, -0.2, 0.3, -0.3, 0.2, 0.0, 0.1,
         -0.1, 0.2, 0.0, 0.1, 0.0, -0.2, 0.1, 0.3,
         0.0, 0.1, 0.3, -0.1, 0.2, 0.1, -0.3, 0.0).finished());

        MatrixXf input(3, 8);
        input << 1.0, 0.5, -0.3, 0.2, 0.7, -0.2, 0.4, -0.1,
                 0.7, -0.2, 0.4, -0.1, 1.0, 0.5, -0.3, 0.2,
                 -0.5, 0.1, 0.8, 0.3, -0.2, 0.6, 0.0, -0.4;

        MatrixXf output = mha.forward(input);

        // Expected output (calculated manually or with a reference implementation)
        MatrixXf expected_output(3, 8);
        expected_output << 0.07560, 0.01823, 0.01417, -0.03108, 0.01650, -0.00595, 0.01269, 0.02446,
                           0.07991, 0.01839, 0.01360, -0.03013, 0.01893, -0.00377, 0.01360, 0.02189,
                           0.06839, 0.01785, 0.01524, -0.03272, 0.01228, -0.00945, 0.01108, 0.02865;

        REQUIRE(output.rows() == expected_output.rows());
        REQUIRE(output.cols() == expected_output.cols());

        for (int i = 0; i < output.rows(); ++i) {
            for (int j = 0; j < output.cols(); ++j) {
                REQUIRE_THAT(output(i, j), Catch::Matchers::WithinAbs(expected_output(i, j), 1e-5));
            }
        }
    }
}