#include <catch2/catch_all.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../src/gpt2.h"
#include "../src/tokenizer.h"
#include "../src/transformer/decoder_layer.h"
#include "../src/transformer/multi_head_attention.h"
#include "../src/transformer/norm_layer.h"  // Include your LayerNorm class definition here
#include "../src/utils.h"
#include "test_utils.h"

TEST_CASE("Vocabulary loader correctly loads GPT-2 vocabulary", "[gpt2]")
{
    // Initialize tokenizer_t with vocabulary and merges files
    tokenizer_t tokenizer("gpt2/vocab.json", "gpt2/merges.txt");

    // Example text to tokenize
    std::string text = "GPT2 is a model developed by OpenAI";

    gpt2_t gpt2;
    gpt2.init();

    gpt2_weights_t gpt_weights = gpt2.get_weights();

    REQUIRE(gpt_weights.token_embedding.rows() == 50257);
    REQUIRE(gpt_weights.token_embedding.cols() == 768);
    REQUIRE(gpt_weights.position_embedding.rows() == 1024);
    REQUIRE(gpt_weights.position_embedding.cols() == 768);

    int seq_length = 10;
    int vocab_size = 50257;

    MatrixXf expected_logits = readMatrixFromFile("tests/test_data/gpt2/gpt2_output.txt", seq_length, vocab_size);

    Eigen::MatrixXf logits = gpt2.forward(text);

    REQUIRE(matrices_approx_equal(logits, expected_logits, 1e-2));

    string_t next_token = gpt2.get_next_max_like_token(logits);

    REQUIRE(next_token == "Ġto");
}