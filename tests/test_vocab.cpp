#include <catch2/catch_all.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../src/load_h5.h"
#include "../src/utils.h"
#include "../src/vocab.h"

TEST_CASE("Vocabulary loader correctly loads GPT-2 vocabulary", "[vocab_loader]")
{
    // Initialize tokenizer_t with vocabulary and merges files
    tokenizer_t tokenizer("gpt2/vocab.json", "gpt2/merges.txt");

    // GPT-2 vocabulary size is 50257
    REQUIRE(tokenizer.get_vocab_size() == 50257);
    // GPT-2 mergers size is 50000
    REQUIRE(tokenizer.get_mergers_size() == 50000);

    // Check some specific tokens
    CHECK(tokenizer.tokenize("Twitter")[0] == 14254);
    CHECK(tokenizer.tokenize("Bitcoin")[0] == 22614);

    CHECK(tokenizer.tokenize("!")[0] == 0);
    CHECK(tokenizer.tokenize("~")[0] == 93);

    // Example text to tokenize
    std::string text = "GPT2 is a model developed by OpenAI";
    // Tokenize the text
    std::vector<int> tokens = tokenizer.tokenize(text);
    std::vector<string_t> detokenized = tokenizer.detokenize(tokens);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " " << detokenized[i] << std::endl;
    }

    // expected tokens from the hugging face python api
    std::vector<int> expected_tokens = {38, 11571, 17, 318, 257, 2746, 4166, 416, 4946, 20185};
    REQUIRE(tokens == expected_tokens);

    GPT2_Weights gpt_weights = load_embeddings("/home/ltl/Documents/machine_learning/gpt2/tf_model.h5");

    REQUIRE(gpt_weights.token_embedding.rows() == 50257);
    REQUIRE(gpt_weights.token_embedding.cols() == 768);
    REQUIRE(gpt_weights.position_embedding.rows() == 1024);
    REQUIRE(gpt_weights.position_embedding.cols() == 768);

    Eigen::MatrixXf embedded_tokens(tokens.size(), gpt_weights.token_embedding.cols());

    for (size_t i = 0; i < tokens.size(); ++i) {
        // Check if the token ID is within the valid range
        if (tokens[i] >= 0 && tokens[i] < gpt_weights.token_embedding.rows()) {
            embedded_tokens.row(i) = gpt_weights.token_embedding.row(tokens[i]);
        } else {
            die("Invalid token ID: " + std::to_string(tokens[i]));
        }
    }

    std::cout << embedded_tokens.col(0) << std::endl;
    std::cout << embedded_tokens.rows() << " " << embedded_tokens.cols() << std::endl;
}