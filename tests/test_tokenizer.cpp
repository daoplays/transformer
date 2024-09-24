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
}