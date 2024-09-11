#include <catch2/catch_all.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../src/vocab.h"

TEST_CASE("Vocabulary loader correctly loads GPT-2 vocabulary", "[vocab_loader]")
{
    string_t gpt2_vocab_path = "gpt2/vocab.json";

    REQUIRE_NOTHROW([&]() {
        auto vocab = load_vocab(gpt2_vocab_path);

        // GPT-2 vocabulary size is 50257
        REQUIRE(vocab.size() == 50257);

        // Check some specific tokens
        CHECK(vocab["<|endoftext|>"] == 50256);  // Special token, should be the last one
        CHECK(vocab["Twitter"] == 14254);
        CHECK(vocab["Bitcoin"] == 22614);

        CHECK(vocab["!"] == 0);
        CHECK(vocab["~"] == 93);

        // Verify that all keys are strings and all values are unique integers
        std::set<int> unique_values;
        std::map<int, std::vector<std::string>> id_to_tokens;
        for (const auto& [token, id] : vocab) {
            unique_values.insert(id);
            id_to_tokens[id].push_back(token);
        }

        // Print out any duplicate entries
        for (const auto& [id, tokens] : id_to_tokens) {
            if (tokens.size() > 1) {
                std::cout << "Duplicate ID " << id << " for tokens: ";
                for (const auto& token : tokens) {
                    std::cout << "'" << token << "' ";
                }
                std::cout << std::endl;
            }
        }
        CHECK(unique_values.size() == vocab.size());
    }());
}