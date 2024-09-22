#pragma once
#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "types/basic_types.h"

// Use nlohmann::json for JSON parsing
using json = nlohmann::json;

class tokenizer_t {
private:

    // Encoder: maps tokens to IDs
    std::map<string_t, int> encoder;
    // Decoder: maps IDs back to tokens
    std::map<int, string_t> decoder;
    // merge_ranks: stores the priority of merge operations
    std::vector<std::pair<string_t, string_t>> merge_ranks;
    // Regex pattern for tokenization - performs initial splitting of input string
    std::regex regex_splitter;
    // Byte-to-unicode mapping
    std::map<uint8_t, char32_t> byte_encoder;

    std::map<uint8_t, char32_t> bytes_to_unicode();

    // Function to find the rank of a pair
    int get_pair_rank(const string_t& first, const string_t& second);

    std::vector<string_t> bpe(const std::u32string& token);

    // UTF-8 to UTF-32 conversion using standard C++
    std::u32string utf8_to_utf32(const string_t& utf8_string)
    {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.from_bytes(utf8_string);
    }

    string_t utf32_to_utf8(const std::u32string& input)
    {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.to_bytes(input);
    }

public:

    tokenizer_t(const string_t& vocab_file, const string_t& merges_file);

    // Tokenize input text
    std::vector<int> tokenize(const string_t& text);
    std::vector<string_t> detokenize(const std::vector<int>& tokens);
    string_t detokenize(const int token);

    // helper functions for testing
    int get_vocab_size() { return encoder.size(); };

    int get_mergers_size() { return merge_ranks.size(); };
};
