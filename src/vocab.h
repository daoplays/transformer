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
    std::map<std::u32string, int> encoder;
    // Decoder: maps IDs back to tokens
    std::map<int, std::u32string> decoder;
    // BPE ranks: stores the priority of merge operations
    std::vector<std::pair<std::u32string, std::u32string>> bpe_ranks;
    // Regex pattern for tokenization
    std::regex pat;
    // Byte-to-unicode mapping
    std::map<uint8_t, char32_t> byte_encoder;

    std::map<uint8_t, char32_t> bytes_to_unicode();

    // Function to find the rank of a pair
    int get_pair_rank(const std::u32string& first, const std::u32string& second);

    std::vector<std::u32string> bpe(const std::u32string& token);

    // UTF-8 to UTF-32 conversion using standard C++
    std::u32string utf8_to_utf32(const std::string& utf8_string)
    {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.from_bytes(utf8_string);
    }

    std::string utf32_to_utf8(const std::u32string& input)
    {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.to_bytes(input);
    }

public:

    tokenizer_t(const std::string& vocab_file, const std::string& merges_file);

    // Tokenize input text
    std::vector<int> tokenize(const std::string& text);
    std::vector<std::string> detokenize(const std::vector<int>& tokens);

    // helper functions for testing
    int get_vocab_size() { return encoder.size(); };

    int get_mergers_size() { return bpe_ranks.size(); };
};
