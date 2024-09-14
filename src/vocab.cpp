#include "vocab.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "logger.h"
#include "types/basic_types.h"
#include "utils.h"

tokenizer_t::tokenizer_t(const string_t& vocab_file, const string_t& merges_file)
{
    // Load vocabulary from JSON file
    std::ifstream vocab_stream(vocab_file);
    json vocab_json;
    vocab_stream >> vocab_json;
    for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        int id = it.value();
        encoder[it.key()] = id;
        decoder[id] = it.key();
    }

    // Load BPE merges from text file
    std::ifstream merges_stream(merges_file);
    string_t line;
    std::getline(merges_stream, line);  // Skip first line (header)

    while (std::getline(merges_stream, line)) {
        if (line.empty()) {
            break;
        }

        size_t split_pos = line.find(' ');
        if (split_pos == string_t::npos) {
            die("Invalid line in merges file: " + line);
        }

        string_t first = line.substr(0, split_pos);
        string_t second = line.substr(split_pos + 1);

        // Add to bpe_ranks
        bpe_ranks.emplace_back(first, second);
    }

    // Compile regex pattern for tokenization
    // This pattern matches various token types: contractions, words, numbers, punctuation, and whitespace
    pat = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+");
    // Initialize byte encoder/decoder
    byte_encoder = bytes_to_unicode();
}

// Function to create byte-to-unicode mapping
std::map<uint8_t, char32_t> tokenizer_t::bytes_to_unicode()
{
    // Purpose: Create a reversible mapping between byte values (0-255) and Unicode characters.
    // The specific implementation here follows the original GPT-2 tokenizer_t's approach.

    std::vector<uint8_t> bs;
    // Step 1: Add printable ASCII characters (33 to 126, i.e., '!' to '~')
    for (int i = 33; i <= 126; ++i)
        bs.push_back(i);
    // Step 2: Add extended ASCII characters (161 - '¡' to 172 - '¬' and 174 - '®'to 255 - 'ÿ')
    for (int i = 161; i <= 172; ++i)
        bs.push_back(i);
    for (int i = 174; i <= 255; ++i)
        bs.push_back(i);

    // Create a copy of bs to store the Unicode mappings
    std::vector<char32_t> cs(bs.begin(), bs.end());
    int n = 0;
    // Step 3: Handle remaining byte values (0-32, 127-160, 173)
    // These include control characters and some extended ASCII characters that might cause issues
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            // Map to Unicode characters starting from 256
            // Note: we add 256 to avoid conflicts with the ASCII range
            cs.push_back(256 + n);
            ++n;
        }
    }

    // Create the final mapping
    std::map<uint8_t, char32_t> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = cs[i];
    }
    return result;

    // Note on implementation choice:
    // 1. This approach prioritizes printable ASCII and common extended ASCII characters.
    // 2. It ensures these characters map to themselves, maintaining readability for common text.
    // 3. Less common byte values (control characters, some extended ASCII) are mapped to higher Unicode points.
    // 4. While a simple 0-255 loop would work, this method optimizes for human-readable output in common cases.
    // 5. This specific implementation ensures compatibility with pre-trained GPT-2 models.
}

int tokenizer_t::get_pair_rank(const string_t& first, const string_t& second)
{
    auto it = std::find(bpe_ranks.begin(), bpe_ranks.end(), std::make_pair(first, second));
    if (it != bpe_ranks.end()) {
        return std::distance(bpe_ranks.begin(), it);
    }
    return -1;  // Pair not found
}

std::vector<string_t> tokenizer_t::bpe(const std::u32string& token)
{
    // Initialize a vector of UTF-32 strings. Each string it just a single character from the input token
    std::vector<std::u32string> word;
    word.reserve(token.size());
    for (char32_t c : token) {
        word.push_back(std::u32string(1, c));
    }

    // Main BPE loop
    while (true) {
        std::pair<std::u32string, std::u32string> best_pair;
        int best_rank = -1;

        // Find the best pair to merge based on rank
        for (size_t i = 0; i < word.size() - 1; ++i) {
            // Get the rank of the current pair
            int rank = get_pair_rank(utf32_to_utf8(word[i]), utf32_to_utf8(word[i + 1]));
            // Update best_pair and best_rank if this pair is better
            if (rank != -1 && (best_rank == -1 || rank < best_rank)) {
                best_pair = {word[i], word[i + 1]};
                best_rank = rank;
            }
        }

        // If no mergeable pair found, exit the loop
        if (best_rank == -1) {
            break;
        }

        // Merge the best pair in the word
        std::vector<std::u32string> new_word;
        for (size_t i = 0; i < word.size(); ++i) {
            if (i < word.size() - 1 && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                // Merge the pair
                new_word.push_back(best_pair.first + best_pair.second);
                ++i;  // Skip the next token as it's now merged
            } else {
                // Keep the token as is
                new_word.push_back(word[i]);
            }
        }

        // Update word with the new merged version
        word = std::move(new_word);
    }

    // Convert the final word vector from UTF-32 to UTF-8
    std::vector<string_t> result;
    for (const std::u32string& w : word) {
        result.push_back(utf32_to_utf8(w));
    }

    return result;
}

// Tokenize input text
std::vector<int> tokenizer_t::tokenize(const string_t& text)
{
    std::vector<int> tokens;

    // Use regex to split text into initial tokens
    std::sregex_iterator iter(text.begin(), text.end(), pat);
    std::sregex_iterator end_iter;  // Default constructor creates a past-the-end iterator

    while (iter != end_iter) {
        string_t utf8_token = iter->str();

        // Apply byte-level encoding
        std::u32string utf32_token;
        for (uint8_t b : utf8_token) {
            utf32_token += byte_encoder.at(b);
        }

        // Apply BPE encoding
        std::vector<string_t> bpe_encoded = bpe(utf32_token);
        for (const string_t& bpe_token : bpe_encoded) {
            int token_id = encoder.at(bpe_token);
            tokens.push_back(token_id);
        }

        // Move to the next regex match
        ++iter;
    }

    return tokens;
}

// Helper function to convert tokens back to text (for debugging)
std::vector<string_t> tokenizer_t::detokenize(const std::vector<int>& tokens)
{
    std::vector<string_t> result;
    for (int token : tokens) {
        string_t token_str = decoder[token];
        result.push_back(token_str);
    }

    return result;
}
