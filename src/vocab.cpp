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

        // create the encoder entry mapping string -> int
        encoder[it.key()] = id;
        // create the corresponding decoder entry int -> string
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

        // the merges file is just space separated
        size_t split_pos = line.find(' ');
        if (split_pos == string_t::npos) {
            die("Invalid line in merges file: " + line);
        }

        string_t first = line.substr(0, split_pos);
        string_t second = line.substr(split_pos + 1);

        // Add this merge rule to merge_ranks
        merge_ranks.emplace_back(first, second);
    }

    // Compile regex pattern for tokenization
    // This pattern matches various token types: contractions, words, numbers, punctuation, and whitespace
    regex_splitter = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+");
    // Initialize byte encoder/decoder
    byte_encoder = bytes_to_unicode();
}

// Function to create byte-to-unicode mapping for GPT-2 tokenization
std::map<uint8_t, char32_t> tokenizer_t::bytes_to_unicode()
{
    // Purpose: Create a specific bijective mapping between byte values (0-255) and Unicode code points.
    // This mapping is designed to be consistent with GPT-2's original tokenization scheme.

    std::vector<uint8_t> bs;
    // Step 1: Add printable ASCII characters (33 to 126, i.e., '!' to '~')
    // Note: We will handl 0-32 (and the other missing values) later
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
    // Step 3: Map remaining byte values (0-32, 127-160, 173) to Unicode points starting at 256
    // This includes control characters, space, delete, and some extended ASCII characters
    // Mapping these to 256+ ensures:
    // 1. Consistency with GPT-2's original tokenization scheme
    // 2. Clear visual distinction of special characters during debugging
    // 3. Avoidance of potential issues with the way text editors handle control characters
    
    for (int b = 0; b < 256; ++b) {

        // if we have already added this byte, skip it
        if (std::find(bs.begin(), bs.end(), b) != bs.end()) 
            continue;

        bs.push_back(b);
        // Map to Unicode characters starting from 256
        // Note: we add 256 to avoid conflicts with the ASCII range
        cs.push_back(256 + n);
        ++n;
        
    }

    // Create the final mapping
    // Note: We need to use char32_t rather than char to handle Unicode code points over 255
    std::map<uint8_t, char32_t> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = cs[i];
    }
    return result;
}


// search the merge list for this pair of strings
int tokenizer_t::get_pair_rank(const string_t& first, const string_t& second)
{
    auto it = std::find(merge_ranks.begin(), merge_ranks.end(), std::make_pair(first, second));
    if (it != merge_ranks.end()) {
        return std::distance(merge_ranks.begin(), it);
    }
    return -1;  // Pair not found
}

// performs byte pair encoding on a UTF-32 encoded input
std::vector<string_t> tokenizer_t::bpe(const std::u32string& input)
{
    // Initialize a vector of UTF-32 tokens. Right now each entry it just a single character from the input, however these will potentially get merged through the BPE process
    std::vector<std::u32string> tokens;
    tokens.reserve(input.size());
    for (char32_t c : input) {
        tokens.push_back(std::u32string(1, c));
    }

    // Main BPE loop
    while (true) {
        std::pair<std::u32string, std::u32string> best_pair;
        int best_rank = -1;

        // Find the best pair to merge based on rank
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            // Get the rank of the current pair
            int rank = get_pair_rank(utf32_to_utf8(tokens[i]), utf32_to_utf8(tokens[i + 1]));
            // Update best_pair and best_rank if this pair is better
            if (rank != -1 && (best_rank == -1 || rank < best_rank)) {
                best_pair = {tokens[i], tokens[i + 1]};
                best_rank = rank;
            }
        }

        // If no mergeable pair found, exit the loop
        if (best_rank == -1) {
            break;
        }

        // Merge the best pair of tokens
        std::vector<std::u32string> merged_tokens;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i < tokens.size() - 1 && tokens[i] == best_pair.first && tokens[i + 1] == best_pair.second) {
                // Merge the pair
                merged_tokens.push_back(best_pair.first + best_pair.second);
                ++i;  // Skip the next token as it's now merged
            } else {
                // Keep the token as is
                merged_tokens.push_back(tokens[i]);
            }
        }

        // Update word with the new merged version
        tokens = std::move(merged_tokens);
    }

    // Convert the final vector from UTF-32 to UTF-8
    std::vector<string_t> result;
    for (const std::u32string& token : tokens) {
        result.push_back(utf32_to_utf8(token));
    }

    return result;
}

// Tokenize input text
std::vector<int> tokenizer_t::tokenize(const string_t& text)
{
    std::vector<int> tokens;

    // Use regex to try and split text into smaller chunks
    std::sregex_iterator iter(text.begin(), text.end(), regex_splitter);
    // A default-constructed std::sregex_iterator represents the past-the-end iterator
    std::sregex_iterator end_iter; 

    // while there are chunks left, tokenize them
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
