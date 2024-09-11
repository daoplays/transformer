#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "logger.h"
#include "types/basic_types.h"

std::unordered_map<string_t, int> load_vocab(const string_t& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    nlohmann::json j;
    file >> j;

    std::unordered_map<string_t, int> vocab;
    for (auto& [token, id] : j.items()) {
        if (!id.is_number()) {
            logger::log_error("Warning: ID for token '" + token + "' is not a number. Skipping.");
            continue;
        }

        int id_value = id.get<int>();
        vocab[token] = id_value;
    }

    return vocab;
}