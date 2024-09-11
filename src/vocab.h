#pragma once
#include <unordered_map>
#include "types/basic_types.h"

std::unordered_map<string_t, int> load_vocab(const string_t& filename);