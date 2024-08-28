#pragma once

#include "types/basic_types.h"

class logger {
public:

    enum class log_level { DEBUG, INFO, ERROR };

    static void log_debug(const string_t& message);
    static void log_info(const string_t& message);
    static void log_error(const string_t& message);

private:

    static void log(log_level level, const string_t& message);
};