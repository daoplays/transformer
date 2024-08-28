#include "logger.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "argument_parser.h"

void logger::log_debug(const string_t& message)
{
    if (args::verbose)
        log(log_level::DEBUG, message);
}

void logger::log_info(const string_t& message)
{
    log(log_level::INFO, message);
}

void logger::log_error(const string_t& message)
{
    log(log_level::ERROR, message);
}

void logger::log(log_level level, const string_t& message)
{
    string_t color_code;
    string_t level_str;

    switch (level) {
        case log_level::DEBUG:
            color_code = "\033[0;32m";  // Green
            level_str = "DEBUG";
            break;
        case log_level::INFO:
            color_code = "\033[0;34m";  // Blue
            level_str = "INFO";
            break;
        case log_level::ERROR:
            color_code = "\033[0;31m";  // Red
            level_str = "ERROR";
            break;
    }

    string_t reset_color = "\033[0m";

    auto now = std::chrono::system_clock::now();

    // Convert to c-style time_t to use std::localtime
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::cout << color_code << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " " << level_str << " " << message << reset_color
              << std::endl;
}