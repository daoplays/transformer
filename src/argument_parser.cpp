#include "argument_parser.h"
#include <iostream>
#include "logger.h"
#include "utils.h"

namespace po = boost::program_options;

namespace args {

bool verbose = false;
bool help = false;
}  // namespace args

// Helper function for regular options
template <class T>
void add_option(po::options_description& opt_description, const string_t full_name, T& variable, const string_t& description)
{
    auto options = opt_description.add_options();
    auto value_semantic = po::value<T>(&variable);

    options(full_name.c_str(), value_semantic, description.c_str());
}


// Specialization for bool
void add_option(po::options_description& opt_desc, const string_t full_name, bool& variable, const string_t& description)
{
    auto options = opt_desc.add_options();
    auto option_value_handler = po::bool_switch(&variable);

    options(full_name.c_str(), option_value_handler, description.c_str());
}

argument_parser_t::argument_parser_t() : opt_desc("Allowed options")
{

    add_option(opt_desc, "help,h", args::help, "produce help message");
    add_option(opt_desc, "verbose,v", args::verbose, "verbose (optional)");
}

bool argument_parser_t::parse(int argc, char* argv[])
{
    try {
        po::store(po::parse_command_line(argc, argv, opt_desc), var_map);
        po::notify(var_map);

    } catch (std::exception& e) {
        string_t error_message = e.what();
        logger::log_error("Error parsing arguments: " + error_message);
        return false;
    } catch (...) {
        logger::log_error("Exception of unknown type");
        return false;
    }

    // if we are just asking for help, return without checking for required arguments
    if (args::help) {
        return true;
    }

    return true;
}

void argument_parser_t::print_help() const
{
    std::cout << opt_desc << "\n";
}