#pragma once

#include <boost/program_options.hpp>
#include "logger.h"
#include "types/basic_types.h"

// the values for the arguments live in the args namespace
namespace args {

extern bool verbose;
extern bool help;
}  // namespace args

class argument_parser_t {
public:

    argument_parser_t();

    // returns true if the arguments were parsed successfully
    // otherwise it will print relevant error and return false
    bool parse(int argc, char* argv[]);
    void print_help(void) const;

private:

    boost::program_options::options_description opt_desc;
    boost::program_options::variables_map var_map;
};