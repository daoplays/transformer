#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <iostream>

const char* SUCCESS_ART = R"(
   _____  _    _  _____ _____ ______  _____ _____
  / ____|| |  | |/ ____/ ____|  ____|/ ____/ ____|
 | (___  | |  | | |   | |    | |__  | (___| (___
  \___ \ | |  | | |   | |    |  __|  \___ \\___ \
  ____) || |__| | |___| |____| |____ ____) |___) |
 |_____/  \____/ \_____\_____|______|_____/_____/

)";

int main(int argc, char* argv[])
{

    int result = Catch::Session().run(argc, argv);

    if (result == 0) {
        std::cout << "\033[1;32m" << SUCCESS_ART << "\033[0m" << std::endl;
    }

    return result;
}