#include <iostream>
#include "argument_parser.h"
#include "eigen_config.h"
#include "transformer/transformer.h"

int main(int argc, char* argv[])
{

    argument_parser_t parser;

    // if arguments cannot be parsed, we print the help message and exit
    if (!parser.parse(argc, argv) || args::help) {
        parser.print_help();
        return 1;
    }

    // Hyperparameters
    int d_model = 512;   // Dimensionality of the model
    int num_heads = 8;   // Number of attention heads
    int d_ff = 2048;     // Dimensionality of feed-forward layer
    int num_layers = 6;  // Number of encoder layers
    int seq_len = 10;    // Sequence length

    // Create transformer
    transformer_t transformer(num_layers, d_model, num_heads, d_ff);

    // Create a simple input (batch_size=1, seq_len=10, d_model=512)
    // In practice, this would be your tokenized and embedded input sequence
    MatrixXf input = MatrixXf::Random(seq_len, d_model);

    // Forward pass
    MatrixXf output = transformer.forward(input);

    std::cout << "Output shape: " << output.rows() << "x" << output.cols() << std::endl;

    return 0;
}