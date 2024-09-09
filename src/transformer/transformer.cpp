#include "transformer.h"

MatrixXf transformer_t::forward(const MatrixXf& X)
{
    // X is the input sequence, shape: [seq_len, d_model]
    MatrixXf output = X;
    // Pass input through each encoder layer
    for (encoder_layer_t& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}