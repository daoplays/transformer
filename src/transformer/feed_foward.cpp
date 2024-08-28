#include "feed_forward.h"

MatrixXf feed_forward_t::forward(const MatrixXf &X)
{
    // First linear transformation followed by ReLU activation
    MatrixXf hidden = (X * W1).unaryExpr([](float x)
                                         { return std::max(0.0f, x); });
    // Second linear transformation
    return hidden * W2;
}