#include "feed_forward.h"

MatrixXf feed_forward_t::forward(const MatrixXf &X)
{
    // First linear transformation followed by ReLU activation
    Eigen::MatrixXf hidden = apply_relu(X * W1);
    // Second linear transformation
    return hidden * W2;
}