#include "feed_forward.h"

MatrixXf feed_forward_t::forward(const MatrixXf& X)
{
    // First linear transformation with bias, followed by ReLU activation
    Eigen::MatrixXf hidden = apply_relu((X * W1.transpose()).rowwise() + b1.transpose());
    // Second linear transformation with bias
    return (hidden * W2.transpose()).rowwise() + b2.transpose();
}