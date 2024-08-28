#include "utils.h"

// Softmax function
// Used in attention mechanism to convert scores to probabilities
VectorXf softmax(const VectorXf &x)
{
    VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x.array() / exp_x.sum();
}
