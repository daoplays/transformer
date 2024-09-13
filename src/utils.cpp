#include "utils.h"
#include <cmath>
#include <random>
#include "logger.h"

void die(const string_t& message)
{
    // Print the error message
    logger::log_error(message);

    // Throw a runtime_error instead of a custom exception
    throw std::runtime_error(message);
}

// Softmax function
// Used in attention mechanism to convert scores to probabilities
VectorXf softmax(const VectorXf& x)
{
    VectorXf exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x.array() / exp_x.sum();
}

/**
 * Performs He initialization on a given Eigen matrix.
 *
 * He initialization is designed to work well with ReLU activation functions.
 * It draws values from a normal distribution with mean 0 and
 * standard deviation sqrt(2 / fan_in).
 *
 * Pros:
 * - Helps maintain the variance of activations across layers, especially for ReLU networks.
 * - Can lead to faster convergence in deep networks with ReLU activations.
 * - Mitigates the vanishing/exploding gradient problem.
 *
 * Cons:
 * - Not ideal for non-ReLU activations (e.g., sigmoid, tanh).
 * - May need adjustment for very deep networks or certain architectures.
 *
 * References:
 * - He, K. et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 *   Performance on ImageNet Classification. arXiv:1502.01852.
 * - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
 *   deep feedforward neural networks. AISTATS.
 *
 * @param matrix The Eigen matrix to initialize
 * @param optional_fan_in The number of input units in the weight tensor
 */
void he_initialization(Eigen::MatrixXf& matrix, std::optional<int> optional_fan_in = std::nullopt)
{

    std::random_device rd;
    std::mt19937 gen(rd());

    int fan_in = optional_fan_in.value_or(matrix.cols());

    // Calculate standard deviation
    float std_dev = std::sqrt(2.0f / fan_in);

    // Create a normal distribution with mean 0 and calculated std_dev
    std::normal_distribution<float> d(0, std_dev);

    // Fill the matrix with values drawn from the distribution
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix(i, j) = d(gen);
        }
    }
}

void allocate_and_initialize(Eigen::MatrixXf& matrix, int rows, int cols)
{
    matrix = Eigen::MatrixXf(rows, cols);
    he_initialization(matrix);
}

float relu(float x)
{
    return std::max(0.0f, x);
}

float gelu(float x) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * std::pow(x, 3.0f))));
}

Eigen::MatrixXf apply_relu(const Eigen::MatrixXf& X)
{
    return X.unaryExpr(&relu);
}

Eigen::MatrixXf apply_gelu(const Eigen::MatrixXf& X)
{
    return X.unaryExpr(&gelu);
}