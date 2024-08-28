#include "eigen_config.h"
#include <optional>

// Softmax function
// Used in attention mechanism to convert scores to probabilities
VectorXf softmax(const VectorXf &x);

void he_initialization(Eigen::MatrixXf& matrix, std::optional<int> fan_in = std::nullopt);
void allocate_and_initialize(Eigen::MatrixXf matrix, int rows, int cols);

// activation functions
float relu(float x);

Eigen::MatrixXf apply_relu(const Eigen::MatrixXf& X);