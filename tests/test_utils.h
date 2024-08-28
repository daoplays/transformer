#pragma once
#include "../src/eigen_config.h"

// Helper function to check if two Eigen matrices are approximately equal
bool matrices_approx_equal(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float epsilon = 1e-5f) ;