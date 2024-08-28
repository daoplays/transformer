#include "test_utils.h"

bool matrices_approx_equal(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float epsilon) {
    return (m1 - m2).cwiseAbs().maxCoeff() < epsilon;
}