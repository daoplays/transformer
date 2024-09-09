#include "norm_layer.h"
#include <iostream>

MatrixXf norm_layer_t::forward(const MatrixXf& x)
{

    Eigen::RowVectorXf mean = x.rowwise().mean();

    Eigen::RowVectorXf var = (x.colwise() - mean.transpose()).array().square().rowwise().mean().sqrt();

    Eigen::MatrixXf x_norm = (x.colwise() - mean.transpose()).array().colwise() / (var.transpose().array() + eps);

    Eigen::MatrixXf result = (x_norm.transpose().array().colwise() * gamma.array()).colwise() + beta.array();

    return result.transpose();
}
