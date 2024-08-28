#include "norm_layer.h"

MatrixXf norm_layer_t::forward(const MatrixXf& x) {
        Eigen::RowVectorXf mean = x.colwise().mean();
        Eigen::RowVectorXf var = ((x.rowwise() - mean).array().square().colwise().sum() / x.rows()).sqrt();
        
        MatrixXf x_norm = (x.rowwise() - mean).array().rowwise() / (var.array() + eps);
        return (x_norm.array().rowwise() * gamma.transpose().array()).rowwise() + beta.transpose().array();
    }
