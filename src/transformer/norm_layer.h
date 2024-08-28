
#pragma once
#include "../eigen_config.h"

class norm_layer_t {
public:
    norm_layer_t(int features, float eps = 1e-5) : gamma(VectorXf::Ones(features)), beta(VectorXf::Zero(features)), eps(eps) {}

   MatrixXf forward(const MatrixXf& x);


private:
    VectorXf gamma, beta;
    float eps;
};