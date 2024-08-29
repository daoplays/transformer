
#pragma once
#include "../eigen_config.h"
#include <iostream>
#include "../utils.h"
class norm_layer_t {
public:
    norm_layer_t(int features, float eps = 1e-5) : gamma(VectorXf::Ones(features)), beta(VectorXf::Zero(features)), eps(eps) {}

   MatrixXf forward(const MatrixXf& x);

   void setGammaBeta(const Eigen::VectorXf& new_gamma, const Eigen::VectorXf& new_beta) {
    if (new_gamma.size() != gamma.size() || new_beta.size() != beta.size()) {
            die("Gamma and beta must have the same size as the number of features");
        }
    gamma = new_gamma;
    beta = new_beta;

    }


private:
    VectorXf gamma, beta;
    float eps;
};