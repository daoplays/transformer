
#pragma once
#include <iostream>
#include "../eigen_config.h"
#include "../utils.h"

class norm_layer_t {
public:

    norm_layer_t(int features, float eps = 1e-5) : gamma(VectorXf::Ones(features)), beta(VectorXf::Zero(features)), eps(eps) {}

    MatrixXf forward(const MatrixXf& x);

    void setGammaBeta(const Eigen::VectorXf& new_gamma, const Eigen::VectorXf& new_beta)
    {
        if (new_gamma.size() != gamma.size() || new_beta.size() != beta.size()) {
            std::cout << "new_gamma: " << new_gamma.size() << std::endl;
            std::cout << "gamma: " << gamma.size() << std::endl;
            std::cout << "new_beta: " << new_beta.size() << std::endl;
            std::cout << "beta: " << beta.size() << std::endl;
            die("Gamma and beta must have the same size as the number of features");
        }
        gamma = new_gamma;
        beta = new_beta;
    }

private:

    VectorXf gamma, beta;
    float eps;
};