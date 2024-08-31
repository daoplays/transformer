#include "test_utils.h"
#include "../src/utils.h"
#include <fstream>
#include <filesystem>

bool matrices_approx_equal(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float epsilon) {
    return (m1 - m2).cwiseAbs().maxCoeff() < epsilon;
}

Eigen::MatrixXf readMatrixFromFile(const std::string& filename, int rows, int cols) {
    
    if (!std::filesystem::exists(filename)) {
        die("file " + filename + " does not exist");
    }
    
    std::ifstream file(filename);
    Eigen::MatrixXf matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix(i, j);
        }
    }
    return matrix;
}

Eigen::VectorXf readVectorFromFile(const std::string& filename) {

    if (!std::filesystem::exists(filename)) {
        die("file " + filename + " does not exist");
    }
    std::ifstream file(filename);
    std::vector<float> data;
    float value;
    while (file >> value) {
        data.push_back(value);
    }
    return Eigen::Map<Eigen::VectorXf>(data.data(), data.size());
}