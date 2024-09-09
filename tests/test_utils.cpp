#include "test_utils.h"
#include <filesystem>
#include <fstream>
#include "../src/utils.h"

bool matrices_approx_equal(const Eigen::MatrixXf& m1, const Eigen::MatrixXf& m2, float epsilon)
{
    return (m1 - m2).cwiseAbs().maxCoeff() < epsilon;
}

Eigen::MatrixXf readMatrixFromFile(const std::string& filename, int rows, int cols)
{
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("File " + filename + " does not exist");
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    Eigen::MatrixXf matrix(rows, cols);
    std::string line;
    for (int i = 0; i < rows; ++i) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file " + filename);
        }
        std::istringstream iss(line);
        for (int j = 0; j < cols; ++j) {
            if (!(iss >> matrix(i, j))) {
                throw std::runtime_error("Error reading value at position (" + std::to_string(i) + "," + std::to_string(j) + ") in file " + filename);
            }
        }
    }
    return matrix;
}

Eigen::VectorXf readVectorFromFile(const std::string& filename)
{

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