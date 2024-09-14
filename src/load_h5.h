#pragma once
#include <H5Cpp.h>
#include "eigen_config.h"
#include "types/basic_types.h"

Eigen::MatrixXf read_matrix_from_h5(const H5::H5File& file, const string_t& dataset_name);
Eigen::VectorXf read_vector_from_h5(const H5::H5File& file, const string_t& dataset_name);
