#pragma once
#include <H5Cpp.h>
#include "eigen_config.h"
#include "types/basic_types.h"
#include "gpt2.h"

Eigen::MatrixXf read_matrix(const H5::H5File& file, const string_t& dataset_name);
gpt2_weights_t load_gpt2_weights(const string_t& h5_file_path);
