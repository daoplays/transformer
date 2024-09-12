#pragma once
#include <H5Cpp.h>
#include "eigen_config.h"
#include "types/basic_types.h"

struct GPT2_Weights {
    Eigen::MatrixXf token_embedding;
    Eigen::MatrixXf position_embedding;
};

Eigen::MatrixXf read_dataset(const H5::H5File& file, const string_t& dataset_name);
GPT2_Weights load_embeddings(const string_t& h5_file_path);
