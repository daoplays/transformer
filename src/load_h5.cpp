#include "load_h5.h"
#include <iomanip>
#include <iostream>

Eigen::MatrixXf read_dataset(const H5::H5File& file, const string_t& dataset_name)
{
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();

    int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data(), NULL);

    std::vector<float> data(dims[0] * dims[1]);
    dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

    return Eigen::Map<Eigen::MatrixXf>(data.data(), dims[1], dims[0]).transpose();
}

void print_matrix_info(const Eigen::MatrixXf& matrix, const std::string& name)
{
    std::cout << name << " shape: " << matrix.rows() << "x" << matrix.cols() << std::endl;
    std::cout << "First few elements of " << name << ":" << std::endl;
    std::cout << std::setprecision(8) << std::fixed;
    for (int i = 0; i < std::min(5, static_cast<int>(matrix.rows())); ++i) {
        for (int j = 0; j < std::min(5, static_cast<int>(matrix.cols())); ++j) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

GPT2_Weights load_embeddings(const string_t& h5_file_path)
{
    GPT2_Weights weights;
    H5::H5File file(h5_file_path, H5F_ACC_RDONLY);

    weights.token_embedding = read_dataset(file, "/transformer/tfgp_t2lm_head_model/transformer/wte/weight:0");
    weights.position_embedding = read_dataset(file, "/transformer/tfgp_t2lm_head_model/transformer/wpe/embeddings:0");

    print_matrix_info(weights.token_embedding, "WTE");
    print_matrix_info(weights.position_embedding, "WPE");
    return weights;
}