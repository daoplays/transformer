#include "load_h5.h"
#include <iomanip>
#include <iostream>

Eigen::MatrixXf read_matrix_from_h5(const H5::H5File& file, const string_t& dataset_name)
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

Eigen::VectorXf read_vector_from_h5(const H5::H5File& file, const std::string& dataset_name) {
    H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data(), NULL);
    
    hsize_t total_size = 1;
    for (hsize_t dim : dims) {
        total_size *= dim;
    }
    
    Eigen::VectorXf vector(total_size);
    dataset.read(vector.data(), H5::PredType::NATIVE_FLOAT);
    
    return vector;
}

