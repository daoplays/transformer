#include "load_h5.h"
#include <iomanip>
#include <iostream>

Eigen::MatrixXf read_matrix(const H5::H5File& file, const string_t& dataset_name)
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

Eigen::VectorXf read_vector(const H5::H5File& file, const std::string& dataset_name) {
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

void print_vector_info(const Eigen::VectorXf& vector, const std::string& name)
{
    std::cout << name << " size: " << vector.size() << std::endl;
    std::cout << "First few elements of " << name << ":" << std::endl;
    std::cout << std::setprecision(8) << std::fixed;
    for (int i = 0; i < std::min(5, static_cast<int>(vector.size())); ++i) {
            std::cout << vector(i)  << std::endl;
        
    }
}

gpt2_weights_t load_gpt2_weights(const string_t& h5_file_path)
{
    gpt2_weights_t weights;
    H5::H5File file(h5_file_path, H5F_ACC_RDONLY);

    string_t base_path = "/transformer/tfgp_t2lm_head_model/transformer/";

    weights.token_embedding = read_matrix(file, base_path + "wte/weight:0");
    weights.position_embedding = read_matrix(file, base_path + "wpe/embeddings:0");

    //print_matrix_info(weights.token_embedding, "WTE");
    //print_matrix_info(weights.position_embedding, "WPE");

    // Load layers
    for (int i = 0; i < 12; ++i) {

        std::string layer_path = base_path + "h_._" + std::to_string(i) + "/";
        gpt2_layer_t layer;

        // Attention weights
        layer.attn_c_attn_weight = read_matrix(file, layer_path + "attn/c_attn/weight:0");
        layer.attn_c_attn_bias = read_vector(file, layer_path + "attn/c_attn/bias:0");
        layer.attn_c_proj_weight = read_matrix(file, layer_path + "attn/c_proj/weight:0");
        layer.attn_c_proj_bias = read_vector(file, layer_path + "attn/c_proj/bias:0");

        //print_matrix_info(layer.attn_c_attn_weight, "Attention Weight");
        //print_matrix_info(layer.attn_c_proj_weight, "Attention Projection");
        //print_vector_info(layer.attn_c_attn_bias, "Attention Bias");
        //print_vector_info(layer.attn_c_proj_bias, "Attention Projection Bias");
      

        // MLP weights
        layer.mlp_c_fc_weight = read_matrix(file, layer_path + "mlp/c_fc/weight:0");
        layer.mlp_c_fc_bias = read_vector(file, layer_path + "mlp/c_fc/bias:0");
        layer.mlp_c_proj_weight = read_matrix(file, layer_path + "mlp/c_proj/weight:0");
        layer.mlp_c_proj_bias = read_vector(file, layer_path + "mlp/c_proj/bias:0");

        // Layer normalization weights
        layer.ln_1_weight = read_vector(file, layer_path + "ln_1/gamma:0");
        layer.ln_1_bias = read_vector(file, layer_path + "ln_1/beta:0");
        layer.ln_2_weight = read_vector(file, layer_path + "ln_2/gamma:0");
        layer.ln_2_bias = read_vector(file, layer_path + "ln_2/beta:0");

        weights.layers.push_back(layer);
    }

    // Load final layer normalization weights
    weights.ln_f_weight = read_vector(file, base_path + "ln_f/gamma:0");
    weights.ln_f_bias = read_vector(file, base_path + "ln_f/beta:0");


    return weights;
}