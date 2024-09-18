#include "gpt2.h"
#include "load_h5.h"

gpt2_weights_t load_gpt2_weights(const string_t& h5_file_path)
{
    gpt2_weights_t weights;
    H5::H5File file(h5_file_path, H5F_ACC_RDONLY);

    string_t base_path = "/transformer/tfgp_t2lm_head_model/transformer/";

    weights.token_embedding = read_matrix_from_h5(file, base_path + "wte/weight:0");
    weights.position_embedding = read_matrix_from_h5(file, base_path + "wpe/embeddings:0");

    // print_matrix_info(weights.token_embedding, "WTE");
    // print_matrix_info(weights.position_embedding, "WPE");

    // Load layers
    for (int i = 0; i < 12; ++i) {

        std::string layer_path = base_path + "h_._" + std::to_string(i) + "/";
        gpt2_layer_t layer;

        // Attention weights
        layer.attn_c_attn_weight = read_matrix_from_h5(file, layer_path + "attn/c_attn/weight:0");
        layer.attn_c_attn_bias = read_vector_from_h5(file, layer_path + "attn/c_attn/bias:0");
        layer.attn_c_proj_weight = read_matrix_from_h5(file, layer_path + "attn/c_proj/weight:0");
        layer.attn_c_proj_bias = read_vector_from_h5(file, layer_path + "attn/c_proj/bias:0");

        // print_matrix_info(layer.attn_c_attn_weight, "Attention Weight");
        // print_matrix_info(layer.attn_c_proj_weight, "Attention Projection");
        // print_vector_info(layer.attn_c_attn_bias, "Attention Bias");
        // print_vector_info(layer.attn_c_proj_bias, "Attention Projection Bias");

        // MLP weights
        layer.mlp_c_fc_weight = read_matrix_from_h5(file, layer_path + "mlp/c_fc/weight:0");
        layer.mlp_c_fc_bias = read_vector_from_h5(file, layer_path + "mlp/c_fc/bias:0");
        layer.mlp_c_proj_weight = read_matrix_from_h5(file, layer_path + "mlp/c_proj/weight:0");
        layer.mlp_c_proj_bias = read_vector_from_h5(file, layer_path + "mlp/c_proj/bias:0");

        // Layer normalization weights
        layer.ln_1_weight = read_vector_from_h5(file, layer_path + "ln_1/gamma:0");
        layer.ln_1_bias = read_vector_from_h5(file, layer_path + "ln_1/beta:0");
        layer.ln_2_weight = read_vector_from_h5(file, layer_path + "ln_2/gamma:0");
        layer.ln_2_bias = read_vector_from_h5(file, layer_path + "ln_2/beta:0");

        weights.layers.push_back(layer);
    }

    // Load final layer normalization weights
    weights.ln_f_weight = read_vector_from_h5(file, base_path + "ln_f/gamma:0");
    weights.ln_f_bias = read_vector_from_h5(file, base_path + "ln_f/beta:0");

    return weights;
}

void gpt2_t::init()
{
    weights = load_gpt2_weights("gpt2/tf_model.h5");

    for (int i = 0; i < num_layers; ++i) {

        // Set weights and biases
        transformer.set_layer_weights(i, weights.layers[i].attn_c_attn_weight, weights.layers[i].attn_c_attn_bias,
                                      weights.layers[i].attn_c_proj_weight, weights.layers[i].attn_c_proj_bias, weights.layers[i].ln_1_weight,
                                      weights.layers[i].ln_1_bias, weights.layers[i].mlp_c_fc_weight.transpose(), weights.layers[i].mlp_c_fc_bias,
                                      weights.layers[i].mlp_c_proj_weight.transpose(), weights.layers[i].mlp_c_proj_bias,
                                      weights.layers[i].ln_2_weight, weights.layers[i].ln_2_bias);
    }

    final_norm_layer.setGammaBeta(weights.ln_f_weight, weights.ln_f_bias);
}

Eigen::MatrixXf gpt2_t::forward(string_t input_string)
{
    std::vector<int> tokens = tokenizer.tokenize(input_string);

    Eigen::MatrixXf embedded_tokens(tokens.size(), weights.token_embedding.cols());

    for (size_t i = 0; i < tokens.size(); ++i) {
        // Check if the token ID is within the valid range
        if (tokens[i] >= 0 && tokens[i] < weights.token_embedding.rows()) {
            embedded_tokens.row(i) = weights.token_embedding.row(tokens[i]);
            embedded_tokens.row(i) += weights.position_embedding.row(i);
        } else {
            die("Invalid token ID: " + std::to_string(tokens[i]));
        }
    }

    Eigen::MatrixXf transformer_output = transformer.forward(embedded_tokens);

    MatrixXf norm_final_output = final_norm_layer.forward(transformer_output);

    MatrixXf logits = norm_final_output * weights.token_embedding.transpose();

    return logits;
}

string_t gpt2_t::get_next_max_like_token(MatrixXf& logits)
{
    Eigen::VectorXf last_token_logits = logits.row(logits.rows() - 1);

    Eigen::VectorXf probabilities = softmax(last_token_logits);

    Eigen::Index max_index;
    probabilities.maxCoeff(&max_index);
    int max_prob_token_id = static_cast<int>(max_index);
    string_t token = tokenizer.detokenize({max_prob_token_id})[0];

    return token;
}
