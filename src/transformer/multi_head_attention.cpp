#include "multi_head_attention.h" 


MatrixXf multi_head_attention_t::forward(const MatrixXf &X) {
    int seq_len = X.rows();

    // Split input for each head
    std::vector<MatrixXf> head_inputs;
    for (int i = 0; i < num_heads; ++i) {
        head_inputs.push_back(X.block(0, i * d_k, seq_len, d_k));
    }

    // Process each head
    std::vector<MatrixXf> head_outputs;
    for (int i = 0; i < num_heads; ++i) {
        head_outputs.push_back(attention_heads[i].forward(head_inputs[i]));
    }

    // Concatenate head outputs
    MatrixXf concatenated_output(seq_len, d_model);
    for (int i = 0; i < num_heads; ++i) {
        concatenated_output.block(0, i * d_k, seq_len, d_k) = head_outputs[i];
    }

    // Final output projection
    return concatenated_output * output_projection;
}