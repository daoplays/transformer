#include "multi_head_attention.h"

MatrixXf multi_head_attention_t::forward(const MatrixXf& X)
{
    int seq_len = X.rows();

    // Compute Q, K, V for all heads at once
    MatrixXf Q = (X * query_weights.transpose()).rowwise() + query_bias.transpose();
    MatrixXf K = (X * key_weights.transpose()).rowwise() + key_bias.transpose();
    MatrixXf V = (X * value_weights.transpose()).rowwise() + value_bias.transpose();

    // Split Q, K, V for each head
    std::vector<MatrixXf> Q_heads, K_heads, V_heads;
    for (int i = 0; i < num_heads; ++i) {
        Q_heads.push_back(Q.block(0, i * d_k, seq_len, d_k));
        K_heads.push_back(K.block(0, i * d_k, seq_len, d_k));
        V_heads.push_back(V.block(0, i * d_k, seq_len, d_k));
    }

    // Process each head
    std::vector<MatrixXf> head_outputs;
    for (int i = 0; i < num_heads; ++i) {
        MatrixXf head_output = attention_head.forward(Q_heads[i], K_heads[i], V_heads[i]);

        head_outputs.push_back(head_output);
    }

    // Concatenate head outputs
    MatrixXf concatenated_output(seq_len, num_heads * d_k);
    for (int i = 0; i < num_heads; ++i) {
        concatenated_output.block(0, i * d_k, seq_len, d_k) = head_outputs[i];
    }

    // Final output projection
    return (concatenated_output * output_projection.transpose()).rowwise() + output_bias.transpose();
}