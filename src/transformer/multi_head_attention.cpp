#include "multi_head_attention.h"

MatrixXf multi_head_attention_t::forward(const MatrixXf& X)
{
    int seq_len = X.rows();

    // Compute Q, K, V for all heads at once
    MatrixXf Q = (X * query_weights.transpose()).rowwise() + query_bias.transpose();
    MatrixXf K = (X * key_weights.transpose()).rowwise() + key_bias.transpose();
    MatrixXf V = (X * value_weights.transpose()).rowwise() + value_bias.transpose();

    
    std::cout << "X:\n" <<  X.row(0).head(10) << std::endl;
    std::cout << "Q:\n" <<  Q.row(0).head(10) << std::endl;
    std::cout << "K: \n" << K.row(0).head(10) << std::endl;
    std::cout << "V: \n" << V.row(0).head(10) << std::endl;

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
        std::cout << "head : " << i << " " << head_output.rows() << " " << head_output.cols()<< std::endl;

        head_outputs.push_back(head_output);
    }

    std::cout << "have run heads" << std::endl;
    // Concatenate head outputs
    MatrixXf concatenated_output(seq_len, num_heads * d_k);
    for (int i = 0; i < num_heads; ++i) {
        concatenated_output.block(0, i * d_k, seq_len, d_k) = head_outputs[i];
    }

    // Final output projection
    return (concatenated_output * output_projection.transpose()).rowwise() + output_bias.transpose();
}

MatrixXf multi_head_attention_t::forward2(const MatrixXf& X)
{
    int seq_len = X.rows();
    //std::cout << "X2:\n" <<  X.rows() << " " << X.cols() << std::endl;
    //std::cout << "X2:\n" <<  X.row(0).head(10) << std::endl;


    //std::cout << "bias " << qkv_bias.size() << std::endl;
    //std::cout << qkv_bias.head(10) << std::endl;
    //std::cout << "weights " << qkv_weights.rows() << " " << qkv_weights.cols() << std::endl;
    //std::cout << qkv_weights.row(0).head(10) << std::endl;

    // Compute Q, K, V for all heads at once
    MatrixXf QKV = (X * qkv_weights).rowwise() + qkv_bias.transpose();


    //std::cout << "QKV " << QKV.rows() << " " << QKV.cols() << std::endl;
    //std::cout << QKV.row(0).head(10) << std::endl;

    Eigen::MatrixXf Q = QKV.leftCols(d_model);
    Eigen::MatrixXf K = QKV.middleCols(d_model, d_model);
    Eigen::MatrixXf V = QKV.rightCols(d_model);

    
    //std::cout << "Q2:\n" <<  Q.row(0).head(10) << std::endl;
    //std::cout << "K2: \n" << K.row(0).head(10) << std::endl;
    //std::cout << "V2: \n" << V.row(0).head(10) << std::endl;

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
        //std::cout << "head : " << i << " " << head_output.rows() << " " << head_output.cols()<< std::endl;
        //std::cout << head_output.row(0).head(10) << std::endl;
        head_outputs.push_back(head_output);
    }

    // Concatenate head outputs
    MatrixXf concatenated_output(seq_len, num_heads * d_k);
    for (int i = 0; i < num_heads; ++i) {
        concatenated_output.block(0, i * d_k, seq_len, d_k) = head_outputs[i];
    }

    // Final output projection
    return (concatenated_output * output_projection).rowwise() + output_bias.transpose();
}