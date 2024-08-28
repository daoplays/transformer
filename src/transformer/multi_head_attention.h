#include "../eigen_config.h"
#include <random>

// Multi-Head Attention class
// This is the core of the transformer architecture
class multi_head_attention_t
{
private:
    int d_model, num_heads;
    MatrixXf W_q, W_k, W_v, W_o;

public:
    multi_head_attention_t(int d_model, int num_heads) : d_model(d_model), num_heads(num_heads)
    {
        // Initialize weights
        // In practice, these weight matrices allow the model to project
        // the input into different subspaces for each attention head
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.02);

        W_q = MatrixXf::NullaryExpr(d_model, d_model, [&]()
                                    { return d(gen); });
        W_k = MatrixXf::NullaryExpr(d_model, d_model, [&]()
                                    { return d(gen); });
        W_v = MatrixXf::NullaryExpr(d_model, d_model, [&]()
                                    { return d(gen); });
        W_o = MatrixXf::NullaryExpr(d_model, d_model, [&]()
                                    { return d(gen); });
    }

    MatrixXf forward(const MatrixXf &X);
};