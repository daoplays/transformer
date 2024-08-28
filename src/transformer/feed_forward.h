#include "../eigen_config.h"
#include <vector>
#include <cmath>
#include <random>
// Feed-Forward Network class
// This adds non-linearity and increases the model's capacity
class feed_forward_t
{
private:
    MatrixXf W1, W2;
    int d_model, d_ff;

public:
    feed_forward_t(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff)
    {
        // Initialize weights
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.02);

        W1 = MatrixXf::NullaryExpr(d_model, d_ff, [&]()
                                   { return d(gen); });
        W2 = MatrixXf::NullaryExpr(d_ff, d_model, [&]()
                                   { return d(gen); });
    }

    MatrixXf forward(const MatrixXf &X);
};