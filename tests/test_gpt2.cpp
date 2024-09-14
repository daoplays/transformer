#include <catch2/catch_all.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../src/load_h5.h"
#include "../src/utils.h"
#include "../src/vocab.h"
#include "../src/transformer/multi_head_attention.h"
#include "../src/transformer/encoder_layer.h"
#include "../src/transformer/norm_layer.h"  // Include your LayerNorm class definition here
#include "test_utils.h"

TEST_CASE("Vocabulary loader correctly loads GPT-2 vocabulary", "[gpt2]")
{
    // Initialize tokenizer_t with vocabulary and merges files
    tokenizer_t tokenizer("gpt2/vocab.json", "gpt2/merges.txt");

    // Example text to tokenize
    std::string text = "GPT2 is a model developed by OpenAI";
    // Tokenize the text
    std::vector<int> tokens = tokenizer.tokenize(text);

    gpt2_weights_t gpt_weights = load_gpt2_weights("gpt2/tf_model.h5");

    REQUIRE(gpt_weights.token_embedding.rows() == 50257);
    REQUIRE(gpt_weights.token_embedding.cols() == 768);
    REQUIRE(gpt_weights.position_embedding.rows() == 1024);
    REQUIRE(gpt_weights.position_embedding.cols() == 768);

    Eigen::MatrixXf embedded_tokens(tokens.size(), gpt_weights.token_embedding.cols());

    for (size_t i = 0; i < tokens.size(); ++i) {
        // Check if the token ID is within the valid range
        if (tokens[i] >= 0 && tokens[i] < gpt_weights.token_embedding.rows()) {
            embedded_tokens.row(i) = gpt_weights.token_embedding.row(tokens[i]);
            embedded_tokens.row(i) += gpt_weights.position_embedding.row(i);
        } else {
            die("Invalid token ID: " + std::to_string(tokens[i]));
        }
    }
   
    int d_model = 768;
    int num_heads = 12;
    int d_ff = 3072;
    int seq_length = 10;
    int vocab_size = 50257;
    
    MatrixXf input = embedded_tokens;
    for (int i = 0; i < 12; i++) {
        encoder_layer_t encoder_layer(d_model, num_heads, d_ff);
        encoder_layer.set_weights(gpt_weights.layers[i].attn_c_attn_weight, gpt_weights.layers[i].attn_c_attn_bias,
                    gpt_weights.layers[i].attn_c_proj_weight, gpt_weights.layers[i].attn_c_proj_bias,
                    gpt_weights.layers[i].ln_1_weight, gpt_weights.layers[i].ln_1_bias,
                    gpt_weights.layers[i].mlp_c_fc_weight.transpose(), gpt_weights.layers[i].mlp_c_fc_bias,
                    gpt_weights.layers[i].mlp_c_proj_weight.transpose(), gpt_weights.layers[i].mlp_c_proj_bias,
                    gpt_weights.layers[i].ln_2_weight, gpt_weights.layers[i].ln_2_bias);

        input = encoder_layer.forward(input);
    }
    
    norm_layer_t layer_norm_final(d_model, 1e-5);
    layer_norm_final.setGammaBeta(gpt_weights.ln_f_weight, gpt_weights.ln_f_bias);
    MatrixXf norm_final_output = layer_norm_final.forward(input);
    

    MatrixXf logits = norm_final_output * gpt_weights.token_embedding.transpose();

    MatrixXf expected_logits = readMatrixFromFile("tests/test_data/gpt2/gpt2_output.txt", seq_length, vocab_size);

    REQUIRE(matrices_approx_equal(logits, expected_logits, 1e-2));

   

    Eigen::VectorXf last_token_logits = logits.row(logits.rows() - 1);

    Eigen::Index max_index;
    last_token_logits.maxCoeff(&max_index);
    int max_prob_token_id = static_cast<int>(max_index);
    
    REQUIRE(max_prob_token_id == 284);


}