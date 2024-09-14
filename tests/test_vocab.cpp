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

TEST_CASE("Vocabulary loader correctly loads GPT-2 vocabulary", "[vocab_loader]")
{
    // Initialize tokenizer_t with vocabulary and merges files
    tokenizer_t tokenizer("gpt2/vocab.json", "gpt2/merges.txt");

    // GPT-2 vocabulary size is 50257
    REQUIRE(tokenizer.get_vocab_size() == 50257);
    // GPT-2 mergers size is 50000
    REQUIRE(tokenizer.get_mergers_size() == 50000);

    // Check some specific tokens
    CHECK(tokenizer.tokenize("Twitter")[0] == 14254);
    CHECK(tokenizer.tokenize("Bitcoin")[0] == 22614);

    CHECK(tokenizer.tokenize("!")[0] == 0);
    CHECK(tokenizer.tokenize("~")[0] == 93);

    // Example text to tokenize
    std::string text = "GPT2 is a model developed by OpenAI";
    // Tokenize the text
    std::vector<int> tokens = tokenizer.tokenize(text);
    std::vector<string_t> detokenized = tokenizer.detokenize(tokens);
    for (size_t i = 0; i < tokens.size(); i++) {
        std::cout << tokens[i] << " " << detokenized[i] << std::endl;
    }

    // expected tokens from the hugging face python api
    std::vector<int> expected_tokens = {38, 11571, 17, 318, 257, 2746, 4166, 416, 4946, 20185};
    REQUIRE(tokens == expected_tokens);

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
    std::cout << "after embedding" << std::endl;
    std::cout << embedded_tokens.row(0).head(10) << std::endl;
    std::cout << embedded_tokens.rows() << " " << embedded_tokens.cols() << std::endl;

    int d_model = 768;
    int num_heads = 12;
    int d_ff = 3072;
    

    norm_layer_t layer_norm(d_model, 1e-5);
    layer_norm.setGammaBeta(gpt_weights.layers[0].ln_1_weight, gpt_weights.layers[0].ln_1_bias);
    MatrixXf norm1_output = layer_norm.forward(embedded_tokens);
    
    
    std::cout << "after layer norm" << std::endl;
    std::cout << norm1_output.rows() << " " << norm1_output.cols() << std::endl;
    //std::cout << norm1_output.row(0) << std::endl;
    
    multi_head_attention_t mha(d_model, num_heads);
    mha.set_weights2(gpt_weights.layers[0].attn_c_attn_weight, gpt_weights.layers[0].attn_c_attn_bias,
                    gpt_weights.layers[0].attn_c_proj_weight, gpt_weights.layers[0].attn_c_proj_bias);

    std::cout << "after attention" << std::endl;
    MatrixXf attn_output = mha.forward(norm1_output);
    std::cout << attn_output.rows() << " " << attn_output.cols() << std::endl;
    //std::cout << attn_output.row(0) << std::endl;

    std::cout << "after adding  attention" << std::endl;
    MatrixXf residual1 = embedded_tokens + attn_output;
    //std::cout <<  norm2_input.row(0) << std::endl;

    norm_layer_t layer_norm_2(d_model, 1e-5);
    layer_norm_2.setGammaBeta(gpt_weights.layers[0].ln_2_weight, gpt_weights.layers[0].ln_2_bias);

    MatrixXf norm2_output = layer_norm_2.forward(residual1);
    
    
    std::cout << "after layer norm2" << std::endl;
    std::cout << norm2_output.rows() << " " << norm2_output.cols() << std::endl;
    std::cout << norm2_output.row(0).head(10) << std::endl;

    feed_forward_t ff(d_model, d_ff);
    ff.set_weights(gpt_weights.layers[0].mlp_c_fc_weight.transpose(), gpt_weights.layers[0].mlp_c_proj_weight.transpose(),
                    gpt_weights.layers[0].mlp_c_fc_bias, gpt_weights.layers[0].mlp_c_proj_bias);

    
    MatrixXf ff_output = ff.forward(norm2_output);
    std::cout << "after ff" << std::endl;
    std::cout << ff_output.rows() << " " << ff_output.cols() << std::endl;
    std::cout << ff_output.row(0).head(10) << std::endl;


    MatrixXf residual2 = residual1 + ff_output;

    std::cout << "residual2" << std::endl;
    std::cout << residual2.rows() << " " << residual2.cols() << std::endl;
    std::cout << residual2.row(0).head(10) << std::endl;

    encoder_layer_t encoder_layer(d_model, num_heads, d_ff);
    encoder_layer.set_weights(gpt_weights.layers[0].attn_c_attn_weight, gpt_weights.layers[0].attn_c_attn_bias,
                    gpt_weights.layers[0].attn_c_proj_weight, gpt_weights.layers[0].attn_c_proj_bias,
                    gpt_weights.layers[0].ln_1_weight, gpt_weights.layers[0].ln_1_bias,
                    gpt_weights.layers[0].mlp_c_fc_weight.transpose(), gpt_weights.layers[0].mlp_c_fc_bias,
                    gpt_weights.layers[0].mlp_c_proj_weight.transpose(), gpt_weights.layers[0].mlp_c_proj_bias,
                    gpt_weights.layers[0].ln_2_weight, gpt_weights.layers[0].ln_2_bias);

    MatrixXf encoder_output = encoder_layer.forward(embedded_tokens);
    std::cout << "encoder_output" << std::endl;
    std::cout << encoder_output.rows() << " " << encoder_output.cols() << std::endl;
    std::cout << encoder_output.row(0).head(10) << std::endl;


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
    std::cout << "transformer output" << std::endl;
    std::cout << input.rows() << " " << input.cols() << std::endl;
    std::cout << input.row(0).head(10) << std::endl;

    norm_layer_t layer_norm_final(d_model, 1e-5);
    layer_norm_final.setGammaBeta(gpt_weights.ln_f_weight, gpt_weights.ln_f_bias);
    MatrixXf norm_final_output = layer_norm_final.forward(input);
    std::cout << "final layer norm" << std::endl;
    std::cout << norm_final_output.rows() << " " << norm_final_output.cols() << std::endl;
    std::cout << norm_final_output.row(0).head(10) << std::endl;

    MatrixXf logits = norm_final_output * gpt_weights.token_embedding.transpose();
    std::cout << "logits" << std::endl;
    std::cout << logits.rows() << " " << logits.cols() << std::endl;
    std::cout << logits.row(0).head(10) << std::endl;

    Eigen::VectorXf last_token_logits = logits.row(logits.rows() - 1);

    Eigen::Index max_index;
    float max_logit = last_token_logits.maxCoeff(&max_index);
    int max_prob_token_id = static_cast<int>(max_index);
    std::cout << "max prob token id: " << max_prob_token_id << std::endl;
    std::vector<string_t> max_prob_token = tokenizer.detokenize({max_prob_token_id});
    std::cout << "max prob token: " << max_prob_token[0] << std::endl;

}