#include "src/fastertransformer/th_op/gptj/GptjOp.h"


namespace th = torch;
namespace ft = fastertransformer;
namespace torch_ext {

GptjOp::GptjOp(const int64_t                 head_num,
               const int64_t                 size_per_head,
               const int64_t                 inter_size,
               const int64_t                 layer_num,
               const int64_t                 vocab_size,
               const int64_t                 rotary_embedding_dim,
               const int64_t                 start_id,
               const int64_t                 end_id,
               const int64_t                 tensor_para_size,
               const int64_t                 pipeline_para_size,
               const std::vector<th::Tensor> weights):
   st_(weights[0].scalar_type())
{
   for (auto t : weights) {
       CHECK_INPUT(t, st_);
   }

   switch (st_) {
       case at::ScalarType::Float:
#ifdef _DEBUG_PRINT_GPTJ
            std::cout << "GptjOp-GptjOp: created ft::FTGptj<float>." << std::endl;
#endif
           ftgptj = new FTGptj<float>((size_t)head_num,
                                      (size_t)size_per_head,
                                      (size_t)inter_size,
                                      (size_t)layer_num,
                                      (size_t)vocab_size,
                                      (size_t)rotary_embedding_dim,
                                      start_id,
                                      end_id,
                                      tensor_para_size,
                                      pipeline_para_size,
                                      weights);
           break;
       case at::ScalarType::Half:
#ifdef _DEBUG_PRINT_GPTJ
            std::cout << "GptjOp-GptjOp: created ft::FTGptj<half>." << std::endl;
#endif
           ftgptj = new FTGptj<half>((size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     (size_t)vocab_size,
                                     (size_t)rotary_embedding_dim,
                                     start_id,
                                     end_id,
                                     tensor_para_size,
                                     pipeline_para_size,
                                     weights);
           break;
#ifdef ENABLE_BF16
#ifdef _DEBUG_PRINT_GPTJ
            std::cout << "GptjOp-GptjOp: created ft::FTGptj<__nv_bfloat16>." << std::endl;
#endif
       case at::ScalarType::BFloat16:
           ftgptj = new FTGptj<__nv_bfloat16>((size_t)head_num,
                                              (size_t)size_per_head,
                                              (size_t)inter_size,
                                              (size_t)layer_num,
                                              (size_t)vocab_size,
                                              (size_t)rotary_embedding_dim,
                                              start_id,
                                              end_id,
                                              tensor_para_size,
                                              pipeline_para_size,
                                              weights);
           break;
#endif
       default:
           throw std::runtime_error("Wrong Tensor type.");
   }
}

GptjOp::~GptjOp()
{
   delete ftgptj;
}

std::vector<th::Tensor> GptjOp::forward(th::Tensor               input_ids,
                                        th::Tensor               input_lengths,
                                        const int64_t            output_len,
                                        th::optional<int64_t>    beam_width_opt,
                                        th::optional<th::Tensor> top_k_opt,
                                        th::optional<th::Tensor> top_p_opt,
                                        th::optional<th::Tensor> beam_search_diversity_rate_opt,
                                        th::optional<th::Tensor> temperature_opt,
                                        th::optional<th::Tensor> len_penalty_opt,
                                        th::optional<th::Tensor> repetition_penalty_opt,
                                        th::optional<th::Tensor> random_seed_opt,
                                        th::optional<int64_t>    request_id,
                                        th::optional<int64_t>    stream_tokens_pipe)
{
   CHECK_TH_CUDA(input_ids);
   CHECK_CONTIGUOUS(input_ids);
   TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
   CHECK_TH_CUDA(input_lengths);
   CHECK_CONTIGUOUS(input_lengths);
   TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");

   const int beam_width = beam_width_opt.has_value() ? (int)beam_width_opt.value() : 1;

   const int  batch_size               = input_ids.size(0);
   const int  max_input_length         = input_ids.size(1);
   const int  total_request_output_len = max_input_length + output_len;
   th::Tensor output_ids               = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
   th::Tensor sequence_lengths =
       torch::empty({batch_size, beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

   th::Tensor cum_log_probs =
       torch::empty({batch_size, beam_width}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

   ftgptj->forward(input_ids,
                   input_lengths,
                   output_ids,
                   sequence_lengths,
                   cum_log_probs,
                   (const size_t)output_len,
                   (const size_t)beam_width,
                   top_k_opt,
                   top_p_opt,
                   beam_search_diversity_rate_opt,
                   temperature_opt,
                   len_penalty_opt,
                   repetition_penalty_opt,
                   random_seed_opt,
                   request_id,
                   stream_tokens_pipe);
   return std::vector<th::Tensor>{output_ids, sequence_lengths, cum_log_probs};
}

}  // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
   torch::jit::class_<torch_ext::GptjOp>("GptjOp")
#else
   torch::jit::class_<torch_ext::GptjOp>("FasterTransformer", "GptjOp")
#endif
       .def(torch::jit::init<int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             int64_t,
                             std::vector<th::Tensor>>())
       .def("forward", &torch_ext::GptjOp::forward);
