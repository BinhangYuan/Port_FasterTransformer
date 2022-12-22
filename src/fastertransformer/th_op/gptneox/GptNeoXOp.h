#define _DEBUG_PRINT_GPTNEOX
#include <iostream>
#include "src/fastertransformer/models/gptneox/GptNeoX.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/stream_tokens_pipe.h"


namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGptNeoX {
public:
   virtual ~IFGptNeoX() {}
   virtual void forward(th::Tensor&              input_ids,
                        th::Tensor&              input_lengths,
                        th::Tensor&              output_ids,
                        th::Tensor&              sequence_lengths,
                        th::Tensor&              cum_log_probs,
                        const size_t             request_output_len,
                        const size_t             beam_width,
                        th::optional<th::Tensor> top_k_opt,
                        th::optional<th::Tensor> top_p_opt,
                        th::optional<th::Tensor> beam_search_diversity_rate_opt,
                        th::optional<th::Tensor> temperature_opt,
                        th::optional<th::Tensor> len_penalty_opt,
                        th::optional<th::Tensor> repetition_penalty_opt,
                        th::optional<th::Tensor> random_seed_opt,
                        th::optional<int64_t>    request_id,
                        th::optional<int64_t>    stream_tokens_pipe) = 0;
};

template<typename T>
class FTGptNeoX: public IFGptNeoX {
public:
   FTGptNeoX(const size_t               head_num,
             const size_t               size_per_head,
             const size_t               inter_size,
             const size_t               layer_num,
             const size_t               vocab_size,
             const size_t               rotary_embedding_dim,
             const int                  start_id,
             const int                  end_id,
             const int                  tensor_para_size,
             const int                  pipeline_para_size,
             const bool                 use_gptj_residual,
             const vector<th::Tensor>   weights):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        rotary_embedding_dim_(rotary_embedding_dim),
        start_id_(start_id),
        end_id_(end_id),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        use_gptj_residual_(use_gptj_residual),
        weights_(weights),
        gptneox_weights_()
   {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap("gemm_config.in");
        cublas_wrapper_mutex_ = new std::mutex();
#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++"  << std::endl;    
        std::cout << "IFGptNeoX-IFGptNeoX: initialized variables: "  << std::endl;
        std::cout << "head_num_: " << head_num_ << std::endl;
        std::cout << "size_per_head_: " << size_per_head_ << std::endl;
        std::cout << "inter_size_: " << inter_size_ << std::endl;
        std::cout << "layer_num_: " << layer_num_ << std::endl;
        std::cout << "vocab_size_: " << vocab_size_ << std::endl;
        std::cout << "rotary_embedding_dim_: " << rotary_embedding_dim_ << std::endl;
        std::cout<< "use_gptj_residual_: " << use_gptj_residual_ << std::endl;
        std::cout << "weights_ lens: " << weights_.size() << std::endl;
        std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++"  << std::endl;    
#endif

       ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size);

#ifdef _DEBUG_PRINT_GPTNEOX 
        std::cout << "IFGptNeoX-IFGptNeoX: " << " ftNcclInitialize. (done)" << std::endl;
#endif

#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++"  << std::endl;   
        std::cout << "IFGptNeoX-IFGptNeoX: gptneox_weights_ variables: "  << std::endl;
        std::cout << "gptneox_weights_.decoder_layer_weights lens: " << gptneox_weights_.decoder_layer_weights.size()  << std::endl;
        std::cout << "gptneox_weights_ decoder_layer_weights post_decoder_layernorm: " << gptneox_weights_.post_decoder_layernorm.gamma  << std::endl;
        std::cout << "gptneox_weights_ decoder_layer_weights post_decoder_embedding: " << gptneox_weights_.post_decoder_embedding.kernel  << std::endl;
#endif

        gptneox_weights_.decoder_layer_weights.reserve(layer_num_);
        for (int i = 0;  i< (int) layer_num_; i++){
            if(isValidLayerParallelIndex(i)){
                //gptneox_weights_.decoder_layer_weights.push_back(new ft::GptNeoXDecoderLayerWeight<T> (
                //    size_per_head_*head_num_, inter_size_, tensor_para_size_, tensor_para_.rank_, use_gptj_residual_));     
                gptneox_weights_.decoder_layer_weights.push_back(new ft::GptNeoXDecoderLayerWeight<T> (
                    0, 0, tensor_para_size_, tensor_para_.rank_, use_gptj_residual_));   
            }
            else{
                gptneox_weights_.decoder_layer_weights.push_back(new ft::GptNeoXDecoderLayerWeight<T>(0, 0));
            }
        }

        for (int i=0; i< (int) layer_num_; i++){
#ifdef _DEBUG_PRINT_GPTNEOX
            std::cout << "IFGptNeoX-IFGptNeoX: set gptneox_weights_ variable: decoder_layer_weights layer <" << i <<">. "   << std::endl;
#endif
            gptneox_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
               get_ptr<T>(weights_[i + 0 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
               get_ptr<T>(weights_[i + 1 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
               get_ptr<T>(weights_[i + 2 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
               get_ptr<T>(weights_[i + 3 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
               get_ptr<T>(weights_[i + 4 * layer_num_]);
            if (!use_gptj_residual_){
                gptneox_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                    get_ptr<T>(weights_[i + 5 * layer_num_]);
            }
            gptneox_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
               get_ptr<T>(weights_[i + 6 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
               get_ptr<T>(weights_[i + 7 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
               get_ptr<T>(weights_[i + 8 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
               get_ptr<T>(weights_[i + 9 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.gamma =
               get_ptr<T>(weights_[i + 10 * layer_num_]);
            gptneox_weights_.decoder_layer_weights[i]->post_attention_layernorm_weights.beta =
               get_ptr<T>(weights_[i + 11 * layer_num_]);
        }

#ifdef _DEBUG_PRINT_GPTNEOX
            std::cout << "IFGptneox-IFGptneox: set gptneox_weights_ pre_decoder_embedding_table."  << std::endl;
#endif
        gptneox_weights_.pre_decoder_embedding_table = get_ptr<T>(weights_[12 * layer_num_ + 0]);

#ifdef _DEBUG_PRINT_GPTNEOX
            std::cout << "IFGptneox-IFGptneox: set gptneox_weights_ post_decoder_layernorm."  << std::endl;
#endif
        gptneox_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 1]);
        gptneox_weights_.post_decoder_layernorm.beta  = get_ptr<T>(weights_[12 * layer_num_ + 2]);
#ifdef _DEBUG_PRINT_GPTNEOX
            std::cout << "IFGptneox-IFGptneox: set gptneox_weights_ post_decoder_embedding."  << std::endl;
#endif
        gptneox_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 3]);

#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "IFGptneox-IFGptneox: " << " gptneox_weights_ loaded" << std::endl;
#endif

        int device_id = 0;
        ft::check_cuda_error(cudaGetDevice(&device_id));
        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
        FT_LOG_INFO("Device %s", prop_.name);
   }

   ~FTGptNeoX() override
   {
       ft::ftNcclParamDestroy(tensor_para_);
       ft::ftNcclParamDestroy(pipeline_para_);
       cublasLtDestroy(cublasltHandle_);
       delete cublas_algo_map_;
       delete cublas_wrapper_mutex_;
   }

   void forward(th::Tensor&              input_ids,
                th::Tensor&              input_lengths,
                th::Tensor&              output_ids,
                th::Tensor&              sequence_lengths,
                th::Tensor&              cum_log_probs,
                const size_t             request_output_len,
                const size_t             beam_width,
                th::optional<th::Tensor> top_k_opt,
                th::optional<th::Tensor> top_p_opt,
                th::optional<th::Tensor> beam_search_diversity_rate_opt,
                th::optional<th::Tensor> temperature_opt,
                th::optional<th::Tensor> len_penalty_opt,
                th::optional<th::Tensor> repetition_penalty_opt,
                th::optional<th::Tensor> random_seed_opt,
                th::optional<int64_t>    request_id,
                th::optional<int64_t>    stream_tokens_pipe) override
   {
#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "IFGptNeox-forward: starts." << std::endl;
#endif
       auto stream                 = at::cuda::getCurrentCUDAStream().stream();
       cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
       cublasSetStream(cublasHandle, stream);
       ft::Allocator<ft::AllocatorType::TH> allocator      = ft::Allocator<ft::AllocatorType::TH>();
       ft::cublasMMWrapper                  cublas_wrapper = ft::cublasMMWrapper(
           cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, &allocator);

       if (std::is_same<T, half>::value) {
           cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
       }
#ifdef ENABLE_BF16
       else if (std::is_same<T, __nv_bfloat16>::value) {
           cublas_wrapper.setBF16GemmConfig();
       }
#endif
       else if (std::is_same<T, float>::value) {
           cublas_wrapper.setFP32GemmConfig();
       }

       const size_t request_batch_size = (size_t)input_ids.size(0);
       const size_t max_input_length   = (size_t)input_ids.size(1);
       const int    total_output_len   = (int)(max_input_length + request_output_len);

#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "IFGptNeoX-forward: start to create ft::GptNeoX<T>." << std::endl;
#endif
       ft::GptNeoX<T> gptneox = ft::GptNeoX<T>(head_num_,
                                               size_per_head_,
                                               inter_size_,
                                               layer_num_,
                                               vocab_size_,
                                               rotary_embedding_dim_,
                                               start_id_,
                                               end_id_,
                                               0,
                                               ft::PromptLearningType::no_prompt,
                                               use_gptj_residual_,
                                               0.0f,
                                               1,
                                               0.0f,
                                               0,
                                               1.0f,
                                               1.0f,
                                               1.0f,
                                               tensor_para_,
                                               pipeline_para_,
                                               stream,
                                               &cublas_wrapper,
                                               &allocator,
                                               false,
                                               &prop_);

#ifdef _DEBUG_PRINT_GPTNEOX
        std::cout << "IFGptNeoX-forward: created ft::GptNeoX<T>." << std::endl;
#endif

       std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);

       std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
           {"input_ids",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{request_batch_size, max_input_length},
                       get_ptr<int>(input_ids)}},
           {"input_lengths",
            ft::Tensor{
                ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
           {"output_seq_len",
            ft::Tensor{
                ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};
       if (beam_width > 1 && beam_search_diversity_rate_opt.has_value()) {
           input_tensors.insert(
               {"beam_search_diversity_rate",
                convert_tensor<float>(beam_search_diversity_rate_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (top_p_opt.has_value()) {
           input_tensors.insert(
               {"runtime_top_p", convert_tensor<float>(top_p_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (top_k_opt.has_value()) {
           input_tensors.insert(
               {"runtime_top_k", convert_tensor<uint>(top_k_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (temperature_opt.has_value()) {
           input_tensors.insert(
               {"temperature", convert_tensor<float>(temperature_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (len_penalty_opt.has_value()) {
           input_tensors.insert(
               {"len_penalty", convert_tensor<float>(len_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (repetition_penalty_opt.has_value()) {
           input_tensors.insert({"repetition_penalty",
                                 convert_tensor<float>(repetition_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }
       if (random_seed_opt.has_value()) {
           input_tensors.insert(
               {"random_seed",
                convert_tensor<unsigned long long int>(random_seed_opt.value(), ft::MemoryType::MEMORY_CPU)});
       }

       int pipe_fd = stream_tokens_pipe.has_value() ? stream_tokens_pipe.value() : -1;
       if (pipe_fd >= 0) {
           this->stream_tokens_pipe_.reset(
               new ft::TokenPipe(pipe_fd, request_id.has_value() ? request_id.value() : 0));
           gptneox.registerCallback(&ft::TokenPipe::stream_tokens_callback, this->stream_tokens_pipe_.get());
       } else {
           this->stream_tokens_pipe_.reset(nullptr);
       }

       std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
           {"output_ids",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                       get_ptr<int>(output_ids)}},
           {"sequence_length",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_INT32,
                       std::vector<size_t>{request_batch_size, beam_width},
                       get_ptr<int>(sequence_lengths)}},
           {"output_log_probs",
            ft::Tensor{ft::MEMORY_GPU,
                       ft::TYPE_FP32,
                       std::vector<size_t>{(size_t)request_output_len, request_batch_size, beam_width},
                       get_ptr<float>(cum_log_probs)}}};

       try {
           gptneox.forward(&output_tensors, &input_tensors, &gptneox_weights_);
       }
       catch (const std::runtime_error& error) {
           FT_LOG_ERROR(error.what());
           ft::FT_CHECK(false);
       }
       catch (const std::exception& error) {
           FT_LOG_ERROR(error.what());
           ft::FT_CHECK(false);
       }
       catch (...) {
           FT_LOG_ERROR("Unknown error");
           ft::FT_CHECK(false);
       }
   }

private:
   const size_t head_num_;
   const size_t size_per_head_;
   const size_t inter_size_;
   const size_t layer_num_;
   const size_t vocab_size_;
   const size_t rotary_embedding_dim_;
   const int    start_id_;
   const int    end_id_;
   const bool   use_gptj_residual_;

   size_t tensor_para_size_;
   size_t pipeline_para_size_;

   std::vector<th::Tensor> weights_;

   ft::NcclParam tensor_para_;
   ft::NcclParam pipeline_para_;

   cublasLtHandle_t         cublasltHandle_;
   std::mutex*              cublas_wrapper_mutex_;
   ft::cublasAlgoMap*       cublas_algo_map_;
   struct cudaDeviceProp    prop_;
   ft::GptNeoXWeight<T>     gptneox_weights_;
   int                      world_size_ = 1;
   int                      rank_       = 0;

   std::unique_ptr<ft::TokenPipe> stream_tokens_pipe_;


    bool isValidLayerParallelIndex(int l)
    {
        int local_num_layer = (int)(ceil(layer_num_ * 1.0f / pipeline_para_size_));
        return l < layer_num_ && (l >= local_num_layer * pipeline_para_.rank_) && (l < local_num_layer * (pipeline_para_.rank_ + 1));
    }
};

class GptNeoXOp: public th::jit::CustomClassHolder {
public:
    GptNeoXOp(const int64_t            head_num,
              const int64_t            size_per_head,
              const int64_t            inter_size,
              const int64_t            layer_num,
              const int64_t            vocab_size,
              const int64_t            rotary_embedding_dim,
              const int64_t            start_id,
              const int64_t            end_id,
              const int64_t            tensor_para_size,
              const int64_t            pipeline_para_size,
              const int64_t            use_gptj_residual,
              const vector<th::Tensor> weights);

   ~GptNeoXOp();

   vector<th::Tensor> forward(th::Tensor               input_ids,
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
                              th::optional<int64_t>    stream_tokens_pipe);

private:
   const at::ScalarType    st_;
   IFGptNeoX*                 ftgptneox;
   std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
