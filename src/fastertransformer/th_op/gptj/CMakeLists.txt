add_library(th_gptj SHARED GptjOp.cc)
target_link_libraries(th_gptj PRIVATE "${TORCH_LIBRARIES}" GptJ th_utils nccl_utils)
