#include <torch/extension.h>

torch::Tensor naive_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor sparse_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask);
torch::Tensor sparse_attention_v2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention",     &naive_attention_forward,
          "Naive dense attention, one thread per output (CUDA)");
    m.def("sparse_attention",    &sparse_attention_forward,
          "Sparse Flash Attention v1 - tiled, full score row (CUDA)");
    m.def("sparse_attention_v2", &sparse_attention_v2_forward,
          "Sparse Flash Attention v2 - Online Softmax (CUDA)");
}
