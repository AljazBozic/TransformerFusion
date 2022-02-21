#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "cpu/metrics.h"
#include "cuda/metrics_gpu.h"

// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chamfer_distance_forward", &metrics::chamfer_distance_forward, "Forward pass of chamfer distance");
    m.def("chamfer_distance_backward", &metrics::chamfer_distance_backward, "Backward pass of chamfer distance");
    m.def("chamfer_distance_forward_gpu", &metrics_gpu::chamfer_distance_forward_cuda, "Forward pass of chamfer distance (CUDA)");
    m.def("chamfer_distance_backward_gpu", &metrics_gpu::chamfer_distance_backward_cuda, "Backward pass of chamfer distance (CUDA)");
}