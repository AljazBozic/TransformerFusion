#pragma once

#include <iostream>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace metrics {

    void chamfer_distance_forward(
        const at::Tensor xyz1, 
        const at::Tensor xyz2, 
        const at::Tensor dist1, 
        const at::Tensor dist2, 
        const at::Tensor idx1, 
        const at::Tensor idx2
    );

    void chamfer_distance_backward(
        const at::Tensor xyz1, 
        const at::Tensor xyz2, 
        at::Tensor gradxyz1, 
        at::Tensor gradxyz2, 
        at::Tensor graddist1, 
        at::Tensor graddist2, 
        at::Tensor idx1, 
        at::Tensor idx2
    );

} // namespace metrics
