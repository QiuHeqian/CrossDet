#include "pytorch_cpp_helper.hpp"

void cross_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax, Tensor argmax_y,
                       float spatial_scale, int pool_mode, int axis);

void cross_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax, Tensor argmax_y,
                       Tensor grad_input, float spatial_scale, int pool_mode, int axis);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("cross_pool_forward", &cross_pool_forward, "cross_pool forward",
        py::arg("input"), py::arg("rois"), py::arg("output"), py::arg("argmax"), py::arg("argmax_y"),
        py::arg("spatial_scale"), py::arg("pool_mode"), py::arg("axis"));
        
  m.def("cross_pool_backward", &cross_pool_backward, "cross_pool backward",
        py::arg("grad_output"), py::arg("rois"), py::arg("argmax"), py::arg("argmax_y"), 
        py::arg("grad_input"), py::arg("spatial_scale"), py::arg("pool_mode"), py::arg("axis"));
}
