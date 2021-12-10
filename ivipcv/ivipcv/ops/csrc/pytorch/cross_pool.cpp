#include "pytorch_cpp_helper.hpp"

#ifdef WITH_CUDA
// CUDA Kernel Launcher 函数声明，函数定义位于cross_pool_cuda.cu
void CrossPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax, Tensor argmax_y, float spatial_scale, int pool_mode, int axis);

void CrossPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois, Tensor argmax, Tensor argmax_y, Tensor grad_input,
                                        float spatial_scale, int pool_mode, int axis);


void cross_pool_forward_cuda(Tensor input, Tensor rois, Tensor output,
                           Tensor argmax, Tensor argmax_y, float spatial_scale, int pool_mode, int axis) {
  CrossPoolForwardCUDAKernelLauncher(input, rois, output, argmax, argmax_y, spatial_scale, pool_mode, axis);
}

void cross_pool_backward_cuda(Tensor grad_output, Tensor rois, Tensor argmax, Tensor argmax_y,
                            Tensor grad_input, float spatial_scale, int pool_mode, int axis) {
  CrossPoolBackwardCUDAKernelLauncher(grad_output, rois, argmax, argmax_y, grad_input, spatial_scale, pool_mode, axis);
}
#endif

void cross_pool_forward(Tensor input, Tensor rois, Tensor output, Tensor argmax, Tensor argmax_y,
                      float spatial_scale, int pool_mode, int axis) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(argmax);

    cross_pool_forward_cuda(input, rois, output, argmax, argmax_y, spatial_scale, pool_mode, axis);
#else
    AT_ERROR("CrossPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CrossPool is not implemented on CPU");
  }
}

void cross_pool_backward(Tensor grad_output, Tensor rois, Tensor argmax, Tensor argmax_y,
                       Tensor grad_input, float spatial_scale, int pool_mode, int axis) {
  if (grad_output.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA_INPUT(grad_output);
    CHECK_CUDA_INPUT(rois);
    CHECK_CUDA_INPUT(argmax);
    CHECK_CUDA_INPUT(grad_input);

    cross_pool_backward_cuda(grad_output, rois, argmax, argmax_y, grad_input, spatial_scale, pool_mode, axis);
#else
    AT_ERROR("CrossPool is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("CrossPool is not implemented on CPU");
  }
}
