#include "pytorch_cuda_helper.hpp"
#include "composite_cross_pool_cuda_kernel.cuh"
#include "stdio.h"


void CrossPoolForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output, Tensor argmax, Tensor argmax_y, 
                                        float spatial_scale, int pool_mode, int axis) 
{
  int output_size = output.numel();
  int input_size = input.numel();
  int batch_size = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "composite_cross_pool_forward_cuda_kernel", [&] {
      composite_cross_pool_forward_cuda_kernel<scalar_t>
    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, pool_mode, axis, input.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            argmax.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(), static_cast<scalar_t>(spatial_scale), 
            batch_size, channels, height, width);
    });

  
  AT_CUDA_CHECK(cudaGetLastError());
}

void CrossPoolBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois, Tensor argmax,Tensor argmax_y, Tensor grad_input,
                                        float spatial_scale, int pool_mode, int axis) {
  int output_size = grad_output.numel(); //< 对每个输入
  int batch_size = grad_input.size(0);
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  at::cuda::CUDAGuard device_guard(grad_output.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_input.scalar_type(), "composite_cross_pool_backward_cuda_kernel", [&] {
      composite_cross_pool_backward_cuda_kernel<scalar_t>
    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            output_size, pool_mode, axis, grad_output.data_ptr<scalar_t>(),
            rois.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            argmax.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(), static_cast<scalar_t>(spatial_scale), 
            batch_size, channels, height, width);
    });

  AT_CUDA_CHECK(cudaGetLastError());
}
