#ifndef COMMON_CUDA_HELPER
#define COMMON_CUDA_HELPER

#include <cuda.h>
#include "stdio.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(const T* input, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  //< 对x,y进行clip
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  //< 初始化四个点的坐标
  int y_low = (int)y; //< 这里直接转int是将小数部分舍弃
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) { // 如果y_low是特征谱的最后一格
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else { // 一般情况，y_low不到顶格
    y_high = y_low + 1; // 有不等关系 y_low <= y <= y_high
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low; // 原点与y_low的距离
  T lx = x - x_low; // 原点与x_low的距离
  T hy = 1. - ly, hx = 1. - lx; // 原点(x ,y)与(x_high, y_high)的距离
  // do bilinear interpolation
  T v1 = input[y_low * width + x_low]; // value1 (x_low, y_low)
  T v2 = input[y_low * width + x_high]; // value2 (x_high, y_low)
  T v3 = input[y_high * width + x_low]; // value3 (x_low, y_high)
  T v4 = input[y_high * width + x_high]; // value4 (x_high, y_high)

  // printf("B\n");
  //< w1~w4是v1~v4的权重，离v1越近w1越大。
  //< 该算式可以理解为将一个1x1的正方形四分，取每份的面积作为权重
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) { //< 前向中return 0的情况
    // empty
    w1 = w2 = w3 = w4 = 0.; //< 权值w1~w4全零代表不反传
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  
  ///< 反向传播只需要计算四个权值w1~w4，在对应的4个点按权值分配梯度即可。
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}
#endif  // COMMON_CUDA_HELPER
