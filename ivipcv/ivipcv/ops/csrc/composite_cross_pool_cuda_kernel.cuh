#ifndef COMPOSITE_CROSS_POOL_CUDA_KERNEL_CUH
#define COMPOSITE_CROSS_POOL_CUDA_KERNEL_CUH

#include "pytorch_cuda_helper.hpp"
#include "stdio.h"
#include "common_cuda_helper.hpp"
#include <string>

//< 双线性插值开关, 0代表不使用双线性插值，1代表启用双线性插值，改变后需要重新编译安装才会生效(重新编译前先删除ivipcv/build/)
#define COMP_ENABLE_BILINEAR_INTERPOLATE 0

#define MATH_PI 3.141592653589793

using namespace std;

template <typename T>
__global__ void composite_cross_pool_forward_cuda_kernel(const int nthreads, const int pool_mode, const int axis, 
  const T* input, const T* rois, T* output, T* argmax, T* argmax_y, const T spatial_scale, const int batch_size,
  const int channels, const int height, const int width){
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    //< 根据axis信息确定输出通道数倍率
    int axis_ratio = max(axis, 1);

    //< algorithm count
    int alg_count = 1;

    //< 根据输出通道需求计算输出通道总数
    int output_ratio = axis_ratio * alg_count; //< 输入的一个通道对应这么多输出通道
    int out_channels = channels * output_ratio; //< 输出特征谱的总通道数

    //< (out_n, out_c, out_h, out_w) 是output中的一个元素，即当前kernel正在计算的元素索引
    int out_w = index % width;
    int out_h = (index / width) % height;
    int out_c = (index / width / height) % out_channels;
    int out_n = index / width / height / out_channels;

    //< 由output元素索引计算对应的input的n, c以及line索引
    int in_n = out_n;
    int in_c = out_c / output_ratio;

    //< line索引 标识x方向线 or y方向线
    int line_idx = -1;
    if(axis == 2){
      line_idx = (out_c / alg_count) % 2;
    }
    else{
      line_idx = axis;
    }
    

    //< 算法索引 0代表max 1代表avg
    int alg_idx = pool_mode;
    // int alg_rank = out_c % alg_count; // 在所有启用的算例中，本算例的序号

    //< 计算指向当前roi的指针
    ///< CAUTION : roi输入格式更改时，此处指针计算也需更改
    const T *offset_rois = rois + (out_n * height * width * 6) + (out_h * width * 6) + (out_w * 6);
    
    //< 计算指向input(in_n, in_c)的指针
    const T* offset_input = input + (in_n * channels + in_c) * height * width;

    //< 取出ROI坐标
    T roi_x1 = offset_rois[0] * spatial_scale;
    T roi_x2 = offset_rois[1] * spatial_scale;
    T roi_y = offset_rois[2] * spatial_scale;
    T roi_x = offset_rois[3] * spatial_scale;
    T roi_y1 = offset_rois[4] * spatial_scale;
    T roi_y2 = offset_rois[5] * spatial_scale;

    //< 横线 x方向
    if(line_idx == 0){
      
      #if COMP_ENABLE_BILINEAR_INTERPOLATE == 1     //< 使用双线性插值
        T bin_x1 = roi_x1;
        T bin_x2 = roi_x2;
        T bin_y = (T)roi_y;
        // add roi offsets and clip to input boundaries
        bin_x1 = min(max(bin_x1, 0.0f), (float)width);
        bin_x2 = min(max(bin_x2, 0.0f), (float)width);
        bin_y = min(max(bin_y, 0.0f), (float)height - 1.0f); //< 这里由于bin_y不作为循环的上界，因此要height-1
      #else                                         //< 不使用双线性插值
        int bin_x1 = floor(roi_x1);
        int bin_x2 = ceil(roi_x2);
        int bin_y = (int)roi_y;
        // add roi offsets and clip to input boundaries
        bin_x1 = min(max(bin_x1, 0), width);
        bin_x2 = min(max(bin_x2, 0), width);
        bin_y = min(max(bin_y, 0), height - 1);
      #endif

      bool is_empty = (bin_x2 <= bin_x1);

      //< 对不同算例
      ///< X线 最大 -------------------------------------------------------------------------------
      if(alg_idx == 0){
        //< 为空
        if(is_empty){
          output[index] = 0;
          argmax[index] = -1;
          argmax_y[index] = -1;
        }
        else{
          //< 计算
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
            T max_val = - FLT_MAX;
            int max_idx = 0;
            for(int w = bin_x1; w < bin_x2; ++ w){
              int offset = bin_y * width + w;
              if (offset_input[offset] > max_val){
                max_val = offset_input[offset];
                max_idx = offset;
              }
            }
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T max_val = - FLT_MAX;
            T max_idx = 0.0f;
            for(T w = bin_x1; w < bin_x2; w += 1.0f){
              T val = bilinear_interpolate(offset_input, height, width, bin_y, w, index);
              if(val > max_val){
                max_val = val;
                max_idx = w;
              }
            }
          #endif
          output[index] = max_val;
          argmax[index] = (T)max_idx;
          argmax_y[index] = bin_y;
        }
      }
      ///< X线 平均 -------------------------------------------------------------------------------
      else if(alg_idx == 1){ //< 线平均
        //< 为空
        if(is_empty){
          output[index] = 0;
        }
        else{
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
            T ave_val = 0;
            int pool_size = (bin_x2 - bin_x1);
            for(int w = bin_x1; w < bin_x2; ++ w){
              int offset = bin_y * width + w;
              ave_val += offset_input[offset];
            }
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T ave_val = 0.0f;
            T pool_size = (bin_x2 - bin_x1);
            for(T w = bin_x1; w < bin_x2; w += 1.0f){
              T val = bilinear_interpolate(offset_input, height, width, bin_y, w, index);
              ave_val += val;
            }
          #endif
          output[index] = ave_val / (T)pool_size;
        }
      }
    }
    //< 竖线 y方向
    else if (line_idx == 1){
      #if COMP_ENABLE_BILINEAR_INTERPOLATE == 1       //< 使用双线性插值
        T bin_x = roi_x;
        T bin_y1 = roi_y1;
        T bin_y2 = roi_y2;

        bin_x = min(max(bin_x, 0.0f), (float)width - 1.0f);
        bin_y1 = min(max(bin_y1, 0.0f), (float)height);
        bin_y2 = min(max(bin_y2, 0.0f), (float)height);
      #else                                           //< 不使用双线性插值
        int bin_x = roi_x;
        int bin_y1 = floor(roi_y1);
        int bin_y2 = ceil(roi_y2);

        bin_x = min(max(bin_x, 0), width - 1);
        bin_y1 = min(max(bin_y1, 0), height);
        bin_y2 = min(max(bin_y2, 0), height);
      #endif

      bool is_empty = (bin_y2 <= bin_y1);

      //< 对不同算例
      ///< Y线 最大 -------------------------------------------------------------------------------
      if(alg_idx == 0){ //< 线最大
        //< 为空
        if(is_empty){
          output[index] = 0;
          argmax[index] = -1;
          argmax_y[index] = -1;
        }
        else{
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
            T max_val = -FLT_MAX;
            int max_idx = 0;
            for(int h = bin_y1; h < bin_y2; ++ h){
              int offset = h * width + bin_x;
              if (offset_input[offset] > max_val) {
                max_val = offset_input[offset];
                max_idx = offset;
              }
            }
            output[index] = max_val;
            argmax[index] = max_idx;
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T max_val = - FLT_MAX;
            T max_idx_y = 0.0f;
            for(T h = bin_y1; h < bin_y2; h += 1.0f){
              T val = bilinear_interpolate(offset_input, height, width, h, bin_x, index);
              if(val > max_val){
                max_val = val;
                max_idx_y = h;
              }
            }
            output[index] = max_val;
            argmax[index] = bin_x;
            argmax_y[index] = max_idx_y;
          #endif
        }
      }
      ///< Y线 平均 -------------------------------------------------------------------------------
      else if(alg_idx == 1){ //< 线平均
        //< 为空
        if(is_empty){
          output[index] = 0;
        }
        else{
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
            T ave_val = 0;
            int pool_size = (bin_y2 - bin_y1);
            for(int h = bin_y1; h < bin_y2; ++ h){
              int offset = h * width + bin_x;
              ave_val += offset_input[offset];
            }
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T ave_val = 0.0f;
            T pool_size = (bin_y2 - bin_y1);
            for(T h = bin_y1; h < bin_y2; h += 1.0f){
              T val = bilinear_interpolate(offset_input, height, width, h, bin_x, index);
              ave_val += val;
            }
          #endif
          output[index] = ave_val / (T)pool_size;
        }
      }
    }
  }
}

template <typename T>
__global__ void composite_cross_pool_backward_cuda_kernel(const int nthreads, const int pool_mode, const int axis, const T* grad_output,
  const T* rois, T* grad_input, T* argmax, T* argmax_y, const T spatial_scale, const int batch_size, const int channels, 
  const int height, const int width){
  CUDA_1D_KERNEL_LOOP(index, nthreads) {

    //< 根据axis信息确定输出通道数倍率
    int axis_ratio = max(axis, 1);

    //< 根据pool_mode确定输出通道数倍率
    int alg_count = 1;

    //< 根据输出通道需求计算输出通道总数
    int output_ratio = axis_ratio * alg_count;  //< 输入的一个通道对应这么多输出通道
    int out_channels = channels * output_ratio; //< 输出特征谱的总通道数，这里channels为grad_input的总通道数

    //< (out_n, out_c, out_h, out_w) 是output中的一个元素，即当前kernel正在计算的元素索引
    int out_w = index % width;
    int out_h = (index / width) % height;
    int out_c = (index / width / height) % out_channels;
    int out_n = index / width / height / out_channels;

    //< 由output元素索引计算对应的input的n, c以及line索引
    int in_n = out_n;
    int in_c = out_c / output_ratio;

    //< line索引 标识x方向线 or y方向线
    int line_idx = -1;
    if(axis == 2){
      line_idx = (out_c / alg_count) % 2;
    }
    else{
      line_idx = axis;
    }

    //< 算法索引 
    int alg_idx = pool_mode;

    //< 计算指向当前roi的指针
    const T *offset_rois = rois + (out_n * height * width * 6) + (out_h * width * 6) + (out_w * 6);
    
    //< 计算指向grad_input(in_n, in_c)的指针
    T* grad_input_offset = grad_input + (in_n * channels + in_c) * height * width;

    //< 取出ROI坐标
    T roi_x1 = offset_rois[0] * spatial_scale;
    T roi_x2 = offset_rois[1] * spatial_scale;
    T roi_y = offset_rois[2] * spatial_scale;
    T roi_x = offset_rois[3] * spatial_scale;
    T roi_y1 = offset_rois[4] * spatial_scale;
    T roi_y2 = offset_rois[5] * spatial_scale;

    //< 横线 x方向
    if(line_idx == 0){
      #if COMP_ENABLE_BILINEAR_INTERPOLATE == 1     //< 使用双线性插值
        T bin_x1 = roi_x1;
        T bin_x2 = roi_x2;
        T bin_y = (T)roi_y;
        // add roi offsets and clip to input boundaries
        bin_x1 = min(max(bin_x1, 0.0f), (float)width);
        bin_x2 = min(max(bin_x2, 0.0f), (float)width);
        bin_y = min(max(bin_y, 0.0f), (float)height - 1.0f);
      #else                                         //< 不使用双线性插值
        int bin_x1 = floor(roi_x1);
        int bin_x2 = ceil(roi_x2);
        int bin_y = roi_y;
        // add roi offsets and clip to input boundaries
        bin_x1 = min(max(bin_x1, 0), width);
        bin_x2 = min(max(bin_x2, 0), width);
        bin_y = min(max(bin_y, 0), height - 1);
      #endif

      bool is_empty = (bin_x2 <= bin_x1);

      //< 对不同池化模式
      ///< X线 最大 -------------------------------------------------------------------------------
      if(alg_idx == 0){
        #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
          int argmax_index = (int)argmax[index];
          if (argmax_index != -1) {
            atomicAdd(grad_input_offset + argmax_index, grad_output[index]); // atomicAdd(X,Y) means *X = *X + Y;
          }
        #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
          T argmax_index_x = argmax[index];
          T argmax_index_y = argmax_y[index];
          if(argmax_index_x != -1 && argmax_index_y != -1){
            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;
            bilinear_interpolate_gradient(height, width, argmax_index_y, argmax_index_x, w1, w2, w3, w4,
              x_low, x_high, y_low, y_high, index);
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
              atomicAdd(grad_input_offset + y_low * width + x_low,
                        grad_output[index] * w1);
              atomicAdd(grad_input_offset + y_low * width + x_high,
                        grad_output[index] * w2);
              atomicAdd(grad_input_offset + y_high * width + x_low,
                        grad_output[index] * w3);
              atomicAdd(grad_input_offset + y_high * width + x_high,
                        grad_output[index] * w4);
            }
          }
        #endif
      }
      ///< X线 平均 -------------------------------------------------------------------------------
      else if(alg_idx == 1){ //< 线平均
        //< 为空
        if(is_empty){

        }
        else{
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0       //< 不使用双线性插值
            int pool_size = (bin_x2 - bin_x1);
            for(int w = bin_x1; w < bin_x2; ++ w){
              int offset = bin_y * width + w;
              atomicAdd(grad_input_offset + offset, (T)(grad_output[index] / (T)pool_size));
            }
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T pool_size = (bin_x2 - bin_x1);
            for(T w = bin_x1; w < bin_x2; w += 1.0f){
              T w1, w2, w3, w4;
              int x_low, x_high, y_low, y_high;
              bilinear_interpolate_gradient(height, width, bin_y, w, w1, w2, w3, w4,
                x_low, x_high, y_low, y_high, index);
              if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                atomicAdd(grad_input_offset + y_low * width + x_low,
                          grad_output[index] * w1 / pool_size);
                atomicAdd(grad_input_offset + y_low * width + x_high,
                          grad_output[index] * w2 / pool_size);
                atomicAdd(grad_input_offset + y_high * width + x_low,
                          grad_output[index] * w3 / pool_size);
                atomicAdd(grad_input_offset + y_high * width + x_high,
                          grad_output[index] * w4 / pool_size);
              }
            }
          #endif
        }
      }
    }
    //< 竖线 y方向
    else if (line_idx == 1){
      #if COMP_ENABLE_BILINEAR_INTERPOLATE == 1       //< 使用双线性插值
        T bin_x = roi_x;
        T bin_y1 =roi_y1;
        T bin_y2 = roi_y2;

        bin_x = min(max(bin_x, 0.0f), (float)width - 1.0f);
        bin_y1 = min(max(bin_y1, 0.0f), (float)height);
        bin_y2 = min(max(bin_y2, 0.0f), (float)height);
      #else                                           //< 不使用双线性插值
        int bin_x = roi_x;
        int bin_y1 = floor(roi_y1);
        int bin_y2 = ceil(roi_y2);

        bin_x = min(max(bin_x, 0), width - 1);
        bin_y1 = min(max(bin_y1, 0), height);
        bin_y2 = min(max(bin_y2, 0), height);
      #endif
      
      bool is_empty = (bin_y2 <= bin_y1);

      //< 对不同算例
      ///< Y线 最大 -------------------------------------------------------------------------------
      if(alg_idx == 0){ //< 线最大
        #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
          int argmax_index = (int)argmax[index];
          if (argmax_index != -1) {
            atomicAdd(grad_input_offset + argmax_index, grad_output[index]); // atomicAdd(X,Y) means *X = *X + Y;
          }
        #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
          T argmax_index_x = argmax[index];
          T argmax_index_y = argmax_y[index];
          if(argmax_index_x != -1 && argmax_index_y != -1){
            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;
            bilinear_interpolate_gradient(height, width, argmax_index_y, argmax_index_x, w1, w2, w3, w4,
              x_low, x_high, y_low, y_high, index);
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
              atomicAdd(grad_input_offset + y_low * width + x_low,
                        grad_output[index] * w1);
              atomicAdd(grad_input_offset + y_low * width + x_high,
                        grad_output[index] * w2);
              atomicAdd(grad_input_offset + y_high * width + x_low,
                        grad_output[index] * w3);
              atomicAdd(grad_input_offset + y_high * width + x_high,
                        grad_output[index] * w4);
            }
          }
        #endif
      }
      ///< Y线 平均 -------------------------------------------------------------------------------
      else if(alg_idx == 1){ //< 线平均
        //< 为空
        if(is_empty){

        }
        else{
          #if COMP_ENABLE_BILINEAR_INTERPOLATE == 0 //< 不使用双线性插值
            int pool_size = (bin_y2 - bin_y1);
            for(int h = bin_y1; h < bin_y2; ++ h){
              int offset = h * width + bin_x;
              atomicAdd(grad_input_offset + offset, (T)(grad_output[index] / (T)pool_size));
            }
          #elif COMP_ENABLE_BILINEAR_INTERPOLATE == 1 //< 使用双线性插值
            T pool_size = (bin_y2 - bin_y1);
            for(T h = bin_y1; h < bin_y2; h += 1.0f){
              T w1, w2, w3, w4;
              int x_low, x_high, y_low, y_high;
              bilinear_interpolate_gradient(height, width, h, bin_x, w1, w2, w3, w4,
                x_low, x_high, y_low, y_high, index);
              if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                atomicAdd(grad_input_offset + y_low * width + x_low,
                          grad_output[index] * w1 / pool_size);
                atomicAdd(grad_input_offset + y_low * width + x_high,
                          grad_output[index] * w2 / pool_size);
                atomicAdd(grad_input_offset + y_high * width + x_low,
                          grad_output[index] * w3 / pool_size);
                atomicAdd(grad_input_offset + y_high * width + x_high,
                          grad_output[index] * w4 / pool_size);
              }
            }
          #endif
        }
      }
    }
  }
}

#endif