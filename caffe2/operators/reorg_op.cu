#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "reorg_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ReorgForward(
    const int nthreads,
    const T* bottom_data,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int stride,
    const int out_channels,
    const int out_height,
    const int out_width,
    T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int in_idx = index;
    int j = index % width; index /= width;
    int i = index % height; index /= height;
    int c = index % channels; index /= channels;
    int n = index % batch_size;
    int out_i = i / stride;
    int out_j = j / stride;
    int out_c = ((i % stride) * stride + j % stride) * channels + c;
    int out_idx = ((n * out_channels + out_c) * out_height + out_i) * out_width + out_j;
    top_data[out_idx] = bottom_data[in_idx];
  }
}

template <typename T>
__global__ void ReorgBackward(
    const int nthreads,
    const T* top_diff,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int stride,
    const int out_channels,
    const int out_height,
    const int out_width,
    T* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int in_idx = index;
    int j = index % width; index /= width;
    int i = index % height; index /= height;
    int c = index % channels; index /= channels;
    int n = index % batch_size;
    int out_i = i / stride;
    int out_j = j / stride;
    int out_c = ((i % stride) * stride + j % stride) * channels + c;
    int out_idx = ((n * out_channels + out_c) * out_height + out_i) * out_width + out_j;
    bottom_diff[in_idx] = top_diff[out_idx];
  }
}

} // namespace

template <>
template <typename T>
bool ReorgOp<CUDAContext>::DoRunWithType() {
  const auto& X = Input(0); // Input data
  auto* Y = Output(0); // Reorganized data

  const int batch_size = X.dim32(0),
            channels = X.dim32(1),
            height = X.dim32(2),
            width = X.dim32(3),
            out_channels = channels * stride_ * stride_,
            out_height = height / stride_,
            out_width = width / stride_;
  Y->Resize(batch_size, out_channels, out_height, out_width);

  const auto size = Y->size();
  ReorgForward<T><<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      X.data<T>(),
      batch_size,
      channels,
      height,
      width,
      stride_,
      out_channels,
      out_height,
      out_width,
      Y->mutable_data<T>());

  return true;
}

template <>
bool ReorgOp<CUDAContext>::RunOnDevice() {
  const auto& X = Input(0); // Input data

  if (X.IsType<float>()) {
    return DoRunWithType<float>();
  } else if (X.IsType<float16>()) {
    return DoRunWithType<float16>();
  } else {
    CAFFE_THROW("Unsupported input type");
  }
}

template <>
template <typename T>
bool ReorgGradientOp<CUDAContext>::DoRunWithType() {
  auto& dY = Input(0); // Gradient of net w.r.t. output of "forward" op
  // (aka "gradOutput")
  auto& X = Input(1); // Input data
  auto* dX = Output(0); // Gradient of net w.r.t. input to "forward" op
  // (aka "gradInput")
  const int batch_size = X.dim32(0),
            channels = X.dim32(1),
            height = X.dim32(2),
            width = X.dim32(3),
            out_channels = channels * stride_ * stride_,
            out_height = height / stride_,
            out_width = width / stride_;
  dX->ResizeLike(X);

  ReorgBackward<T><<<
      CAFFE_GET_BLOCKS(dY.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      dY.size(),
      dY.data<T>(),
      batch_size,
      channels,
      height,
      width,
      stride_,
      out_channels,
      out_height,
      out_width,
      dX->mutable_data<T>());

  return true;
}

template <>
bool ReorgGradientOp<CUDAContext>::RunOnDevice() {
  const auto& X = Input(0); // Input data

  if (X.IsType<float>()) {
    return DoRunWithType<float>();
  } else if (X.IsType<float16>()) {
    return DoRunWithType<float16>();
  } else {
    CAFFE_THROW("Unsupported input type");
  }
}

REGISTER_CUDA_OPERATOR(Reorg, ReorgOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(ReorgGradient, ReorgGradientOp<CUDAContext>);

} // namespace caffe2
