#include "caffe2/core/common_gpu.h"
#ifdef CAFFE_HAS_CUDA_FP16

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/leaky_relu_op.h"

#include "caffe2/utils/conversions.h"

namespace caffe2 {
namespace {
__global__ void LeakyReluKernelHalf(const int N, const half alpha, const half* X, half* Y) {
  const half kZero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
    Y[i] = __hgt(X[i], kZero) ? X[i] : __hmul(X[i], alpha);
  }
}

__global__ void LeakyReluGradientKernelHalf(
    const int N, const half alpha, const half* Y, const half* dY, half* dX) {
  const half kZero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, N) {
    dX[i] = __hgt(Y[i], kZero) ? dY[i] : __hmul(dY[i], alpha);
  }
}
} // namespace

template <>
LeakyReluOp<float16, CUDAContext>::LeakyReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {
    float alpha = 0;
    if (HasArgument("alpha")) {
      alpha = OperatorBase::GetSingleArgument<float>("alpha", 0);
    }
    alpha_ = convert::cpu_float2half_rn(alpha);
}

template <>
bool LeakyReluOp<float16, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  LeakyReluKernelHalf<<<
      CAFFE_GET_BLOCKS(X.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      X.size(),
      convert::float16ToHalf(alpha_),
      reinterpret_cast<const half*>(X.data<float16>()),
      reinterpret_cast<half*>(Y->mutable_data<float16>()));
  return true;
}

template <>
LeakyReluGradientOp<float16, CUDAContext>::LeakyReluGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {
    float alpha = 0;
    if (HasArgument("alpha")) {
      alpha = OperatorBase::GetSingleArgument<float>("alpha", 0);
    }
    alpha_ = convert::cpu_float2half_rn(alpha);
}

template <>
bool LeakyReluGradientOp<float16, CUDAContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(Y);
  CAFFE_ENFORCE_EQ(Y.size(), dY.size());
  LeakyReluGradientKernelHalf<<<
      CAFFE_GET_BLOCKS(Y.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      Y.size(),
      convert::float16ToHalf(alpha_),
      reinterpret_cast<const half*>(Y.data<float16>()),
      reinterpret_cast<const half*>(dY.data<float16>()),
      reinterpret_cast<half*>(dX->mutable_data<float16>()));
  return true;
}

OPERATOR_SCHEMA(LeakyReluFp16);
OPERATOR_SCHEMA(LeakyReluFp16Gradient);

REGISTER_CUDA_OPERATOR(LeakyReluFp16, LeakyReluOp<float16, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    LeakyReluFp16Gradient,
    LeakyReluGradientOp<float16, CUDAContext>);
} // namespace caffe2

#endif  // CAFFE_HAS_CUDA_FP16
