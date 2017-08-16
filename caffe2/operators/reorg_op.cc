#include "reorg_op.h"

namespace caffe2 {

template <>
bool ReorgOp<CPUContext>::RunOnDevice() {
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

  const auto* Xdata = X.data<float>();
  auto* Ydata = Y->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          int in_idx = ((n * channels + c) * height + i) * width + j;
          int out_i = i / stride_;
          int out_j = j / stride_;
          int offset = (i % stride_) * stride_ + j % stride_;
          int out_c = offset * channels + c;
          int out_idx = ((n * out_channels + out_c) * out_height + out_i) * out_width + out_j;
          Ydata[out_idx] = Xdata[in_idx];
        }
      }
    }
  }

  return true;
}

template <>
bool ReorgGradientOp<CPUContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  auto* dX = Output(0);

  const int batch_size = X.dim32(0),
            channels = X.dim32(1),
            height = X.dim32(2),
            width = X.dim32(3),
            out_channels = channels * stride_ * stride_,
            out_height = height / stride_,
            out_width = width / stride_;
  dX->Resize(batch_size, channels, height, width);

  const auto* dYdata = dY.data<float>();
  auto* dXdata = dX->mutable_data<float>();

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          int in_idx = ((n * channels + c) * height + i) * width + j;
          int out_i = i / stride_;
          int out_j = j / stride_;
          int offset = (i % stride_) * stride_ + j % stride_;
          int out_c = offset * channels + c;
          int out_idx = ((n * out_channels + out_c) * out_height + out_i) * out_width + out_j;
          dXdata[in_idx] = dYdata[out_idx];
        }
      }
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(Reorg, ReorgOp<CPUContext>);
REGISTER_CPU_OPERATOR(ReorgGradient, ReorgGradientOp<CPUContext>);

// Input: X
// Output: Y
OPERATOR_SCHEMA(Reorg)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Implement reorg operation in darknet.
)DOC")
    .Arg("order", "A StorageOrder string (Default: \"NCHW\").")
    .Arg("stride", "The stride size (Default: 1).")
    .Input(
        0,
        "X",
        "The input 4-D tensor of data. Only NCHW order is currently supported.")
    .Output(
        0,
        "Y",
        "Reorganized output 4-D tensor of shape "
        "(batch_size, channels * stride * stride, height / stride, width / stride).");

// Input: dY (aka "gradOutput"), X
// Output: dX (aka "gradInput")
OPERATOR_SCHEMA(ReorgGradient).NumInputs(1).NumOutputs(1);

class GetReorgGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ReorgGradient",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(Reorg, GetReorgGradient);

} // namespace caffe2
