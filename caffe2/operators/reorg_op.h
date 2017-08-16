#ifndef REORG_OP_H_
#define REORG_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ReorgOp final : public Operator<Context> {
 public:
  ReorgOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        stride_(OperatorBase::GetSingleArgument<int>("stride", 1)) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
    CAFFE_ENFORCE_GT(stride_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template<typename T>
  bool DoRunWithType() {
    CAFFE_NOT_IMPLEMENTED;
  }

  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
  int stride_;
};

template <class Context>
class ReorgGradientOp final : public Operator<Context> {
 public:
  ReorgGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        stride_(OperatorBase::GetSingleArgument<int>("stride", 1)) {
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
    CAFFE_ENFORCE_GT(stride_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template<typename T>
  bool DoRunWithType() {
    CAFFE_NOT_IMPLEMENTED;
  }

  bool RunOnDevice() override;

 protected:
  StorageOrder order_;
  int stride_;
};

} // namespace caffe2

#endif // REORG_OP_H_
