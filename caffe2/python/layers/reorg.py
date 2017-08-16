## @package reorg
# Module caffe2.python.layers.reorg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)


class Reorg(ModelLayer):

    def __init__(self, model, input_record, order='NCHW', stride=1,
                 name='reorg', **kwargs):
        super(Reorg, self).__init__(model, name, input_record, **kwargs)
        self.order = order
        self.stride = stride
        assert isinstance(input_record, schema.Scalar),\
            "Incorrect input type. Excpected Scalar, but received: {0}".\
            format(input_record)
        assert order == 'NCHW', "order should be 'NCHW'"

        batch_size, channels, height, width = input_record.field_type().shape
        data_type = input_record.field_type().base
        output_shape = (
            batch_size,
            channels * stride * stride,
            height // stride,
            width // stride
        )
        self.output_schema = schema.Scalar(
            (data_type, output_shape),
            self.get_next_blob_reference('output'))

    def add_ops(self, net):
        net.Reorg(
            self.input_record.field_blobs(),
            self.output_schema.field_blobs(),
            order=self.order,
            stride=self.stride,
        )
