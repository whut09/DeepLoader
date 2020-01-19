from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeploader.x2onnx.converters.torch2onnx import convert_model


if __name__ == '__main__':
    #
    model = load_your_torch_module()
    onnx_file_path = 'test.onnx'
    input_shape = [(1, 3, 112, 112)]
    input_names = ['img']
    convert_model(model, onnx_file_path, input_shape, input_names)
