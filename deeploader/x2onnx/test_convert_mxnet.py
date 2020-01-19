from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from deeploader.x2onnx.converters.mx2onnx import convert_model


def get_onnx_path(param_path):
    p = param_path.replace('/mxnet/', '/onnx/')
    p = p.replace('.params', '.onnx')
    return p


if __name__ == '__main__':
    model_dir = '../models/'
    model_list = []
    # facerec
    model = model_dir + 'mobilefacenet/mxnet/model-0000.params'
    input_shape = [(1, 3, 112, 112)]
    output_name = "fc1_output"
    model_list.append({'model': model, 'shape': input_shape, 'output': output_name})
    # MTCNN
    # det1
    model = model_dir + 'MTCNN/mxnet/det1-0001.params'
    input_shape = [(1, 3, 12, 12)]
    output_name = ""
    var_shape = [('?', 3, '?', '?')]
    model_list.append({'model': model, 'shape': input_shape, 'output': output_name,
                       'var_shape': var_shape})

    for m in model_list:
        var_shape = None
        if 'var_shape' in m:
            var_shape = m['var_shape']
        convert_model(m['model'], get_onnx_path(m['model']), m['shape'], m['output'], var_shape)
