from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mxnet as mx
import onnxruntime as rt

from converters import  onnx_mx
from backends.MxnetBackend import MxnetBackend, get_output_sym
from backends.OnnxBackend import OnnxBackend
from converters.x2onnx_util import *


def test_convert(pretrained, onnx_file_path, epoch=0, input_shape=[(1, 3, 112, 112)],
                 output_name=None):
    # pip install onnx==1.3.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
    # onnx_file_path='mobileface-zq.onnx'
    sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained, epoch)
    sym = get_output_sym(sym, output_name)
    # print(arg_params)
    arg_params.update(aux_params)
    onnx_mx.export_model(sym, arg_params, input_shape, onnx_file_path=onnx_file_path, verbose=True)
    print('DONE!')
    return onnx_file_path


def parse_mx_checkpoint(param_path):
    fname = os.path.basename(param_path)
    fname, _ = fname.split('.')
    name, epoch = fname.split('-')
    return os.path.dirname(param_path) + '/' + name, int(epoch)


def check_model(pretrained, onnx_file_path, input_shape, output_name, var_shape):
    param_path, epoch = parse_mx_checkpoint(pretrained)
    input_shape0 = input_shape[0]
    # check shape
    # check_onnx_model(onnx_file_path)
    np.random.seed(10)
    extractor = MxnetBackend()
    extractor.init(param_path + ',' + str(epoch), 1, input_shape0,
                   output_name)

    session = OnnxBackend()
    session.init(onnx_file_path)
    return check_model_outputs(extractor, session, input_shape, var_shape)


def convert_model(pretrained, onnx_file_path, input_shape, output_name, var_shape=None):
    param_path, epoch = parse_mx_checkpoint(pretrained)
    makedirs(onnx_file_path)
    print(param_path, epoch)
    print(onnx_file_path)
    test_convert(param_path, epoch=epoch, input_shape=input_shape, onnx_file_path=onnx_file_path,
                 output_name=output_name)
    # var shape conversion
    if var_shape:
        import onnx
        model = onnx.load(onnx_file_path)
        for idx, shape in enumerate(var_shape):
            for dim, val in enumerate(shape):
                if isinstance(val, str):
                    model.graph.input[idx].type.tensor_type.shape.dim[dim].dim_param = val
                else:
                    model.graph.input[idx].type.tensor_type.shape.dim[dim].dim_value = val
        onnx.save(model, onnx_file_path)

    return check_model(pretrained, onnx_file_path, input_shape, output_name, var_shape)


def check_onnx_model(model_path):
    import onnx
    from onnx import shape_inference
    original_model = onnx.load(model_path)

    # Check the model and print Y's shape information
    onnx.checker.check_model(original_model)
    print('Before shape inference, the shape info of Y is:\n{}'.format(
        original_model.graph.value_info))

    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(original_model)

    # Check the model and print Y's shape information
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(
        inferred_model.graph.value_info))
