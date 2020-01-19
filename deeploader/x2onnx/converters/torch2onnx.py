from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from converters import onnx_torch
from backends.OnnxBackend import OnnxBackend
from backends.TorchBackend import TorchBackend
from converters.x2onnx_util import *


def convert_model(model, onnx_file_path, input_shape, input_names, output_names=None, var_shape=None):
    makedirs(onnx_file_path)
    input_shape0 = input_shape[0]
    input_data = np.random.randn(*input_shape0).astype(np.float32)
    input_data = torch.from_numpy(input_data).cuda()
    onnx_torch.export(model, input_data, onnx_file_path, verbose=False,
                      input_names=input_names, output_names=output_names)

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

    # check shape
    # check_onnx_model(onnx_file_path)
    np.random.seed(10)
    extractor = TorchBackend()
    extractor.init(model)

    session = OnnxBackend()
    session.init(onnx_file_path)
    return check_model_outputs(extractor, session, input_shape, var_shape)
