import abc

import mxnet as mx
from mxnet import nd

from .base import BaseBackend


def get_output_sym(sym, output_name):
    outputs = sym.list_outputs()
    inputs = sym.list_inputs()
    all_layers = sym.get_internals()
    if output_name:
        sym = all_layers[output_name]
    else:
        out_syms = []
        for out in all_layers:
            name = out.name
            if name.find('label') > 0:
                continue
            _name = name + '_output'
            # print(_name)
            if _name in outputs:
                out_syms.append(out)
        # print(out_syms)
        # out_syms.append(all_layers['conv4_output'])
        sym = mx.sym.Group(out_syms)
        # sym = all_layers['prelu4_output']
    return sym


class MxnetBackend(BaseBackend):
    def init(self, model_path, batch_size, data_shape, output_name, 
             data_name='data', label_name=None):
        print('data_shape', data_shape)
        ctx = mx.gpu(0)
        vec = model_path.split(',')
        prefix = vec[0]
        epoch = 0
        if len(vec) > 1:
            epoch = int(vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        sym = get_output_sym(sym, output_name)
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=(label_name,))
        data_shapes = [(data_name, (batch_size, data_shape[1], data_shape[2], data_shape[3]))]
        label_shapes = None
        if label_name:
            label_shapes = [(label_name, (batch_size,))]
        model.bind(data_shapes=data_shapes,label_shapes=label_shapes, for_training=False, grad_req='null')
        model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        #print(model.data_shapes)
        #print(model.data_names)
        # save context
        self.label_name = label_name
        self.model = model
        self.batch_size = batch_size
        self.verbose()

    def get_inputs(self):
        return self.model.data_names

    def get_outputs(self):
        return self.model.output_names

    def run(self, output_names, input_feed, run_options=None):
        x_batch = input_feed['data']
        nd_x = nd.array(x_batch)
        # print(nd_x)
        if self.label_name:
            _label = nd.ones((nd_x.shape[0],))
            db = mx.io.DataBatch(data=(nd_x, ), label=(_label, ))
        else:
            db = mx.io.DataBatch(data=(nd_x,))
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        model_output_names = self.model.output_names
        outputs = []
        for idx, out in enumerate(net_out):
            if output_names and model_output_names[idx] not in output_names:
                continue
            f = out.asnumpy()
            outputs.append(f)
        return outputs

    def close(self):
        self.model._reset_bind()
        self.model = None
