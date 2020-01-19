import onnxruntime as rt

from .base import BaseBackend


def parse_onnx_version(ver_int):
    major = ver_int >> 48
    minor = ver_int >> 32 & (2 ** 16)
    patch = ver_int & (2 ** 32)
    ver_str = '%d.%d.%d' % (major, minor, patch)
    return ver_str


class OnnxBackend(BaseBackend):
    @staticmethod
    def _print_meta(meta):
        print('Model meta:')
        print('  description:{}'.format(meta.description))
        print('  domain:{}'.format(meta.domain))
        print('  graph_name:{}'.format(meta.graph_name))
        print('  producer_name:{}'.format(meta.producer_name))
        print('  version:{}'.format(parse_onnx_version(meta.version)))

    def init(self, path_or_bytes, sess_options=None, providers=[]):
        session = rt.InferenceSession(path_or_bytes, sess_options, providers)
        self.session = session
        self.verbose()

    def verbose(self):
        OnnxBackend._print_meta(self.session.get_modelmeta())
        super(OnnxBackend, self).verbose()

    def get_inputs(self):
        l = []
        for idx, m in enumerate(self.session.get_inputs()):
            l.append(m.name)
        return l

    def get_outputs(self):
        l = []
        for idx, m in enumerate(self.session.get_outputs()):
            l.append(m.name)
        return l

    def run(self, output_names, input_feed, run_options=None):
        outputs = self.session.run(output_names, input_feed, run_options)
        return outputs

    def close(self):
        self.session._reset_session()
        self.session = None
