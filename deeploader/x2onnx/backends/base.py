import abc


class BaseBackend(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def init(self, *args, **kwargs):
        pass

    def verbose(self):
        print('Inputs  :{}'.format(self.get_inputs()))
        print('Outputs :{}'.format(self.get_outputs()))

    def get_inputs(self):
        return ['data']

    def get_outputs(self):
        return []

    @abc.abstractmethod
    def run(self, output_names, input_feed, run_options=None):
        '''
        run model to get outputs
        :param output_names:
        :param input_feed:
        :param run_options:
        :return: output list
        '''

    @abc.abstractmethod
    def close(self):
        pass
