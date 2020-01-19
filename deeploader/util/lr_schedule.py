import logging
import math


class MultiStepScheduler(object):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

       step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

       base_lr * pow(factor, k+1)

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change
    factor: float
        The factor to change the learning rate.
    warmup_steps: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: string
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """

    def __init__(self, step, factor=0.1, base_lr=0.01, decay_mode='', warmup_steps=0, warmup_begin_lr=0,
                 warmup_final_lr=0, warmup_mode='linear', warmup_end_callback=None):
        assert isinstance(step, list) and len(step) >= 1
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        if warmup_final_lr > 0:
            self.warmup_final_lr = warmup_final_lr
        else:
            self.warmup_final_lr = base_lr
        self.warmup_mode = warmup_mode
        self.decay_mode = decay_mode
        self.warmup_end_callback = warmup_end_callback
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def get_warmup_lr(self, num_update):
        assert num_update < self.warmup_steps
        if self.warmup_mode == 'linear':
            increase = (self.warmup_final_lr - self.warmup_begin_lr) \
                       * float(num_update) / float(self.warmup_steps)
            return self.warmup_begin_lr + increase
        elif self.warmup_mode == 'constant':
            return self.warmup_begin_lr
        else:
            raise ValueError("Invalid warmup mode %s" % self.warmup_mode)

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        # notify warmup end
        if self.warmup_end_callback:
            self.warmup_end_callback()
            self.warmup_end_callback = None
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while self.cur_step_ind <= len(self.step) - 1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
            else:
                break
        lr = self.base_lr
        # decaying
        if num_update <= self.step[-1]:
            last_step = 0
            if self.cur_step_ind > 0:
                last_step = self.step[self.cur_step_ind - 1]
            final_lr = lr * self.factor
            interval = self.step[self.cur_step_ind] - last_step
            todo = (self.step[self.cur_step_ind] - num_update)
            pgr = 1.0 - todo * 1.0 / interval
            # print(pgr)
            if self.decay_mode == 'cosine':
                lr = final_lr + (lr - final_lr) * \
                     (1 + math.cos(math.pi * pgr)) / 2
            elif self.decay_mode == 'linear':
                lr = final_lr + (lr - final_lr) * (1-pgr)
        return lr


if __name__ == '__main__':
    decay_mode = 'cosine'
    #decay_mode = 'linear'
    # decay_mode = ''
    lr_schedule = MultiStepScheduler(step=[10000, 20000, 40000], factor=0.1, base_lr=1, decay_mode=decay_mode)
    x = []
    y = []
    for step in range(0,60000,300):
        lr = lr_schedule(step)
        print('step:%4d lr:%.5f' % (step, lr))
        x.append(step)
        y.append(lr)
    import matplotlib as mpl

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.xlabel('FLOPs', fontsize=10)
    plt.ylabel('layer', fontsize=10)

    plt.savefig(decay_mode + '.png')
    plt.show()