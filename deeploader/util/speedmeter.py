import time
from collections import OrderedDict


class Speedometer(object):
    def __init__(self, num_ema=10):
        self.num_ema = num_ema
        self.reset()

    def reset(self):
        self.total = 0
        self.start = time.time()
        self.last = -1
        self.ema = -1

    def __call__(self, count):
        """Callback to Show speed."""
        now = time.time()
        if self.last < 0:
            speed = count/(now - self.start)
            self.ema = speed
            self.last = now
            self.total += count
            return self.ema
        # ema
        speed = count/(now - self.last)
        self.ema = (self.ema * (self.num_ema-1) + speed) / self.num_ema
        self.last = now
        self.total += count
        return self.ema
    
    @property
    def speed(self):
        return self.ema
        
    @property   
    def avg(self):
        now = time.time()
        return self.total/(now - self.start)


class TimeRecorder(object):
    def __init__(self):
        self.record = OrderedDict()
        self.last = None

    def reset(self):
        self.record.clear()
        self.last = None

    def start(self, tag):
        """Callback to Show speed."""
        now = time.time()
        self.record[tag] = [now]
        self.last = tag

    def stop(self, tag):
        now = time.time()
        self.record[tag].append(now)
        self.last = None

    def into(self, tag):
        if self.last:
            self.stop(self.last)
        self.start(tag)

    def get_time(self, tag):
        l = self.record[tag]
        return l[-1] - l[0]

    def report(self, tag_list=None):
        tags = list(self.record.keys())
        fmt_str = ''
        for tag in tags:
            if tag_list and tag not in tag_list:
                continue
            fmt_str += '%s:%.4f ' % (tag, self.get_time(tag))
        print(fmt_str)
