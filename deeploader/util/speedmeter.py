import time

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
        

