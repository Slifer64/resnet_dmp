import time

class Timer:

    def __init__(self):

        self.t_start = time.time()
        self.t_end = time.time()

    def tic(self):

        self.t_start = time.time()

    def toc(self, format='sec'):

        self.t_end = time.time()
        elaps_t = self.t_end - self.t_start

        if format == 'sec':
            return elaps_t
        elif format == 'milli':
            return elaps_t*1e3
        elif format == 'micro':
            return elaps_t * 1e6
        elif format == 'nano':
            return elaps_t * 1e9
        else:
            raise RuntimeError('Unsupported format {0:s}'.format(format))