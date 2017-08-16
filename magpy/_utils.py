## utilities for maGPy

## nicer exit from ctrl-c
import signal, sys
def signalHandler(signal, frame):
        print(' Keyboard interupt, exiting.')
        sys.exit(0)
signal.signal(signal.SIGINT, signalHandler)

## use '@timeit' to decorate a function for timing
import time
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        for r in range(10): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec' % (f.__name__, te-ts) )
        return result
    return timed
