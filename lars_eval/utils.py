from contextlib import contextmanager
from tqdm.auto import tqdm
from multiprocessing import Queue, Pool, RLock

import lars_eval.context as ctx

def tqdm_pool_initializer(q,lock,initializer,args):
    # Set process id, tqdm lock
    ctx.set_pid(q.get())
    tqdm.set_lock(lock)

    if initializer is not None:
        initializer(*args)

@contextmanager
def TqdmPool(processes, initializer=None, initargs=None, *args, **kwargs):
    """Wrapper of multiprocessing.Pool, suitable for use with tqdm. Workers are numbered in a global variable `PROC_I`."""

    q = Queue()
    for i in range(processes):
        q.put(i+1)

    with Pool(processes, initializer=tqdm_pool_initializer, initargs=(q,RLock(),initializer,initargs), *args, **kwargs) as p:
        yield p
