# import pytorch_lightning  # <--- Comment or uncomment to test

import os
import multiprocessing

pool = multiprocessing.Pool(8)
print(pool.map(os.sched_getaffinity, [0] * 8))