import time
import numpy as np
from multiprocessing import Process, Manager


def worker(l1, l2):
    while True:
        print(f"received l1: {l1}")
        l2 = l1.reverse()
        print(f"reversed l2: {l2}")
        time.sleep(0.5)


if __name__ == '__main__':
    manager = Manager()
    d = manager.dict()
    l1 = manager.list(range(10))
    l2 = manager.list(range(10))

    p = Process(target=worker, args=(l1, l2))
    p.start()
    # p.join()

    while True:
        for i in range(10):
            l1[i] = np.random.choice(12)
        print(f"l1: {l1}")  # Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        s1 = time.time()
        print(f"l2: {l2}")  # Output: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        s2 = time.time()
        print(f"time: {s2 - s1}")
        time.sleep(0.1)
