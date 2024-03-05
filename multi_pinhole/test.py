# speed test (np.nonzero vs np.where vs np.argwhere)
import time

import numpy as np


def random_a(shape_a, seed=1234):
    random_generator = np.random.default_rng(seed=seed)
    i = random_generator.integers(0, shape_a[0] - 1, size=shape_a[0] // 2)
    a = np.zeros(shape_a)
    a[i] = 1
    return a


if __name__ == '__main__':
    lst = [np.where(np.random.random(i) > 0.7)[0] for i in np.random.randint(0, 2000, size=1000)]

    start = time.time()
    arr1 = np.unique(np.concatenate(lst))
    print(time.time() - start)

    start = time.time()
    arr2 = np.unique(np.hstack(lst))
    print(time.time() - start)

    start = time.time()
    arr3 = np.unique(np.concatenate(lst).ravel())
    print(time.time() - start)

    start = time.time()
    arr4 = np.unique(np.hstack(lst).ravel())
    print(time.time() - start)

    start = time.time()
    arr5 = np.array(set(np.concatenate(lst)))
    print(time.time() - start)

    start = time.time()
    arr6 = np.array(set(np.hstack(lst)))
    print(time.time() - start)
