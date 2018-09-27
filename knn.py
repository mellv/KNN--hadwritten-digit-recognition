import math
import pandas as pd
import struct
import numpy as np
from math import sqrt
from collections import Counter

def data_exp(filename):
    with open(filename, 'rb') as f:
        null, d_type, dims = struct.unpack('>HBB', f.read(4))
        data = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        if filename == "train-images.idx3-ubyte":
            x = np.fromstring(f.read(), dtype=np.uint8).reshape(data[0], data[1] * data[2])
            return x
        if filename == "t10k-images.idx3-ubyte":
            x = np.fromstring(f.read(), dtype=np.uint8).reshape(data[0], data[1] * data[2])
            return x
        y = np.fromstring(f.read(), dtype=np.uint8).reshape(data)
        return y
