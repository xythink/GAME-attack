
import matplotlib.pyplot as plt
import numpy as np
import os, time
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def clear_dir(path):
    if os.path.exists(path):
        del_list = os.listdir(path)
        for f in del_list:
            file_path = os.path.join(path, f)
            os.remove(file_path)
            print("Delete success:", file_path)
        os.removedirs(path)
        time.sleep(5)
