import numpy as np
import torch


arr = torch.from_numpy(np.array([[1, 2], [3, 4], [5, 6]]))

print(arr[0:1])