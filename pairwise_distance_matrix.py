import numpy as np
import torch

x = torch.tensor(np.random.rand(100, 64), dtype=torch.float32)
y = torch.tensor(np.random.rand(20, 64), dtype=torch.float32)

dist = torch.cdist(x, y, p=2)

print(dist)