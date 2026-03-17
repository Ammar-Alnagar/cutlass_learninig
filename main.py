import numpy as np
import torch

# -----------------------------------------
data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())


x = torch.arange(6).reshape(2, 3)
print(x[:, 1])
