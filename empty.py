import time
import torch

device = 3
# 2G ==> 9219M
zeros = torch.zeros((1000, 1000, 2000)).to(device)
while True:
    time.sleep(1)