import time
import torch

device = 0
# 2G ==> 9219M
zeros = torch.zeros((1000, 1000, 3000)).to(device)
while True:
    time.sleep(1)