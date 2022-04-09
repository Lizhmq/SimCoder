import os, json
import faiss
import numpy as np
import torch
from torch.utils.data import Dataset
from path_utils import *
from mmap_dataset import MmapDataset
from knn.pq_wrapper import TorchPQCodec
from dataset import KNNDataset



subtokenids, sizes = np.zeros((1000, 3000)), np.ones(1000) * 3000
ds = KNNDataset(subtokenids, sizes, "/home/lizhuo/lzzz/SimCoder/data", "test", k=128)