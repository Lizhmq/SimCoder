import os, json
import faiss
import numpy as np
import torch
from torch.utils.data import Dataset
from path_utils import *
from mmap_dataset import MmapDataset
from knn.pq_wrapper import TorchPQCodec




class KNNDataset(Dataset):

    def __init__(self, subtokens, sizes, data_path, split, k, tokenizer):
        """
        Build a LM dataset with KNN info.

        Args:
            subtokens: a list of subtoken ids
            sizes: list of subtokens length
            data_path: path to the data directory
            split: train/valid/test
            k: number of neighbors
        """
        self.dtype = np.float16
        self.invalid_context = 512
        self.k = k
        self.subtokens = subtokens
        self.sizes = np.array(sizes)
        self.cum_sizes = np.cumsum(self.sizes)
        self.cum_sizes = np.insert(self.cum_sizes, 0, 0)
        self.num_tokens = sum(self.sizes)
        self.neighbor_info = json.load(open(os.path.join(dstore_path(data_path, "train"), "info.json")))
        self.info = json.load(open(os.path.join(dstore_path(data_path, split), "info.json")))
        self.train_tokens = self.neighbor_info["dstore_size"]
        self.quantize_neighbor_feat = np.load(quantized_feature_path(data_path, "train"))
        self.neighbor_offsets = MmapDataset(neighbor_path(data_dir=data_path, mode=split, k=self.k),
                                    dtype=np.int64, shape=(self.num_tokens, self.k), warmup=False)
        self.neighbor_tokens = MmapDataset(value_path(data_dir=data_path, mode="train"),
                                    dtype=np.int32,
                                    shape=(self.train_tokens, 2), warmup=False)
        quantiz_path = quantizer_path(data_dir=data_path, suffix="", norm=False)  
        self.tgt_quantizer = TorchPQCodec(index=faiss.read_index(quantiz_path))
        self.tokenizer = tokenizer
        self.init_block()
        self.init_special_tokens()

    def init_block(self, block_size=64):
        """
        Set file index, file offset and sample length.
        """
        file_index, file_offset, sample_length = [], [], []
        for i in range(len(self.sizes)):
            start_idx = 0
            while start_idx < self.sizes[i]:
                file_index.append(i)
                file_offset.append(start_idx)
                sample_length.append(min(block_size, self.sizes[i] - start_idx))
                start_idx += sample_length[-1]
        self.file_index = np.array(file_index)
        self.file_offset = np.array(file_offset)
        self.sample_length = np.array(sample_length)

    def init_special_tokens(self):
        self.pad = 0
        self.unk = 1
        self.eos = 2

    def double_idx_to_one(self, file_idx, offset):
        """
        Convert a double index to a single index.
        """
        return self.cum_sizes[file_idx] + offset

    def __len__(self):
        """
        Batch numbers.
        """
        return len(self.file_index)

    def __getitem__(self, index):
        file_idx, offset = self.file_index[index], self.file_offset[index]
        sample_length = self.sample_length[index]
        
        buffer = torch.LongTensor(self.subtokens[file_idx])
        s, e = offset, offset + sample_length
        target = buffer[s:e]
        if s == 0:
            source = torch.cat([buffer.new([self.eos]), buffer[0:e-1]])
        else:
            source = buffer[s-1:e-1]
        
        global_offset = self.double_idx_to_one(file_idx, offset)
        offsets = global_offset - 1 + torch.arange(0, sample_length, dtype=torch.int64)
        neighbor_offset = self.neighbor_offsets[offsets]  # [L, k]
        all_tokens, all_reps, valid_masks = [], [], []
        for tgt_idx in range(len(target)):
            neighbor_tokens = []
            neighbor_reps = []    
            neighbors = neighbor_offset[tgt_idx]
            for offset in neighbors:
                if offset == -1:
                    continue
                if abs(offsets[tgt_idx] - offset) < self.invalid_context:
                    continue
                neighbor_tokens.append(self.neighbor_tokens[offset])
                neighbor_reps.append(self.quantize_neighbor_feat[offset])
            neighbor_reps = torch.from_numpy(np.array(neighbor_reps))
            neighbor_reps = self.tgt_quantizer.decode(neighbor_reps)
            all_tokens.append(neighbor_tokens)
            ks = neighbor_reps.shape[0]
            valid_mask = [1] * ks
            if ks < self.k:
                neighbor_reps = torch.cat([neighbor_reps, torch.zeros(self.k - ks, neighbor_reps.shape[1])])
                neighbor_tokens = neighbor_tokens + [torch.LongTensor((0, 0))] * (self.k - ks)
                valid_mask = valid_mask + [0] * (self.k - ks)
            all_reps.append(neighbor_reps)
            valid_masks.append(valid_mask)
        while len(all_reps) < self.block_size:
            all_reps.append(torch.zeros(self.k, neighbor_reps.shape[1]))
            valid_masks.append([0] * self.k)
        if len(source) < self.block_size:
            source = torch.cat(source, torch.LongTensor([self.pad] * (self.block_size - len(source))))
            target = torch.cat(target, torch.LongTensor([self.pad] * (self.block_size - len(target))))
        all_reps = torch.stack(all_reps, dim=0)
        # it seems we do not need neighbor_tokens
        return source, target, all_reps, valid_masks