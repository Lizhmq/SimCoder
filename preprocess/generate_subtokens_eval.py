# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:36:53 2022

@author: DrLC
"""

import pickle, gzip
import os
import torch
import numpy
import h5py
    

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device("cuda")
    bs = 38
    chunks = 100000
    block_size = 512
    print_each = 1000
    h5_path = "../data/py150k/codegpt_small_py_adaptedgpt2.subtoken_eval.h5"
    
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("../model/codegpt-small-py-adaptedgpt2")
    
    with gzip.open("../data/py150k/eval_token.pkl.gz") as f:
        data = pickle.load(f)
        
    i = 0
    seg = 0
    batch, save_n = [], []
    file_id = numpy.asarray([])
    token_id = numpy.asarray([])
    tokens = None
    tokens_all = numpy.asarray([])

    while i < len(data):
        
        d = data[i]
        if tokens is None:
            tokens = tokenizer.tokenize(d)
            tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
            tokens = tokenizer.convert_tokens_to_ids(tokens)
        sample = tokens[seg: seg + block_size]
        file_id = numpy.concatenate([file_id, numpy.asarray([i for _ in range(len(sample))])])
        token_id = numpy.concatenate([token_id, numpy.asarray([seg + j for j in range(len(sample))])])
        tokens_all = numpy.concatenate([tokens_all, sample])
        
        seg += block_size
        save_n.append(len(sample))
        pad_num = block_size - len(sample)
        sample += [tokenizer.pad_token_id for _ in range(pad_num)]
        batch.append(sample)
        
        if len(batch) >= bs:
            batch, save_n = [], []
            
        while file_id.shape[0] >= chunks:
            assert tokens_all.shape[0] == file_id.shape[0] and file_id.shape[0] == token_id.shape[0]
            save_tokens_all = numpy.asarray(tokens_all[:chunks])
            save_file_id = numpy.asarray(file_id[:chunks])
            save_token_id = numpy.asarray(token_id[:chunks])
            tokens_all = tokens_all[chunks:]
            file_id = file_id[chunks:]
            token_id = token_id[chunks:]
            
            with h5py.File(h5_path, "a") as f:
                try:
                    tokens_dset = f['subtoken']
                    tokens_dset.resize(tokens_dset.shape[0] + chunks, axis=0)
                    file_dset = f['file_id']
                    file_dset.resize(file_dset.shape[0] + chunks, axis=0)
                    token_dset = f['token_id']
                    token_dset.resize(token_dset.shape[0] + chunks, axis=0)
                except KeyError:
                    tokens_dset = f.create_dataset("subtoken", (chunks,), maxshape=(None, ),
                                                chunks=(chunks, ))
                    file_dset = f.create_dataset("file_id", (chunks, ), maxshape=(None, ),
                                                 chunks=(chunks, ))
                    token_dset = f.create_dataset("token_id", (chunks, ), maxshape=(None, ),
                                                 chunks=(chunks, ))
                tokens_dset[-chunks:] = save_tokens_all
                file_dset[-chunks:] = save_file_id
                token_dset[-chunks:] = save_token_id
            
        if seg >= len(tokens):
            seg = 0
            i += 1
            tokens = None
            if (i + 1) % print_each == 0:
                print ("%.4f..." % ((i+1)/len(data)*100))
            

