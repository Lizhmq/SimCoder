# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:58:28 2022

@author: DrLC
"""

import pickle, gzip
import os
import torch
import numpy
import h5py


def run_model(model, batch, save_n, device):
    
    rep = []
    _input = torch.tensor(batch).to(device)
    hidden = model(input_ids=_input,
                   output_hidden_states=True,
                   return_dict=True).hidden_states[-1]
    for i, h in enumerate(hidden):
        for _h in h[:save_n[i]]:
            rep.append(_h.cpu().detach().numpy())
    return numpy.asarray(rep)
    

def write_h5_file(h5_path, chunks, save_rep, save_file_id, save_token_id):
    
    with h5py.File(h5_path, "a") as f:
        try:
            rep_dset = f['representation']
            rep_dset.resize(rep_dset.shape[0] + chunks, axis=0)
            file_dset = f['file_id']
            file_dset.resize(file_dset.shape[0] + chunks, axis=0)
            token_dset = f['token_id']
            token_dset.resize(token_dset.shape[0] + chunks, axis=0)
        except KeyError:
            rep_dset = f.create_dataset("representation", (chunks, 768), maxshape=(None, 768),
                                        chunks=(chunks, 768))
            file_dset = f.create_dataset("file_id", (chunks, ), maxshape=(None, ),
                                         chunks=(chunks, ))
            token_dset = f.create_dataset("token_id", (chunks, ), maxshape=(None, ),
                                         chunks=(chunks, ))
        rep_dset[-chunks:] = save_rep
        file_dset[-chunks:] = save_file_id
        token_dset[-chunks:] = save_token_id
    
    

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    device = torch.device("cuda")
    bs = 38
    chunks = 100000
    block_size = 512
    print_each = 1000
    h5_path = "../data/py150k/codegpt_small_py_adaptedgpt2.representation_eval.h5"
    
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained("../model/codegpt-small-py-adaptedgpt2")
    model = GPT2LMHeadModel.from_pretrained("../model/codegpt-small-py-adaptedgpt2").to(device)
    model.eval()
    
    with gzip.open("../data/py150k/eval_token.pkl.gz") as f:
        data = pickle.load(f)
        
    i = 0
    seg = 0
    batch, save_n = [], []
    rep = None
    file_id = numpy.asarray([])
    token_id = numpy.asarray([])
    tokens = None

    while i < len(data):
        
        d = data[i]
        if tokens is None:
            tokens = tokenizer.tokenize(d)
            tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
            tokens = tokenizer.convert_tokens_to_ids(tokens)
        sample = tokens[seg: seg + block_size]
        file_id = numpy.concatenate([file_id, numpy.asarray([i for _ in range(len(sample))])])
        token_id = numpy.concatenate([token_id, numpy.asarray([seg + j for j in range(len(sample))])])
        
        seg += block_size
        save_n.append(len(sample))
        pad_num = block_size - len(sample)
        sample += [tokenizer.pad_token_id for _ in range(pad_num)]
        batch.append(sample)
        
        if len(batch) >= bs:
            if rep is None:
                rep = run_model(model, batch, save_n, device)
            else:
                rep = numpy.concatenate([rep, run_model(model, batch, save_n, device)])
            batch, save_n = [], []
            
        while rep is not None and rep.shape[0] >= chunks:
            assert rep.shape[0] == file_id.shape[0] and file_id.shape[0] == token_id.shape[0]
            save_rep = numpy.asarray(rep[:chunks])
            save_file_id = numpy.asarray(file_id[:chunks])
            save_token_id = numpy.asarray(token_id[:chunks])
            rep = rep[chunks:]
            file_id = file_id[chunks:]
            token_id = token_id[chunks:]
            
            write_h5_file(h5_path, chunks, save_rep, save_file_id, save_token_id)
            
        if seg >= len(tokens):
            seg = 0
            i += 1
            tokens = None
            if (i + 1) % print_each == 0:
                print ("%.4f..." % ((i+1)/len(data)*100))
            

    print (rep.shape)
    if rep.shape[0]:
        
        n_pad = chunks-rep.shape[0]
        rep = numpy.concatenate([rep, numpy.zeros([n_pad, 768])])
        file_id = numpy.concatenate([file_id, numpy.ones([n_pad, ])*-1])
        token_id = numpy.concatenate([token_id, numpy.ones([n_pad, ])*-1])
        
        write_h5_file(h5_path, chunks, rep, file_id, token_id)
                
