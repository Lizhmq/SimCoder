# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:49:38 2022

@author: DrLC
"""

from tokenize import load_parser, obtain_token
import os
import pickle, gzip
import tqdm


def read_meta(dir_path="../data/py150k"):
    
    with open(os.path.join(dir_path, "python100k_train.txt"), "r") as f:
        train_files = f.readlines()
    with open(os.path.join(dir_path, "python50k_eval.txt"), "r") as f:
        eval_files = f.readlines()
    return train_files, eval_files
    
def read_data(files, dir_path="../data/py150k"):
    
    data_path = os.path.join(dir_path, "data")
    data = []
    err_num = 0
    for _p in files:
        p = os.path.join(data_path, _p.strip())
        try:
            with open(p, "r") as f:
                data.append(f.read())
        except UnicodeDecodeError:
            err_num += 1
    print ("TOTAL =", len(files), ", ERR =", err_num)
    return data

def tokenize_data(data, parser):
    
    ret = []
    err_num = 0
    for d in tqdm.tqdm(data):
        try:
            _bytes = bytes(d, "latin1")
            tree = parser.parse(_bytes)
            tokens = obtain_token(tree.root_node, _bytes)
        except UnicodeEncodeError:
            err_num += 1
        ret.append(" ".join(tokens))
    print ("TOTAL =", len(data), ", ERR =", err_num)
    return ret
    


if __name__ == "__main__":
    
    dir_path="../data/py150k"
    train_raw = os.path.join(dir_path, "train_raw.pkl.gz")
    eval_raw = os.path.join(dir_path, "eval_raw.pkl.gz")
    train_token = os.path.join(dir_path, "train_token.pkl.gz")
    eval_token = os.path.join(dir_path, "eval_token.pkl.gz")
    
    parser = load_parser()
    
    train_files, eval_files = read_meta(dir_path)
    
    if not os.path.isfile(train_raw):
        train_data = read_data(train_files, dir_path)
        with gzip.open(train_raw, "wb") as f:
            pickle.dump(train_data, f)
    else:
        with gzip.open(train_raw, "rb") as f:
            train_data = pickle.load(f)
            
    if not os.path.isfile(eval_raw):
        eval_data = read_data(eval_files, dir_path)
        with gzip.open(eval_raw, "wb") as f:
            pickle.dump(eval_data, f)
    else:
        with gzip.open(eval_raw, "rb") as f:
            eval_data = pickle.load(f)
            
    if not os.path.isfile(train_token):
        train_data = tokenize_data(train_data, parser)
        with gzip.open(train_token, "wb") as f:
            pickle.dump(train_data, f)
            
    if not os.path.isfile(eval_token):
        eval_data = tokenize_data(eval_data, parser)
        with gzip.open(eval_token, "wb") as f:
            pickle.dump(eval_data, f)
    