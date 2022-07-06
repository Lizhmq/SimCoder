# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:36:42 2022

@author: DrLC
"""

import tarfile
import os

def unzip(path="../data/py150_files.tar.gz", tgt_path="../data/py150k"):
    
    if os.path.isdir(tgt_path):
        return
    f = tarfile.open(path)
    f.extractall(tgt_path)
    f.close()
    
def unzip_data(path="../data/py150k"):
    
    if os.path.isdir(os.path.join(path, "data")):
        return
    f = tarfile.open(os.path.join(path, "data.tar.gz"))
    f.extractall(os.path.join(path, "data"))
    f.close()
    
if __name__ == "__main__":
    
    unzip()
    unzip_data()