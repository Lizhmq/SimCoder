import os, json
import h5py
import numpy as np
from tqdm import tqdm


def h52np(h5_file, np_path):
    """
    Reads a HDF5 file and returns a numpy array.

    :param h5_file: HDF5 file to read
    :param np_path: Output path to the numpy file
    """

    os.makedirs(np_path, exist_ok=True)
    f = h5py.File(h5_file, "r")
    dstore_size, hidden_size = f["representation"].shape
    print(f["representation"].shape)
    print(f["file_id"].shape)
    chunk_size = 100000
    ["file_id"]
    ["token_id"]
    key_file = os.path.join(np_path, "keys.npy")
    val_file = os.path.join(np_path, "vals.npy")
    keys = np.memmap(key_file,
                        dtype=np.float16,
                        mode="w+",
                        shape=(dstore_size, hidden_size))
    vals = np.memmap(val_file,
                        dtype=np.int32,
                        mode="w+",
                        shape=(dstore_size, 2))
    for i in tqdm(range(0, dstore_size, chunk_size)):
        keys[i:i+chunk_size] = f["representation"][i:i+chunk_size].astype(np.float16)
        vals[i:i+chunk_size] = np.concatenate((f["file_id"][i:i+chunk_size].reshape(-1, 1),\
                                               f["token_id"][i:i+chunk_size].reshape(-1, 1)), axis=1)
        keys.flush()
        vals.flush()
    
    info = {
        "dstore_size": dstore_size,
        "hidden_size": hidden_size,
        "vocab_size": 50000,            # TODO: get this from the h5 file
        "dstore_fp16": True,
        "val_size": 2
    }
    json.dump(info, open(os.path.join(np_path, "info.json"), "w"),
            sort_keys=True, indent=4, ensure_ascii=False)
    return

# h52np("/home/zhanghz/mem-lm/data/py150k/codegpt_small_py_adaptedgpt2.representation.h5", "./data/train_dstore/")
h52np("/home/zhanghz/mem-lm/data/py150k/codegpt_small_py_adaptedgpt2.representation_eval.h5", "./data/test_dstore/")