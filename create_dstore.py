import os, json
import numpy as np


def save_from_mat(keymat, valmat, vocab_size=50000, dstore_fp16=True, dstore_dir="./data/train_dstore"):
    """
    Key: (N, H), Value: (N, 2)
    """
    os.makedirs(dstore_dir, exist_ok=True)
    info = {
        "dstore_size": keymat.shape[0],
        "hidden_size": keymat.shape[1],
        "vocab_size": vocab_size,
        "dstore_fp16": dstore_fp16,
        "val_size": valmat.shape[1]
    }
    np.save(os.path.join(dstore_dir, "keys.npy"), keymat.astype(np.float16 if dstore_fp16 else np.float32))
    np.save(os.path.join(dstore_dir, "vals.npy"), valmat)
    json.dump(info, open(os.path.join(dstore_dir, "info.json"), "w"),
            sort_keys=True, indent=4, ensure_ascii=False)




def main():
    data_len = 10000
    keymat = np.random.random((data_len, 256)) * 100
    valmat = np.random.randint(0, 50000, (data_len, 2))
    save_from_mat(keymat, valmat)


if __name__ == '__main__':
    main()