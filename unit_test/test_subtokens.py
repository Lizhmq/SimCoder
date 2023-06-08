import h5py
from transformers import GPT2Tokenizer


token_p = "../data/py150k/codegpt_small_py_adaptedgpt2.subtoken_eval.h5"
model_p = "../pretrained_model/codegpt-small-py-adapted"

tokenizer = GPT2Tokenizer.from_pretrained(model_p)

"""
file_id, token_id, subtoken
"""
with h5py.File(token_p, "r") as f:
    tokenids = f["subtoken"]
    ids = [int(i) for i in tokenids[:100]]
    print(ids)
    print(tokenizer.convert_ids_to_tokens(ids))

"""
[50258, 2, 43907, 25, 7400, 11338, 28, 19, 6482, 10394, 28, 19, 2705, 8658, 11338, 28, 19, 1303, 220, 220, 220, 49962, 739, 262, 24843, 13789, 11, 10628, 362, 13, 15, 357, 1169, 366, 34156, 15341, 345, 743, 1303, 220, 220, 220, 407, 779, 428, 2393, 2845, 287, 11846, 351, 262, 13789, 13, 921, 743, 7330, 1303, 220, 220, 220, 257, 4866, 286, 262, 13789, 379, 1303, 1303, 220, 220, 220, 220, 220, 220, 220, 220, 2638, 1378, 2503, 13, 43073, 13, 2398, 14, 677, 4541, 14, 43, 2149, 24290, 12, 17, 13, 15, 1303, 1303, 220, 220, 220, 17486]
['<s>', '#', 'Ġvim', ':', 'Ġtab', 'stop', '=', '4', 'Ġshift', 'width', '=', '4', 'Ġsoft', 'tab', 'stop', '=', '4', 'Ġ#', 'Ġ', 'Ġ', 'Ġ', 'ĠLicensed', 'Ġunder', 'Ġthe', 'ĠApache', 'ĠLicense', ',', 'ĠVersion', 'Ġ2', '.', '0', 'Ġ(', 'the', 'Ġ"', 'License', '");', 'Ġyou', 'Ġmay', 'Ġ#', 'Ġ', 'Ġ', 'Ġ', 'Ġnot', 'Ġuse', 'Ġthis', 'Ġfile', 'Ġexcept', 'Ġin', 'Ġcompliance', 'Ġwith', 'Ġthe', 'ĠLicense', '.', 'ĠYou', 'Ġmay', 'Ġobtain', 'Ġ#', 'Ġ', 'Ġ', 'Ġ', 'Ġa', 'Ġcopy', 'Ġof', 'Ġthe', 'ĠLicense', 'Ġat', 'Ġ#', 'Ġ#', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġ', 'Ġhttp', '://', 'www', '.', 'apache', '.', 'org', '/', 'lic', 'enses', '/', 'L', 'IC', 'ENSE', '-', '2', '.', '0', 'Ġ#', 'Ġ#', 'Ġ', 'Ġ', 'Ġ', 'ĠUnless']
"""