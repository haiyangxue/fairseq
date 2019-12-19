#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
Convert pt model to npz
"""
import torch
import json
import numpy as np
import sys

if len(sys.argv) != 4:
    sys.stderr.write('usage: %s + pt_path + save_npz_path + para_name_json ' % __file__)
    sys.exit(-1)

pt_path = sys.argv[1]
save_npz_path = sys.argv[2]
para_name = sys.argv[3]
npz = {}
t2fs = {}
fs2t = {}
with open(para_name, "r") as f:
    t2fs = json.load(f)

for item in t2fs.keys():
    fs2t[t2fs[item]] = item
    # print(item+" : "+t2t2fairseq[item])
only_in_pt = []
model_dict = torch.load(pt_path)

for item in model_dict["model"].items():
    # print(item[0] + "   " + str(list(item[1].size())))
    if item[0] in fs2t.keys():
        npz[fs2t[item[0]]] = item[1].data.cpu().numpy()
    else:
        only_in_pt.append(item[0])
print("only_in_pt")
print(only_in_pt)

np.savez(save_npz_path, **npz)
print("Done!")
# data = np.load('test.npz')
# for item in data.items():
#     print(item)
