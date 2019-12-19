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

if len(sys.argv) != 2:
    sys.stderr.write('usage: %s + pt_path ' % __file__)
    print()
    sys.exit(-1)

pt_path = sys.argv[1]

only_in_pt = []
model_dict = torch.load(pt_path)

for item in model_dict["model"].items():
    if "encoder.transformer_layers.1.fc1.weight" in item[0]:
        print(item[0] + "   " + str(item[1]))

    # print(item[0] + "   " + str(list(item[1].size())))
print("size： "+str(len(model_dict["model"].items())))


