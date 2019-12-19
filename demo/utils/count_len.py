#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
Convert npz model to tensorflow
"""
import sys
with open(sys.argv[1], "r") as f1:
    num = 0
    index = 0
    sen = f1.readline()
    lens = {}
    while sen:
        index += 1
        if index % 1000000 == 0:
            print(index)
        words = sen.strip().split(" ")
        sen_len = len(words)
        # print(sen_len)
        # exit()
        if (int(sen_len / 10)) in lens.keys():
            lens[int(sen_len / 10)] += 1
        else:
            lens[int(sen_len / 10)] = 1
        if sen_len <= 100:
            num += 1
        # else:
        #     print(words)
        #     print(sen_len)
        sen = f1.readline()

print(num)
print(index)

for item in sorted(lens.keys()):
       print(str(item)+":"+str(lens[item]) )
