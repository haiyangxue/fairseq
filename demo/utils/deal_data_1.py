#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
清理含有unk数据
"""
import sys
index=0
with open(sys.argv[3], "w") as w1:
    with open(sys.argv[4], "w") as w2:
        with open(sys.argv[1], "r") as f1:
            with open(sys.argv[2], "r") as f2:

                sen1_list=f1.readlines()
                sen2_list=f2.readlines()
                for sen1,sen2 in zip(sen1_list,sen2_list):
                    if "<unk>" not in sen2:
                        index += 1
                        w1.write(sen1)
                        w2.write(sen1.split(" ")[0]+" "+sen2)

                print(index)



