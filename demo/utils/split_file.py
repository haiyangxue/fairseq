#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
split big train data
"""
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join
import threading
import sys


def split(data_path, target, split_num, output_dir, src, tgt):
    files_list = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    print("train." + src)
    if "train." + src not in files_list or "train." + tgt not in files_list:
        print("dir must contain train dataset")
        exit()
    train_src = "train." + src
    train_tgt = "train." + tgt

    if "valid." + src not in files_list or "valid." + tgt not in files_list:
        print("dir must contain valid dataset")
        exit()

    if "vocab." + src not in files_list or "vocab." + tgt not in files_list:
        print("dir must contain vocab ")
        exit()

    if target == src:
        train_name = train_src
        train_path = join(data_path, train_src)

    else:
        train_name = train_tgt
        train_path = join(data_path, train_tgt)

    index = 0
    name = 1
    dataList = []
    print("split...")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    with open(train_path, 'r') as f_source:
        for line in f_source:
            index += 1
            dataList.append(line)
            if index % 1000000 == 0:
                print(train_name + " " + str(index))
            if index % split_num == 0:
                if not os.path.exists(output_dir + "/data" + str(name)):
                    try:
                        os.makedirs(output_dir + "/data" + str(name))
                    except:
                        continue
                with open(output_dir + "/data" + str(name) + "/" + train_name, 'w+') as f_target:
                    for data in dataList:
                        f_target.write(data)
                name += 1
                dataList = []

    if len(dataList) != 0:
        # 将剩余行追加到最后一个文件
        with open(output_dir + "/data" + str(name - 1) + "/" + train_name, 'a') as f_target:
            for data in dataList:
                f_target.write(data)

    print("Done!")
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == "__main__":
    data_path = sys.argv[1]
    split_num = int(sys.argv[2])
    line_num = int(sys.argv[3].split(" ")[0])
    num = int(line_num / split_num)
    src = sys.argv[5]
    tgt = sys.argv[6]
    split_name = sys.argv[7]
    print(sys.argv)

    target = src
    output_dir = join(data_path, split_name)
    print(num)
    t1 = threading.Thread(target=split, args=(data_path, target, num, output_dir, src, tgt))
    t1.start()

    target = tgt
    t2 = threading.Thread(target=split, args=(data_path, target, num, output_dir, src, tgt))

    t2.start()
