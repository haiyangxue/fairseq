#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
Eval Test
"""

import sys
import os
import threading


class DecodeThread(threading.Thread):
    def __init__(self, path, cp, i, test_path, test_path_name, sufixes, gpu_id):
        super(DecodeThread, self).__init__()
        # 调用父类的构造方法
        self.path = path
        self.cp = cp
        self.i = i
        self.test_path = test_path
        self.test_path_name = test_path_name
        self.sufixes = sufixes
        self.res = None
        self.gpu_id = gpu_id

    def run(self):
        print(
            "./decode_all_d.sh " + self.path + " " + self.cp + " " + str(self.i) + " " + self.test_path +
            self.test_path_name[
                self.i] + "." + self.sufixes + " " + str(self.gpu_id))
        self.res = os.popen(
            "./decode_all_d.sh " + self.path + " " + self.cp + " " + str(self.i) + " " + self.test_path +
            self.test_path_name[
                self.i] + "." + self.sufixes + " " + str(self.gpu_id)).read()

    def get_result(self):
        return self.res


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list


def decode_one_gpu(path, cp, i, test_path, test_path_name, sufixes, gpu_id):
    output = os.popen(
        "./decode_all_d.sh " + path + " " + cp + " " + str(i) + " " + test_path + test_path_name[
            i] + "." + sufixes + " " + str(gpu_id)).read()
    return output


def decode_file(path, cp, test_path, test_path_name, sufixes, use_threads=False,one_cp=False):
    score = 0
    score_6 = 0
    threads = None
    res_str = ''
    if cp.split(".")[-1] == "pt" and cp != "checkpoint_best.pt" and cp != "checkpoint_last.pt":
        print(use_threads)
        if use_threads:
            threads = [DecodeThread(path, cp, ii + 1, test_path, test_path_name, sufixes, ii + 1) for ii in range(7)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        for i in range(1, 8):
            with open(path + "/result/" + test_path_name[i], "a") as res:

                if use_threads:
                    print("./decode_all_d.sh " + path + " " + cp + " " + str(i) + " " + test_path + test_path_name[
                        i] + "." + sufixes + " " + str(i))
                    output = threads[i - 1].get_result()
                else:
                    print("./decode_all_d.sh " + path + " " + cp + " " + str(i) + " " + test_path + test_path_name[
                        i] + "." + sufixes + " " + str(7))
                    output = decode_one_gpu(path, cp, i, test_path, test_path_name, sufixes, 7)
                print(output)
                if not one_cp:
                    res.write(cp + " : " + output[7:12] + "\n")
                res_str = res_str + output[7:12] + ", "
                score += float(output[7:12])
                if test_path_name[i] != "keke0626":
                    score_6 += float(output[7:12])
        print(cp + " : " + res_str + " : " + str(round(score / 7, 2)) + " : " + str(round(score_6 / 6, 2)) + "\n")
        with open(path + "/result/avg", "a") as res:

            if not one_cp:
                res.write(
                    cp + " : " + res_str + " : " + str(round(score / 7, 2)) + " : " + str(round(score_6 / 6, 2)) + "\n")


def decode_from_start(path, test_path_name, test_path, sufixes, use_threads=False):
    if not os.path.exists(path + "/result/"):
        os.mkdir(path + "/result/")
    if not os.path.exists(path + "/res_files/"):
        os.mkdir(path + "/res_files2/")
    for cp in get_file_list(path):
        decode_file(path, cp, test_path, test_path_name, sufixes, use_threads=use_threads)


def decode_from_record(path, test_path_name, test_path, sufixes, use_threads=False):
    with open(path + "/result/" + test_path_name[-1], "r") as trans_r:
        trans_list = trans_r.readlines()
        for index in range(len(trans_list)):
            trans_list[index] = trans_list[index].split(" : ")[0]
    cur_cps = get_file_list(path)
    last_trans_index = cur_cps.index(trans_list[-1])
    # print(trans_list[-1])
    # print(last_trans_index)
    # exit()
    # for cp in get_file_list(path):
    if last_trans_index + 1 < len(cur_cps):
        for index in range(last_trans_index + 1, len(cur_cps)):
            cp = cur_cps[index]
            decode_file(path, cp, test_path, test_path_name, sufixes, use_threads=use_threads)


if __name__ == "__main__":
    checkpoint_path = "/search/odin/haiyang/fairseq_exp/baseline/demo/cp/shuf_words"
    testset_path = "/search/odin/haiyang/fairseq_exp/baseline/demo/data/test_data/test/"

    testset_name = ["", "ime3k201612", "iw2013", "keke0626", "news1k201701", "wmt17dev2002",
                    "wmt17test2001", "test"]
    last_test_res = checkpoint_path + "/result/" + testset_name[-1]
    tgt_sufixes = "en"
    one_cp=True
    cp="checkpoint_40_560000.pt"
    if one_cp:
        decode_file(checkpoint_path, cp, testset_path, testset_name, tgt_sufixes, use_threads=True, one_cp=one_cp)
    else:
        if os.path.exists(last_test_res) and os.path.getsize(last_test_res):
            # 如果最后一个测试集被翻，则这个checkpoint翻译完毕
            decode_from_record(checkpoint_path, testset_name, testset_path, tgt_sufixes, use_threads=True)
        else:
            decode_from_start(checkpoint_path, testset_name, testset_path, tgt_sufixes, use_threads=True)
