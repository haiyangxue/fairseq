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
    def __init__(self, gpu_id, cp, checkpoint_path,task):
        super(DecodeThread, self).__init__()
        # 调用父类的构造方法
        self.checkpoint_path = checkpoint_path
        self.cp = cp
        self.gpu_id = gpu_id
        self.task=task

    def run(self):
        # print(self.gpu_id)
        # print(
        #     "./decode_all_d.sh " + self.path + " " + self.cp + " " + str(self.i) + " " + self.test_path +
        #     self.test_path_name[
        #         self.i] + "." + self.sufixes + " " + str(self.gpu_id))
        self.res = os.popen(
            "./decode_all_c.sh " + self.gpu_id + " " + self.checkpoint_path + "/" + self.cp+" "+self.task).read()

    def get_result(self):
        return self.res


def decode_file(checkpoint_path, item, gpu_ids, tasks):
    threads = [DecodeThread(gpu_ids[ii], item[ii], checkpoint_path, tasks[ii]) for ii in range(len(item))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, items in enumerate(item):
        output = threads[i - 1].get_result()
        print(output)


if __name__ == "__main__":
    checkpoint_path = "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp"
    # item = ["asr_baseline_lr03_noise02/checkpoint17.pt",
    #         "asr_our_fuen_03_noise02/checkpoint17.pt","asr_our_fuen_03_noise_char/checkpoint6.pt",
    #         "asr_our_fuen_03_noise_char_mgpu/checkpoint6.pt","asr_our_fuen_03_noise_char_mgpu/checkpoint16.pt",
    #         "asr_our_fuen_03_noiseall_mgpu/checkpoint6.pt"]
    item = ["multi_gpu_fuen_noiseall_alpha0.3/checkpoint4.pt",
            "multi_gpu_fuen_noiseall_alpha0.5/checkpoint4.pt", "multi_gpu_fuen_noiseall_alpha0.7/checkpoint4.pt"
            ]
    # "asr_our_fuen_03_noiseall_mgpu/checkpoint3.pt"
    # "asr_our_fuen_03_noiseall_mgpu/checkpoint6.pt"
    gpu_ids = ["0", "1", "2", "3", "4", "5", "6", "7"]
    tasks = [ "audio_translation", "audio_translation", "audio_translation", "audio_translation", "audio_translation","audio_translation","audio_translation"]
    decode_file(checkpoint_path, item, gpu_ids, tasks)
