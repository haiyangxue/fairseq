#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections import namedtuple
import concurrent.futures
from itertools import chain
import argparse
import os
import json
import multiprocessing
import torchaudio

from fairseq.data import Dictionary

MILLISECONDS_TO_SECONDS = 0.001


def process_sample(aud_path, lable_src,lable_tgt,minlength_ms, utt_id):
    input = {}
    output = {}
    si, ei = torchaudio.info(aud_path)

    length_ms = int(si.length / si.channels / si.rate / MILLISECONDS_TO_SECONDS)
    if length_ms > minlength_ms:
        input["length_ms"] = length_ms
        input["path"] = aud_path
        input["src_input"] = lable_src[1].strip()
        output["tgt_output"] = lable_tgt[1].strip()
        assert lable_src[0] == lable_tgt[0]
        return {lable_src[0]: {"input": input, "output": output}}
    else:
        return {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dirs", nargs="+", default=['-'], required=True,
                        help="input directories with audio files")
    parser.add_argument("--labels-src", required=True,
                        help="aggregated src input labels with format <ID LABEL> per line",
                        type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument("--labels-tgt", required=True,
                        help="aggregated tgt input labels with format <ID LABEL> per line",
                        type=argparse.FileType('r', encoding='UTF-8'))

    parser.add_argument("--audio-format", choices=["flac", "wav"], default="wav")
    parser.add_argument("--output", required=True, type=argparse.FileType('w'),
                        help="path to save json output")
    parser.add_argument("--minlength-ms", default=0,
                        help="path to save json output")
    args = parser.parse_args()

    # with open("/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/fairseq/speech_recognition/datasets/fine-turning/train.zh2","w") as zh:
    #     with open(
    #         "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/fairseq/speech_recognition/datasets/fine-turning/train.en2",
    #         "w") as en:
    #         for line_src,line_tgt in zip(args.labels_src,args.labels_tgt):
    #             (utt_id, label) = line_src.split(" ", 1)
    #             si, ei = torchaudio.info("/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/fairseq/speech_recognition/datasets/fine-turning/raw_data/train_seg/"+utt_id+".wav")
    #             length_ms = int(si.length / si.channels / si.rate / MILLISECONDS_TO_SECONDS)
    #             if length_ms>500:
    #                 zh.write(line_src)
    #                 en.write(line_tgt)
    # exit()
    labels_src = {}
    index_id=0
    for line in args.labels_src:
        (utt_id, label) = line.split(" ", 1)
        labels_src[utt_id] = (index_id,label)
        index_id+=1
        # print(str(index_id)+" "+utt_id)
    if len(labels_src) == 0:
        raise Exception('No labels found in ', args.labels_src)

    labels_tgt = {}
    index_id=0
    for line in args.labels_tgt:
        (utt_id, label) = line.split(" ", 1)
        labels_tgt[utt_id] = (index_id,label)
        if utt_id not in labels_src.keys():
            print(index_id)
            raise Exception('Inconsistent labels',utt_id)
        index_id+=1

    if len(labels_tgt) == 0:
        raise Exception('No labels found in ', args.labels_tgt)

    Sample = namedtuple('Sample', 'aud_path utt_id')
    samples = []
    # print(args.audio_dirs)
    for path, _, files in chain.from_iterable(os.walk(path) for path in args.audio_dirs):
        # print(path)
        # print(files)
        for f in files:
            if f.endswith(args.audio_format):
                if len(os.path.splitext(f)) != 2:
                    raise Exception('Expect <utt_id.extension> file name. Got: ', f)
                utt_id = os.path.splitext(f)[0]
                if utt_id not in labels_src:
                    continue
                samples.append(Sample(os.path.join(path, f), utt_id))

    utts = {}
    num_cpu = multiprocessing.cpu_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpu) as executor:
        future_to_sample = {executor.submit(process_sample, s.aud_path, labels_src[s.utt_id], labels_tgt[s.utt_id],int(args.minlength_ms),s.utt_id): s for s in samples}
        index=0
        for future in concurrent.futures.as_completed(future_to_sample):
            if future.result() is not None:
                try:
                    data = future.result()
                    index += 1
                except Exception as exc:
                    print('generated an exception: ', exc)
                else:
                    utts.update(data)

    print(len(utts))
    json.dump({"utts": utts}, args.output,ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
