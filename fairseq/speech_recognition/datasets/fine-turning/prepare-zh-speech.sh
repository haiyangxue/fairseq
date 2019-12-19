#!/usr/bin/env bash

usr_dir=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/fairseq/speech_recognition/datasets/fine-turning
data_dir=$usr_dir/baidu
echo "Prepare train and dev jsons"
for part in train valid; do
    audio_dirs=$usr_dir/raw_data/${part}_seg
#audio_dirs=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/train/train2
    python ${usr_dir}/asr_char_prep_json.py --audio-dirs ${audio_dirs} --labels-src ${data_dir}/${part}.zh --labels-tgt ${data_dir}/${part}.en  --output ${data_dir}/${part}.json
done
echo "Prepare train and dev jsons Done!"

