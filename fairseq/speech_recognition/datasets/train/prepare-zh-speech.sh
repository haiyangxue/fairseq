#!/usr/bin/env bash

usr_dir=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition
audio_dirs=$usr_dir/datasets/zh_asr_data
dictionary=$audio_dirs/dict.txt
echo "Prepare train and dev jsons"
for part in train dev; do
echo "python ${usr_dir}/datasets/asr_char_prep_json.py --audio-dirs ${audio_dirs}/${part} --labels ${audio_dirs}/${part}.txt  --dictionary ${dictionary} --output ${part}.json"
    python ${usr_dir}/datasets/asr_char_prep_json.py --audio-dirs ${audio_dirs}/${part} --labels ${audio_dirs}/${part}.txt  --dictionary ${dictionary} --output ${part}.json
done
echo "Prepare train and dev jsons Done!"

