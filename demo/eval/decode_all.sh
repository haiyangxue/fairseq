#!/bin/bash

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATA_BIN=$root_dir/../work/db
MAX_LEN=300
LEN_PEN=1
NO_REPEAT=0
BEAM_SIZE=5
N_BEST=1

CUDA_VISIBLE_DEVICES=1 python ../../generate.py $DATA_BIN \
    --path $1/$2  \
    --beam $BEAM_SIZE --batch-size 50 \
    --lenpen $LEN_PEN \
    --nbest $N_BEST \
    --max-len-b $MAX_LEN \
    --left-pad-source False \
    --left-pad-target False \
    --remove-bpe \
    --gen-subset test$3  > ./raw/raw.txt

grep ^H ./raw/raw.txt | cut -f1,3- | cut -c3- | sort -k1n | cut -f2-   > ${1}/res_files/${2}.${3}
#cat ${1}/res_files/${2}.${3} | ../scripts/recaser/detruecase.perl \
#| ../scripts/tokenizer/detokenizer.perl -l en > ${1}/res_files/plain.${2}.${3}
TGT_FILE=$4
#TGT_PRED=${1}/res_files/plain.${2}.${3}
TGT_PRED=${1}/res_files/${2}.${3}
perl $root_dir/../../scripts/multi-bleu.perl $TGT_FILE < $TGT_PRED


