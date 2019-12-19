#!/bin/bash

DATA_BIN=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/db/asr_trainset
MAX_LEN=300
LEN_PEN=1
NO_REPEAT=0
BEAM_SIZE=5
N_BEST=1
input=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/test
CUDA_VISIBLE_DEVICES=$1 python /search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/interactive.py $DATA_BIN\
    -s zh -t en \
    --buffer-size 1024 \
    --task $3 \
    --path $2 \
    --input  $input\
    --remove-bpe \
    --lenpen $LEN_PEN \
    --nbest $N_BEST \
    --max-len-b $MAX_LEN \
    --no-repeat-ngram-size $NO_REPEAT \
    --beam $BEAM_SIZE


#grep ^H ./raw/raw_c.txt | cut -f1,3- | cut -c3- | sort -k1n | cut -f2-   > ${1}/res_files/${2}.${3}
##cat ${1}/res_files/${2}.${3} | ../scripts/recaser/detruecase.perl \
##| ../scripts/tokenizer/detokenizer.perl -l en > ${1}/res_files/plain.${2}.${3}
#TGT_FILE=$4
##TGT_PRED=${1}/res_files/plain.${2}.${3}
#TGT_PRED=${1}/res_files/${2}.${3}
#perl $root_dir/../../scripts/multi-bleu.perl $TGT_FILE < $TGT_PRED


