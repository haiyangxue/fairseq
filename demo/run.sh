#!/bin/bash
if [ $# -ne 1 ]
then
    echo "usage: $0 + stage"
        exit
        fi
STAGE=$1

export CUDA_VISILBE_DEVICES=7

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#DATA=$root_dir/data/40w
DATA=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/fairseq/speech_recognition/datasets/fine-turning/asr_train
SRCDICT=$root_dir/data/vocab.zh
TGTDICT=$root_dir/data/vocab.en
#DATA_BIN=$root_dir/db/baidu_2.8w_r
#DATA_BIN=$root_dir/db/asr_trainset
DATA_BIN=$root_dir/db/baidu_trainset
SAVE=$root_dir/cp/test
MAX_EPOCH=10
MAX_TOKENS=700
N_WORKER=28
SRC=zh
TGT=en

if [ $STAGE = 1 ]
then

processing(){
 python $root_dir/../preprocess.py \
        -s $SRC -t $TGT \
        --trainpref $DATA/train \
        --validpref $DATA/valid \
        --srcdict $SRCDICT \
        --tgtdict $TGTDICT \
        --destdir $DATA_BIN\
        --workers  $N_WORKER

}

processing
fi

if [ $STAGE = 2 ]
then
#--arch audio_transformer \
#--task audio_translation \
#--save-interval-updates 10000 \
train(){

#MAX_TOKENS=20
#        --save-interval-updates 1000 \
#    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
#--reset-lr-scheduler \
#        --reset-meters \
#        --reset-optimizer \
#        --restore-file=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/cwmt/checkpoint_best.pt \
#        --save-interval-updates 10 \
#        --restore-file=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/tttt/checkpoint_1_10.pt \
#        --audio-pt /search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/pretrain_asr/checkpoint_15_124000.pt \
#        --mt-pt=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/cwmt/checkpoint_best.pt \

DATA_BIN=$root_dir/db/asr_trainset
MAX_TOKENS=400
SAVE=$root_dir/cp/multi_gpu_fuen_noiseall_alpha0.9
#        --restore-file=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/multi_gpu_fuen/checkpoint_1_1.pt \

 mkdir $SAVE
    cp -r ../fairseq/models $SAVE
    cp -r ../fairseq/modules $SAVE
    cp -r ./run.sh $SAVE
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0

    python $root_dir/../train.py $DATA_BIN \
        -s $SRC -t $TGT \
        --arch audio_transformer_wmt_en_de_big \
        --task audio_translation \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
        --warmup-updates 4000 \
        --lr 0.0003 --min-lr 1e-09 \
        --restore-file /search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/multi_gpu_fuen_alpha/checkpoint_1_1.pt \
        --encoder-layers 6 \
        --decoder-layers 6 \
        --dropout 0.2 --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens $MAX_TOKENS \
        --max-source-positions 60 \
        --max-target-positions 60 \
        --skip-invalid-size-inputs-valid-test \
        --max-epoch $MAX_EPOCH \
        --log-interval 10 \
        --keep-last-epochs 80\
        --save-dir $SAVE\
        --tensorboard-logdir $SAVE

#        -s $SRC -t $TGT \
#        --arch transformer_wmt_en_de_big \
#        --task translation \
#        --audio-pt /search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/pretrain_asr/checkpoint_15_124000.pt \
#        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07\
#        --warmup-updates 4000 \
#        --lr 0.0007 --min-lr 1e-09 \
#        --dropout 0.2 --weight-decay 0.0 \
#        --restore-file=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/cwmt/checkpoint_best.pt \
#        --skip-invalid-size-inputs-valid-test \
#        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#        --max-tokens $MAX_TOKENS --num-workers 3\
#        --max-source-positions 60 \
#        --max-target-positions 60 \
#        --max-epoch $MAX_EPOCH \
#        --keep-last-epochs 50\
#        --log-interval 1 \
#        --save-dir $SAVE\
#        --ddp-backend=no_c10d \
#        --tensorboard-logdir $SAVE

}
train
fi


if [ $STAGE = 3 ]
then

interactive_from_file(){

#    MODEL=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/test/checkpoint15.pt
    SRC_FILE=./test
    TGT_FILE=$DATA/test.tgt
    TGT_PRED=$root_dir/work/test.src.tgt
    LEN_PEN=1.0
    MAX_LEN=200
    NO_REPEAT=0
    BEAM_SIZE=4
    N_BEST=1

#    export CUDA_VISILBE_DEVICES=7
#    python $root_dir/../interactive.py $DATA_BIN\
#    -s $SRC -t $TGT \
#    --buffer-size 1024 \
#    --path $MODEL\
#    --input $SRC_FILE \
#    --remove-bpe \
#    --lenpen $LEN_PEN \
#    --nbest $N_BEST \
#    --max-len-b $MAX_LEN \
#    --no-repeat-ngram-size $NO_REPEAT \
#    --beam $BEAM_SIZE |tee res.tmp
MODEL=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/test_asr/checkpoint1.pt
DATA_BIN=$root_dir/db/asr_trainset
    CUDA_VISIBLE_DEVICES=7 python $root_dir/../generate.py $DATA_BIN \
    --path  $MODEL \
    --task audio_translation \
    --beam $BEAM_SIZE --batch-size 1 \
    --remove-bpe \
    --lenpen $LEN_PEN \
    --nbest $N_BEST \
    --max-len-b $MAX_LEN \
    --skip-invalid-size-inputs-valid-test \
    --no-repeat-ngram-size $NO_REPEAT \
    --gen-subset valid |tee res.tmp
    grep ^H res.tmp | cut -f1,3- | cut -c3-  | cut -f2-   > res3

}
interactive_from_file
fi


if [ $STAGE = 4 ]
then
cp_path=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp
cps="$cp_path/asr_our_nofuen_03_noise/checkpoint3.pt \
$cp_path/asr_our_encoder_03/checkpoint3.pt \
$cp_path/asr_our_noencoder_01/checkpoint3.pt \
$cp_path/asr_our_noencoder_03/checkpoint5.pt \
$cp_path/asr_our_fuen_03_noise/checkpoint3.pt"
#MODEL=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/test_base2/checkpoint13.pt
MODEL=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/cp/test_asr/checkpoint1.pt
#    SRC_FILE=/search/odin/haiyang/fairseq_exp/baseline/demo/test/test_asr_re.zh
    SRC_FILE=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/demo/test
    TGT_FILE=/search/odin/haiyang/fairseq_exp/baseline/demo/test/test_asr_re.en
    LEN_PEN=1.0
    MAX_LEN=400
    NO_REPEAT=0
    BEAM_SIZE=4
    N_BEST=1

DATA_BIN=$root_dir/db/asr_trainset
SRC=zh
TGT=en
for filename in $cps
do
CUDA_VISIBLE_DEVICES=7 python $root_dir/../interactive.py $DATA_BIN\
    -s $SRC -t $TGT \
    --buffer-size 1024 \
    --task audio_translation \
    --path $filename\
    --input $SRC_FILE \
    --remove-bpe \
    --lenpen $LEN_PEN \
    --nbest $N_BEST \
    --max-len-b $MAX_LEN \
    --no-repeat-ngram-size $NO_REPEAT \
    --beam $BEAM_SIZE

#exit
    done
fi
