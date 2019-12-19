#!/bin/bash
if [ $# -ne 1 ]
then
    echo "usage: $0 + stage"
        exit
        fi
STAGE=$1

export CUDA_VISILBE_DEVICES=7

root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
project_dir=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq
MODEL_PATH=$root_dir/cp/asr_zh
DIR_FOR_PREPROCESSED_DATA=$root_dir/datasets/zh_asr_data
if [ $STAGE = 2 ]
then

train(){
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
#export CUDA_VISIBLE_DEVICES=7
MODEL_PATH=$root_dir/cp/asr_zh
RE_MODEL_PATH=$root_dir/cp/asr_zh
#    --restore-file $RE_MODEL_PATH/checkpoint_3_18000.pt \

python -u $project_dir/train.py $DIR_FOR_PREPROCESSED_DATA \
    --save-dir $MODEL_PATH \
    --max-epoch 80 \
    --task speech_recognition \
    --arch vggtransformer_2 \
    --optimizer adadelta \
    --lr 1.0 \
    --adadelta-eps 1e-8 \
    --adadelta-rho 0.95 \
    --clip-norm 10.0  \
    --max-tokens 5000 \
    --log-format json \
    --log-interval 10 \
    --update-freq 1 \
    --num-workers 0 \
    --save-interval-updates 2000 \
    --criterion cross_entropy_acc \
    --user-dir $root_dir

}
train
fi
if [ $STAGE = 3 ]
then

inference(){
export CUDA_VISIBLE_DEVICES=7
MODEL_PATH=$root_dir/cp/asr_zh
sr_path=/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition
SET=valid
RES_DIR=$sr_path/res
RES_REPORT=$sr_path/res/report
python -u infer.py $DIR_FOR_PREPROCESSED_DATA \
    --task speech_recognition \
    --max-tokens 25000 \
    --nbest 1 \
    --path $MODEL_PATH/checkpoint_9_74000.pt \
    --beam 10 \
    --results-path $RES_DIR \
    --batch-size 40 \
    --gen-subset $SET \
    --user-dir $sr_path

#sclite -r ${RES_DIR}/ref.word-checkpoint_last.pt-${SET}.txt -h ${RES_DIR}/hypo.word-checkpoint_last.pt-${SET}.txt -i rm -o all stdout > $RES_REPORT

}
inference
fi

if [ $STAGE = 4 ]
then

export CUDA_VISIBLE_DEVICES=7
MODEL_PATH=$root_dir/cp/test
RE_MODEL_PATH=$root_dir/cp/asr_zh

python -u $project_dir/train.py $DIR_FOR_PREPROCESSED_DATA \
    --save-dir $MODEL_PATH \
    --max-epoch 80 \
    --task speech_recognition \
    --arch vggtransformer_2 \
    --optimizer adadelta \
    --lr 1.0 \
    --adadelta-eps 1e-8 \
    --adadelta-rho 0.95 \
    --clip-norm 10.0  \
    --max-tokens 5000 \
    --log-format json \
    --log-interval 1000 \
    --update-freq 1 \
    --num-workers 0 \
    --restore-file $RE_MODEL_PATH/checkpoint_best.pt \
    --save-interval-updates 1 \
    --criterion cross_entropy_acc \
    --user-dir $root_dir

fi
