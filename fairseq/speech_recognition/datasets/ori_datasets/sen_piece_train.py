import sentencepiece as spm
import sys
args=" ".join(sys.argv[1:])

# args="--input=out_data/data/lang_char/input.txt --vocab_size=5000 --model_type=unigram --model_prefix=data/lang_char/train_960_unigram5000 --input_sentence_size=100000000 --unk_id=3 --eos_id=2 --pad_id=1 --bos_id=-1 --character_coverage=1"
spm.SentencePieceTrainer.Train(args)
