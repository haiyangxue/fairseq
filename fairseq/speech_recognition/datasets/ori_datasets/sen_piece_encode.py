import sentencepiece as spm
import sys
path=sys.argv[1]
input=sys.argv[2]
out=sys.argv[3]
sp = spm.SentencePieceProcessor()
sp.Load(path)
with open(input) as f:
    with open(out,"w") as w:
        for item in f:
            w.write(" ".join(sp.EncodeAsPieces(item))+"\n")
