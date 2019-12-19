#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
Shuffle parallel corpus
"""
import sys
import os

if len(sys.argv) != 3:
    sys.stderr.write('usage: %s + corpus1 + corpus2 ' % __file__)
    sys.exit(-1)

corpus1 = sys.argv[1]
corpus2 = sys.argv[2]
output_corpus1=corpus1+".shuf"
output_corpus2=corpus2+".shuf"

corpus1_list=[]
corpus2_list=[]
index=0
with open(corpus1, "r") as c1:
    with open(corpus2, "r") as c2:
        with open("./temp.txt", "w") as temp:
            sen1=c1.readline().strip()
            sen2=c2.readline()
            while sen1:
                index+=1
                if index%1000000==0:
                    print("reading data ... "+str(index))
                temp.write(sen1+"##&&"+sen2)
                # corpus1_list.append(sen1)
                # corpus2_list.append(sen2)
                sen1 = c1.readline().strip()
                sen2 = c2.readline()

print("shuffling data ... ")
os.system("shuf "+"./temp.txt -o ./temp.shuf.txt" )

with open("./temp.shuf.txt", "r") as temp:
    with open(output_corpus1, "w") as o1:
        with open(output_corpus2, "w") as o2:
            # sen1 = c1.readline().strip()
            sen = temp.readline()
            while sen:
                index += 1
                if index % 1000000 == 0:
                    print("reading data ... " + str(index))
                sen1=sen.split("##&&")[0]+"\n"
                sen2=sen.split("##&&")[1]
                o1.write(sen1)
                o2.write(sen2)

                sen = temp.readline()
