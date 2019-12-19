def get_dic():
    dic = {}
    dic[0] = "****"
    dic[1] = "<pad>"
    dic[2] = "<eos>"
    dic[3] = "<unk>"
    index = 3
    with open("/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/dict.txt")as dict:
        dict_list = dict.readlines()
        for item in dict_list:
            index += 1
            dic[index] = item.split(" ")[0]
    return dic
def print_zh(src,dic):
    srt=""
    for item in src:
        srt+=dic[item]
    print(srt)

test=[  28,  137,   25,  424,  797, 1389,  653,  301,   27, 1389,  653,   21,
          610,  301,   21,  230, 1189,   55,   33,   44,  383, 1962,  420, 1314,
           81,   16,   22,  110,  174,   21,  610,  301,   21,  230, 1189,   12,
          133,   68,    2]
dic=get_dic()
print_zh(test,dic)

