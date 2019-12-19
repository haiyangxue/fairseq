import os
import json
def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list

#THCHS30
def THCHS30():
    path="/search/odin/haiyang/fairseq_exp/asr/THCHS30/data_thchs30/dev"
    text="/search/odin/haiyang/fairseq_exp/asr/THCHS30/data_thchs30/dev.txt"
    index=0
    with open(text,"w") as w:
        for item in get_file_list(path):
            if item.split(".")[-1]=="trn":
                index+=1
                with open(path+"/"+item) as f:
                    sens=f.readlines()
                    print(str(index) + " " + item.split(".")[0] + " " + sens[0])
                    w.write(item.split(".")[0]+" "+sens[0])

def ST_CMDS():
    path="/search/odin/haiyang/fairseq_exp/asr/ST-CMDS/ST-CMDS-20170001_1-OS"
    text="/search/odin/haiyang/fairseq_exp/asr/ST-CMDS/text.txt"
    index=0
    with open(text,"w") as w:
        for item in get_file_list(path):
            if item.split(".")[-1]=="txt":
                index+=1
                with open(path+"/"+item) as f:
                    sens=f.readlines()
                    print(str(index)+" "+item.split(".")[0]+" "+sens[0])
                    w.write(item.split(".")[0]+" "+sens[0]+"\n")
def primewords():
    # index = 0
    # text = "/search/odin/haiyang/fairseq_exp/asr/primewords_md_2018_set1/text.txt"
    # with open("/search/odin/haiyang/fairseq_exp/asr/primewords_md_2018_set1/set1_transcript.json", 'r') as load_f:
    #     load_dict = json.load(load_f)
    #     with open(text, "w") as w:
    #         for item in load_dict:
    #             index += 1
    #             print(str(index) + " " + item["file"].split(".")[0] + " " + item["text"])
    #             w.write(item["file"].split(".")[0] + " " + item["text"] + "\n")
    #处理不一致
    index = 0
    text = "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/aishell_transcript_v0.8.txt"
    with open(text, 'r') as load_f:
        sens_list={}
        sens=load_f.readlines()
        for item in sens:
            sens_list[item.split(" ")[0]]=item
        with open(text+".2", "w") as w:
            for item in get_file_list("/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/data_aishell"):
                if item.split(".")[0] in sens_list:
                    index+=1
                    print(str(index)+" "+sens_list[item.split(".")[0]])
                    w.write(sens_list[item.split(".")[0]])

def aidatatang():
    #解压
    # path="/search/odin/haiyang/fairseq_exp/asr/aidatatang_200zh/corpus/dev"
    # index = 0
    # for item in get_file_list(path):
    #     index += 1
    #     print(str(index) + " " + item)
    #     os.system("tar -xzvf "+path+"/"+item+" -C " +path)
    #得到trans
    path = "/search/odin/haiyang/fairseq_exp/asr/aidatatang_200zh/corpus/dev"
    text = "/search/odin/haiyang/fairseq_exp/asr/aidatatang_200zh/dev.txt"
    index = 0
    with open(text, "w") as w:
        for item in get_file_list(path):
            if os.path.isdir(path + "/" + item):

                for item2 in get_file_list(path + "/" + item):
                    if item2.split(".")[-1] == "txt":
                        index += 1
                        with open(path + "/" + item+ "/" + item2) as f:
                            sens = f.readlines()
                            print(str(index) + " " + item2.split(".")[0] + " " + sens[0])
                            w.write((item2.split(".")[0] + " " + sens[0]).strip()+"\n")
def data_aishell():
    path="/search/odin/haiyang/fairseq_exp/asr/data_aishell/wav"
    index = 0
    for item in get_file_list(path):
        index += 1
        print(str(index) + " " + item)
        os.system("tar -xzvf "+path+"/"+item+" -C " +path)

def dev():
    path = "/search/odin/haiyang/fairseq_exp/asr/THCHS30/data_thchs30/dev"
    path2="/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/THCHS30-train"
    path3="/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/dev/"
    text = "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/THCHS30-train/text.txt"
    text2 = "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/dev/text.txt"

    index = 0
    with open(text2, "w") as w:
        # with open(text+".2", "w") as w2:
            with open(text) as f:
                sens_list = {}
                sens = f.readlines()
                for item in sens:
                    sens_list[item.split(" ")[0]] = item
                for item2 in get_file_list(path):
                    # os.system("mv "+path2+"/"+item2.split(".")[0]+".wav "+path3)
                    w.write(sens_list[item2.split(".")[0]])

index = 0
text = "/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/test.token.txt"
with open(text, 'r') as load_f:
        sens_list={}
        sens=load_f.readlines()
        for item in sens:
            sens_list[item.split(" ")[0]]=item
        with open(text + ".2", "w") as w:
            for item in get_file_list("/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/train2"):
                if item.split(".")[0] in sens_list:
                    index+=1
                    print(str(index)+" "+sens_list[item.split(".")[0]])
                    w.write(sens_list[item.split(".")[0]])
