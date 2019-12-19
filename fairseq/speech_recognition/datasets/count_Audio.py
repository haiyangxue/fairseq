import json
count_time=0
index=0
with open('/search/odin/haiyang/fairseq_exp/e2e_trans/fairseq/examples/speech_recognition/datasets/zh_asr_data/train.json', 'r') as f:
    data = json.load(f)
    for item in data["utts"].keys():
        count_time+=data["utts"][item]["input"]["length_ms"]
        index+=1
print(count_time)
print(count_time/3600000)
print(index)
