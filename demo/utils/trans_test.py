import os
import sys
a='his way of connecting with the family was always making us a dish , making us dinner , " louis gal@@ icia said .'
# b=a.replace('"','')
# print(b)
# exit()

file_path = "/search/odin/haiyang/fairseq_exp/t2t_decoder/test_data/wmt17test2001.src"
with open(file_path, "r") as f:
    # line=a.replace('"','\"')
    line = f.readline().strip().replace('"','\\"').replace('$', '\\$').replace('`', '\\`')
    while line:
        print(line)
        # exit()
        trans = os.popen(
            './client --ip 127.0.0.1 --port 10098 -t 1 -n 1 -a 0 --source_language en --target_language zh -i "' + line + '"').readlines()
        print(trans[0].strip())
        line = f.readline().strip().replace('"','\\"').replace('$', '\\$').replace('`', '\\`')
