import torch
import numpy
import json
import random


def as_text(v):  ## 生成unicode字符串
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def is_text(v):
    return isinstance(v, str)


def isChinese(s):
    s = as_text(s)
    return all(u'\u4e00' <= c <= u'\u9fff' or c == u'〇' for c in s)


def random_picks_num(weight_data):
    total = sum(weight_data.values())
    ra = random.uniform(0, total)
    curr_sum = 0
    ret = None
    keys = weight_data.keys()
    for k in keys:
        curr_sum += weight_data[k]
        if ra <= curr_sum:
            ret = k
            break
    return ret


def random_picks(weight_data, char, chosen_index):
    count = 0
    while count < 10:
        count += 1
        index = numpy.random.randint(0, len(weight_data))
        chose_key = list(weight_data.keys())[index]
        if chose_key not in chosen_index and chose_key != char:
            break

    return chose_key


def char22id(dict_path):
    char2id = {}
    id2char = {}

    with open(dict_path, "r") as f:
        chars = f.readlines()
        for index, char in enumerate(chars):
            char = char.split(" ")[0]
            char2id[char] = index + 4
            id2char[index + 4] = char

    return char2id, id2char


def begin_chars(begin_path, char2id):
    begin_char_ids = []
    with open(begin_path, "r") as f:
        chars = f.readlines()
    for char in chars:
        if char.strip() in char2id:
            begin_char_ids.append(char2id[char.strip()])

        else:
            charBPE = []
            charBPE.append(char2id[char.strip()[0] + "@@"])
            charBPE.append(char2id[char.strip()[1]])

            begin_char_ids.append(charBPE)

            # begin_char_ids.append(char2id[char.strip()])

    return begin_char_ids


def noise_chars(noise_path, char2id):
    noise_char_ids = []
    with open(noise_path, "r") as f:
        chars = f.readlines()
    for char in chars:
        if char.strip() in char2id:

            noise_char_ids.append(char2id[char.strip()])

        else:
            # print(char.strip())
            # exit()
            charBPE = []
            charBPE.append(char2id[char.strip()[0] + "@@"])
            charBPE.append(char2id[char.strip()[1]])

            noise_char_ids.append(charBPE)

    return noise_char_ids


def print_sen(src, id2char):
    for id in src:
        if id not in [0, 1, 2, 3]:
            print(id2char[id], end=" ")
    print()


def char_to_pinyin(path):
    char_to_pinyin = {}
    with open(path, 'r') as load_f:
        pinyin_dic = json.load(load_f)

    for (k, v) in pinyin_dic.items():
        for (k2, v2) in v.items():
            char_to_pinyin[k2] = k
            # print(str(k2.encode("utf-8")) + " " + str(k.encode("utf-8")))
    return char_to_pinyin


def get_Char22py(homophone_table):
    char2py = {}
    py2chars = {}
    for k1 in list(homophone_table.keys()):
        py2chars[k1] = homophone_table[k1].keys()
        for k2 in list(homophone_table[k1].keys()):
            char2py[k2] = k1

    return char2py, py2chars

    # def test(id2char, char2id, char2py, homophone_table):
    #     index_yes=0
    #     index_no=0
    #     for item in char2py:
    #         if item  in char2id.keys():
    #             print("Yes",end=" ")
    #             print(item)
    #             index_yes+=1
    #         else:
    #             print("No", end=" ")
    #             print(item)
    #             index_no+=1
    #     print(index_yes)
    #     print(index_no)
    #     exit()


def avoid_special_id(srcs_numpy):
    select_time = 0
    while True:
        # print("chose id ....")
        select_time += 1
        index = numpy.random.randint(0, len(srcs_numpy))
        id = srcs_numpy[index]
        if id not in [0, 1, 2, 3]:
            break
        if select_time >= 10:
            return None, None

    return index, id


def drop_word(srcs, symbol_drop, id2char=None):
    srcs_numpy = srcs.data.cpu().numpy()
    # for item in srcs_numpy:
    #     print_sen(item, id2char)
    indexs = ((torch.cuda.FloatTensor(srcs.shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
    if len(srcs_numpy[0]) > 3:
        for index in indexs:
            chosen_id = srcs_numpy[index[0]][index[1]]
            if chosen_id not in [0, 1, 2]:
                srcs_numpy[index[0]][index[1]] = 2  # unk
    srcs = torch.Tensor(srcs_numpy).cuda().long()
    return srcs


def drop_word_pad(srcs, symbol_drop, id2char=None):
    srcs_numpy = srcs.data.cpu().numpy()
    src_row = []

    for i in range(len(srcs)):
        # print_sen(srcs_numpy[i], id2char)
        indexs = ((torch.cuda.FloatTensor(srcs[i].shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero().view(1,
                                                                                                              -1).squeeze(
            0).data.cpu().numpy()

        src_del = numpy.delete(srcs_numpy[i], indexs).tolist()
        # print_sen(src_del, id2char)
        src_row.append(src_del)

    pad_type = 0
    srcs_numpy = numpy_fillna(numpy.array(src_row), pad_type)

    srcs = torch.Tensor(srcs_numpy).cuda().long()
    return srcs


def numpy_fillna(data, pad_type):
    # Get lengths of each row of data
    lens = numpy.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = numpy.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = numpy.full(mask.shape, pad_type)
    out[mask] = numpy.concatenate(data)
    return out


def replace_comma(srcs, symbol_drop, id2char):
    srcs_numpy = srcs.data.cpu().numpy()
    indexs = numpy.where(srcs_numpy == 5)
    # indexs_exclamation=
    # print("**********")
    # for item in srcs_numpy:
    #     print_sen(item, id2char)

    # indexs = ((srcs == 5.0 ) != 0).nonzero()
    # print(indexs)
    # exit()
    if len(srcs_numpy[0]) > 3 and len(indexs[0]) > 0:
        # for item in srcs_numpy:
        #     print_sen(item, id2char)
        # print("***********")
        for i in range(len(indexs[0])):
            flag = numpy.random.randint(0, 5)
            if flag <= 1:
                srcs_numpy[indexs[0][i]][indexs[1][i]] = 6
            elif flag == 2:
                srcs_numpy[indexs[0][i]][indexs[1][i]] = 23
                # else:
                #     srcs_numpy[indexs[0][i]][indexs[1][i]] = 98

                # for item2 in srcs_numpy:
                #     print_sen(item2, id2char)
                # exit()
    srcs = torch.Tensor(srcs_numpy).cuda().long()
    return srcs


def homophone_replace(srcs, id2char, char2id, char2py, homophone_table):
    srcs_numpy = srcs.data.cpu().numpy()

    srcs_new = []
    for i in range(len(srcs)):
        # indexs = ((torch.cuda.FloatTensor(srcs[i].shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
        char_type = None
        cur_src = srcs_numpy[i]
        # print("---------------------")
        # print_sen(cur_src, id2char)
        chose_count = 0
        while chose_count < 10:
            chose_count += 1
            # print("chose char ...")
            index, id = avoid_special_id(cur_src)
            char = id2char[int(id)]

            if isChinese(char):

                if index - 1 > 0:
                    id_pre = srcs_numpy[i][index - 1]
                    if id_pre not in [0, 1, 2, 3]:
                        char_pre = id2char[int(id_pre)]
                        if "@@" in char_pre:
                            char = char_pre[:-2] + char
                            char_type = 3
                            break

                if char in char2py.keys():
                    char_type = 1
                    break

            elif "@@" in char:
                id = srcs_numpy[i][index + 1]
                if id not in [0, 1, 2, 3]:
                    char_next = id2char[int(id)]
                    char = char[:-2] + char_next
                    char_type = 2
                    if char in char2py.keys():
                        break

        if char_type is not None:
            circle_index = 0

            chosen_list = []

            while True:
                circle_index += 1

                if char not in char2py.keys() or circle_index > 10:
                    # print("!!!!!!!!!!!")
                    # print(circle_index)
                    # print(char_type, end=" ")
                    # print(char)
                    break
                chosen_char = random_picks(homophone_table[char2py[char]], char, chosen_list)
                chosen_list.append(chosen_char)

                # print(char_type, end="# ")
                # print(char, end=" ")
                # print(chosen_char)

                if char_type == 1:
                    if chosen_char in char2id.keys() and chosen_char != char:
                        cur_src[index] = char2id[chosen_char]
                        break

                elif char_type == 2:
                    if chosen_char[0] + "@@" in char2id.keys() and chosen_char[1:] in char2id.keys():
                        cur_src[index] = char2id[chosen_char[0] + "@@"]
                        cur_src[index + 1] = char2id[chosen_char[1:]]

                        break
                else:

                    if chosen_char[0] + "@@" in char2id.keys() and chosen_char[1:] in char2id.keys():
                        cur_src[index - 1] = char2id[chosen_char[0] + "@@"]
                        cur_src[index] = char2id[chosen_char[1:]]
                        break

        # if char_type == 2:
        #     print(char, end=" ")
        #     print(chosen_char)
        #     exit()

        # print_sen(cur_src, id2char)

        srcs_new.append(cur_src.tolist())

    srcs = torch.Tensor(srcs_new).cuda().long()
    # exit()
    return srcs

    # def homophone_replace(srcs, symbol_drop, id2char, char2id, char2py, homophone_table):
    #     indexs = ((torch.cuda.FloatTensor(srcs.shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
    #
    #     srcs = srcs.data.cpu().numpy()
    #     if len(srcs[1]) > 5:
    #         # ran_1 = random.randint(0, len(src_words) - 1)
    #         for index in indexs:
    #             id = srcs[index[0]][index[1]]
    #             if id not in [0, 1, 2, 3]:
    #                 # pad = '<pad>', eos = '</s>', unk = '<unk>'
    #                 char = id2char[int(id)]
    #                 if "@@" in char:
    #
    #                 elif:
    #
    #                 # print(char)
    #                 if isChinese(char) and char in char2py:
    #                     chosen_char = char
    #                     while True:
    #                         if chosen_char == char:
    #                             chosen_char = random_picks_num(homophone_table[char2py[char]])
    #                         else:
    #                             break
    #
    #                     # print(char)
    #                     if chosen_char in char2id.keys():
    #                         id = char2id[chosen_char]
    #
    #             srcs[index[0]][index[1]] = id
    #     srcs = torch.Tensor(srcs).cuda().long()
    #
    #     return srcs


def repeat(srcs, src_lengths, symbol_drop):
    srcs_numpy = srcs.data.cpu().numpy()
    src_lengths_numpy = src_lengths.data.cpu().numpy()

    id = 0
    srcs_new = []
    for i in range(len(srcs)):
        select_time = 0

        # indexs = ((torch.cuda.FloatTensor(srcs[i].shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
        cur_src = srcs_numpy[i]

        while id in [0, 1, 2, 3]:

            select_time += 1

            index = numpy.random.randint(0, len(srcs[i]))
            id = srcs_numpy[i][index]
            if select_time >= 10:
                return srcs, src_lengths
        # self.print_sen(cur_src)
        id = 0

        offset = numpy.random.randint(0, 4)

        # noise_id = self.voise_chars_ids[numpy.random.randint(0, len(self.voise_chars_ids))]
        repeat_id = srcs_numpy[i][index]
        if index + offset >= len(srcs[i]):
            offset = 0
        # cur_src = numpy.insert(cur_src, index + offset, repeat_id, 0)

        if isinstance(repeat_id, list):
            cur_src = numpy.insert(cur_src, index + offset, repeat_id[0], 0)
            cur_src = numpy.insert(cur_src, index + offset + 1, repeat_id[1], 0)

        else:
            cur_src = numpy.insert(cur_src, index + offset, repeat_id, 0)
        # self.print_sen(cur_src)

        srcs_new.append(cur_src.tolist())
        src_lengths_numpy[i] = src_lengths_numpy[i] + 1

    srcs = torch.Tensor(srcs_new).cuda().long()

    src_lengths = torch.Tensor(src_lengths_numpy).cuda().long()
    return srcs, src_lengths


def insert(srcs, src_lengths, pos, insert_chars):
    srcs_numpy = srcs.data.cpu().numpy()
    src_lengths_numpy = src_lengths.data.cpu().numpy()
    id = 0
    srcs_new = []
    # print(srcs_numpy)

    for i in range(len(srcs)):
        # indexs = ((torch.cuda.FloatTensor(srcs[i].shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()

        select_time = 0

        # self.print_sen(cur_src)
        cur_src = srcs_numpy[i]
        noise_id = insert_chars[numpy.random.randint(0, len(insert_chars))]

        if pos is not None:
            while id in [0, 1, 2, 3]:
                select_time += 1
                pos += 1
                if pos < len(srcs_numpy[i]):
                    id = srcs_numpy[i][pos]
                else:
                    return srcs, src_lengths
                if select_time >= 10:
                    return srcs, src_lengths

            id = 0
            if isinstance(noise_id, list):  # BPE的情况
                cur_src = numpy.insert(cur_src, pos, noise_id[0], 0)
                cur_src = numpy.insert(cur_src, pos + 1, noise_id[1], 0)

            else:
                cur_src = numpy.insert(cur_src, pos, noise_id, 0)

            pos = 0  # 重置下一句的pos

        else:

            while id in [0, 1, 2, 3]:
                select_time += 1
                index = numpy.random.randint(0, len(srcs[i]))
                id = srcs_numpy[i][index]
                if select_time >= 10:
                    return srcs, src_lengths
            id = 0
            if isinstance(noise_id, list):

                cur_src = numpy.insert(cur_src, index, noise_id[0], 0)
                cur_src = numpy.insert(cur_src, index + 1, noise_id[1], 0)

            else:
                cur_src = numpy.insert(cur_src, index, noise_id, 0)

        # self.print_sen(cur_src)
        src_lengths_numpy[i] = src_lengths_numpy[i] + 1
        srcs_new.append(cur_src.tolist())

    srcs = torch.Tensor(srcs_new).cuda().long()
    src_lengths = torch.Tensor(src_lengths_numpy).cuda().long()
    return srcs, src_lengths


def replace(srcs, insert_chars=None,symbol_drop=None):
    srcs_numpy = srcs.data.cpu().numpy()
    indexs = ((torch.cuda.FloatTensor(srcs.shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
    if len(srcs_numpy[0]) > 3:
        for index in indexs:
            c_id = insert_chars[numpy.random.randint(0, len(insert_chars))]
            chosen_id = srcs_numpy[index[0]][index[1]]
            if chosen_id not in [0, 1, 2]:
                srcs_numpy[index[0]][index[1]] =c_id   # unk
    srcs = torch.Tensor(srcs_numpy).cuda().long()
    # srcs_numpy = srcs.data.cpu().numpy()
    # # for item in srcs_numpy.tolist():
    # #     print(item)
    # id = 0
    # srcs_new = []
    # index=0
    # time=0
    # for i in range(len(srcs)):
    #     # indexs = ((torch.cuda.FloatTensor(srcs[i].shape).uniform_() > 1.0 - symbol_drop) != 0).nonzero()
    #
    #     cur_src = srcs_numpy[i]
    #
    #     while id in [0, 1, 2, 3]:
    #         index = numpy.random.randint(0, len(srcs[i]))
    #         id = srcs_numpy[i][index]
    #         time+=1
    #         if time >5:
    #             return srcs
    #     id = 0
    #
    #     # print_sen(cur_src)
    #     noise_id = insert_chars[numpy.random.randint(0, len(insert_chars))]
    #     cur_src[index] = noise_id
    #
    #     # print_sen(cur_src)
    #
    #     srcs_new.append(cur_src.tolist())
    #
    # srcs = torch.Tensor(srcs_new).cuda().long()
    return srcs


def insert_char(srcs, src_lengths, dic_char2seg, insert_chars, src_dic):
    """插入语气词"""

    srcs_numpy = srcs.data.cpu().numpy()
    src_lengths_numpy = src_lengths.data.cpu().numpy()
    id = 0
    srcs_new = []
    for i in range(len(srcs)):
        sen = src_dic.string(srcs[i])
        select_time = 0
        index=0

        while id in [0, 1, 2]:
            select_time += 1
            if select_time >= 5:
                return srcs, src_lengths
            if sen in dic_char2seg.keys():
                words_list = dic_char2seg[sen].split(" ")
                if len(words_list) < 4:
                    return srcs, src_lengths
                noise_index = numpy.random.randint(0, len(words_list)-2)
                new_be_sen_char = words_list[noise_index + 1][0]
                if not isChinese(new_be_sen_char):
                    continue
                id = src_dic.index(new_be_sen_char)
                index = numpy.where(srcs_numpy[i] == id)[0][0]

            else:
                index = numpy.random.randint(0, len(srcs[i])-1)
                id = srcs_numpy[i][index]
        id = 0
        cur_src = srcs_numpy[i]
        noise_char = random_picks_num(insert_chars)
        noise_id=src_dic.index(noise_char)
        cur_src = numpy.insert(cur_src, index, noise_id, 0)

        # self.print_sen(cur_src)
        src_lengths_numpy[i] = src_lengths_numpy[i] + 1
        srcs_new.append(cur_src.tolist())

    srcs = torch.Tensor(srcs_new).cuda().long()
    src_lengths = torch.Tensor(src_lengths_numpy).cuda().long()
    return srcs, src_lengths
def get_list_id(char_list,src_dic):
    id_list=[]
    for i in char_list:
        id_list.append(src_dic.index(i))
    return id_list


def repeat_char(srcs, src_lengths, dic_char2seg, src_dic):
    """重复"""

    srcs_numpy = srcs.data.cpu().numpy()
    src_lengths_numpy = src_lengths.data.cpu().numpy()
    srcs_new = []
    for i in range(len(srcs)):
        sen = src_dic.string(srcs[i])
        # print(sen)
        select_time = 0
        index=0
        pad = False
        id =0
        while id in [0, 1, 2]:
            select_time += 1
            if select_time >= 5:
                break
            if sen in dic_char2seg.keys():
                words_list = dic_char2seg[sen].split(" ")
                if len(words_list) < 4:
                    return srcs, src_lengths
                noise_index = numpy.random.randint(0, len(words_list)-2)#选中分词中的词
                insert_index = numpy.random.randint(noise_index, min(len(words_list),noise_index + 4))#得到插入的位置
                chosen_word = words_list[noise_index]
                insert_pos=words_list[insert_index]
                if not isChinese(chosen_word):
                    continue
                # print(new_be_sen_char)
                if len(chosen_word) > 2:
                    continue
                elif len(chosen_word)==2:
                    id=(src_dic.index(chosen_word[0]),src_dic.index(chosen_word[1]))
                    # index = numpy.where(srcs_numpy[i] == id[0])[0][0]

                else:
                    id =src_dic.index(chosen_word)
                    pad=True
                    # index = numpy.where(srcs_numpy[i] == id)[0][0]
                # print(sen)
                # print(srcs_numpy[i])
                # print(insert_pos[0])
                # print(src_dic.index(insert_pos[0]))
                # print(insert_pos)
                # print(src_dic.index(insert_pos))
                #非中文
                if src_dic.index(insert_pos[0]) in srcs_numpy[i]:
                    # print("&&&&&&&&&")
                    insert_pos=insert_pos[0]
                # print(src_dic.index(insert_pos))
                if src_dic.index(insert_pos) in srcs_numpy[i]:
                    index = numpy.where(srcs_numpy[i] == src_dic.index(insert_pos))[0][0]
                    break
                else:
                    continue

            else:
                pad = True
                index = numpy.random.randint(0, len(srcs[i])-1)
                id = (srcs_numpy[i][index])
                index = numpy.random.randint(index, min(len(srcs[i])-1,index + 4))#得到插入的位置

        cur_src = srcs_numpy[i]
        if select_time>=5:
            cur_src = numpy.insert(cur_src, len(cur_src), 0, 0)
            cur_src = numpy.insert(cur_src, len(cur_src), 0, 0)
        else:
            if pad:
                cur_src = numpy.insert(cur_src, index, id, 0)
                cur_src = numpy.insert(cur_src, len(cur_src), 0, 0)
            else:
                cur_src = numpy.insert(cur_src, index, id[0], 0)
                cur_src = numpy.insert(cur_src, index+1, id[1], 0)
        # print(cur_src)
        # exit()


        # self.print_sen(cur_src)
        src_lengths_numpy[i] = src_lengths_numpy[i] + 2
        srcs_new.append(cur_src.tolist())

    srcs = torch.Tensor(srcs_new).cuda().long()
    src_lengths = torch.Tensor(src_lengths_numpy).cuda().long()
    return srcs, src_lengths

def shuffle_window(srcs, windows_size=None, window_num=None, id2char=None):
    """局部按照窗口大小来打乱句子，支持多次，可变窗口大小"""
    srcs_numpy = srcs.data.cpu().numpy()
    # print(srcs)
    # print_sen(srcs_numpy,id2char)
    # srcs_numpy = srcs
    if len(srcs_numpy[0]) < windows_size + window_num + 2:
        return srcs

    for i in range(len(srcs)):
        begin_id = 0
        index = 0
        select_time = 0

        for ii in range(window_num):
            while begin_id in [0, 1]:
                index = numpy.random.randint(0, len(srcs_numpy[i]) - windows_size)
                # print("index")
                # print(index)
                begin_id = srcs_numpy[i][index]
                select_time += 1
                if select_time >= 10:
                    return srcs
                # print("begin_id")
                # print(begin_id)
            slide_list = srcs_numpy[i][index:index + windows_size]
            # print(slide_list)
            numpy.random.shuffle(slide_list)
            # print(slide_list)

            for iii in range(windows_size):
                srcs_numpy[i][index + iii] = slide_list[iii]

        # shuffle_list.append(src_list[index +ii] for ii in range(windows_size))

        # for check_id in range(begin_id,begin_id+windows_size):
        #     if check_id in [0,1]:
        #         begin_id=0
    # print_sen(srcs_numpy,id2char)

    srcs = torch.Tensor(srcs_numpy).cuda().long()
    return srcs


def flip(srcs, dic_char2seg, src_dic, punc_ids):
    """倒装，句首的词移到句尾"""
    # print(numpy.where(srcs[0] == 5)[0][0])
    srcs_numpy = srcs.data.cpu().numpy()
    for i in range(len(srcs)):
        period_index = -2
        # print(srcs_numpy[i])
        sen = src_dic.string(srcs[i])
        # print(sen)
        if sen not in dic_char2seg.keys():
            # print("*********")
            continue
        words_list = dic_char2seg[sen].split(" ")
        if len(words_list) < 4:
            return srcs
        # print(dic_char2seg[sen])
        index = numpy.random.randint(0, min(len(words_list), 3))
        new_be_sen_char = words_list[index + 1][0]
        # print(new_be_sen_char)
        if not isChinese(new_be_sen_char):
            continue
        char_id = src_dic.index(new_be_sen_char)
        # print(srcs_numpy[i])
        for ii in punc_ids:
            period_index = numpy.where(srcs_numpy[i] == ii)[0]
            # print("*****************")
            # print(period_index)
            if len(period_index):
                period_index = period_index[0]
                break
            else:
                period_index = -1

        # period_id = src_dic.index("。")
        # period_index = numpy.where(srcs_numpy[i] == period_id)[0][0]
        char_index = numpy.where(srcs_numpy[i] == char_id)[0][0]

        if period_index != -1:
            insert_index = period_index
        # print(char_id)
        else:
            # eos的位置
            insert_index = numpy.where(srcs_numpy[i] == 1)[0][0]
        if char_index > insert_index:
            # 选中的位置不能在句号后
            continue
        # print(char_index)
        # print("*****************")
        # print(srcs_numpy[i])
        # print(srcs_numpy[i][char_index:insert_index])
        # print(srcs_numpy[i][0:char_index])
        # print(srcs_numpy[i][insert_index:])
        # exit()

        new_sen = numpy.append(srcs_numpy[i][char_index:insert_index], srcs_numpy[i][0:char_index])
        new_sen = numpy.append(new_sen, srcs_numpy[i][insert_index:])
        srcs_numpy[i] = new_sen
        # print(srcs_numpy[i])
        # exit()
    srcs = torch.Tensor(srcs_numpy).cuda().long()
    return srcs


def get_char2seg_dic():
    count = 0
    dic_char2seg = {}
    # with open("/search/odin/haiyang/fairseq_exp/robust/demo/data/zh2en/train.zh") as char:
    #     with open("/search/odin/haiyang/fairseq_exp/robust/demo/data/zh2en/train.seg.zh") as seg:
    with open("/search/odin/haiyang/fairseq_exp/baseline/demo/data/data06/train.zh") as char:
        with open("/search/odin/haiyang/fairseq_exp/baseline/demo/data/data06/train.seg.zh") as seg:
            for line1, line2 in zip(char, seg):
                # if len(line1.split(" ")) < 16:
                    # print(line1)
                    count += 1
                    if count % 1000000 == 0:
                        print(count)
                    dic_char2seg[line1.strip()] = line2.strip()
            print(count)
    # exit()

    return dic_char2seg


def get_id_list(src_dic, char_list):
    id_list = []
    for char in char_list:
        id_list.append(src_dic.index(char))

    return id_list


if __name__ == '__main__':
    a = [[0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # if
    print(numpy.where(numpy.array(a)[0] == 11)[0])
    # shuffle_window(numpy.array(a), 3, 1)
    # flip(numpy.array(a), None, None)
