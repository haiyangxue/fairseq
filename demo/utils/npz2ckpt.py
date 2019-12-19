#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# xuehaiyang: xuehaiyang@sogou-inc.com
#

"""
Convert npz model to tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) != 4:
    sys.stderr.write('usage: %s + original_t2t_model_path + npz_model_path + output_t2t_model_path ' % __file__)
    sys.exit(-1)

original_t2t_model = sys.argv[1]
npz_model = sys.argv[2]
output_t2t_model = sys.argv[3]

if output_t2t_model[-1] != "/":
    output_t2t_model = output_t2t_model + "/"

npz2ckpt_dic = {"symbol_modality/target_emb/weights_0": "symbol_modality_9882_512/target_emb/weights_"
    , "symbol_modality/softmax/weights_0_V": "symbol_modality_9882_1024/softmax/weights_"
    , "symbol_modality/softmax/weights_0_U": "symbol_modality_9882_1024/softmax/upper_weights_"
    , "symbol_modality/input_emb/weights_0": "symbol_modality_48899_512/input_emb/weights_"
                }

npz_model = np.load(npz_model)

modality_split = {}
# the embeddings are divided into blocks
for item in npz_model.files:
    if item in npz2ckpt_dic.keys():

        avg = int(npz_model[item].shape[0] / 16)
        indexs = np.zeros(16)
        yu = npz_model[item].shape[0] % 16
        start = 0
        end = 0
        for i in range(16):
            if i > 0:
                indexs[i] = indexs[i - 1] + avg
                start = int(indexs[i - 1])
            else:
                indexs[i] = avg
            if i < yu:
                indexs[i] += 1
            end = int(indexs[i])
            t2t_modality_name = npz2ckpt_dic[item] + str(i)

            modality_split[t2t_modality_name] = npz_model[item][start:end]


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
            tf.gfile.Exists(path + ".index"))


def main(_):
    # Get the checkpoint from flags and run some basic checks.

    if not original_t2t_model:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % original_t2t_model)

    # Read variables from  checkpoint
    print("Reading variables checkpoints...")

    var_list = tf.contrib.framework.list_variables(original_t2t_model)
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)

    reader = tf.contrib.framework.load_checkpoint(original_t2t_model)

    for name in var_values:
        tensor = reader.get_tensor(name)
        if name in modality_split.keys():
            tensor = modality_split[name]
        elif name in npz_model.files or name[0:6] == "symbol":

            name_new = name
            if name[0:6] == "symbol":
                name_items = name.split("/")
                name_items[0] = "symbol_modality"
                name_new = "/".join(name_items)

            if len(npz_model[name_new].shape) == 2 and name != "body/target_space_embedding/kernel":
                tensor = npz_model[name_new][np.newaxis, np.newaxis, :, :]
            else:
                tensor = npz_model[name_new]

        var_dtypes[name] = tensor.dtype
        var_values[name] = tensor
    print("Read from checkpoint %s", original_t2t_model)

    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(
        0, name="global_step", trainable=False, dtype=tf.int64)
    saver = tf.train.Saver(tf.all_variables())

    # Build a model consisting only of variables, set them to values.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               six.iteritems(var_values)):
            sess.run(assign_op, {p: value})
        # Use the built saver to save the checkpoint.
        saver.save(sess, output_t2t_model, global_step=global_step)

    print("load npz model saved in %s", output_t2t_model)


if __name__ == "__main__":
    tf.app.run()
