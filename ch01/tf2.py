#!/usr/bin/python
# coding=utf-8
'''
@Author: Ulysses
@Date: 2020-02-24 16:51:56
@LastEditors: Ulysses
@LastEditTime: 2020-03-02 16:24:01
'''
#%%
import tensorflow as tf
assert tf.__version__.startswith('2.')

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算并打印
print('a+b=',a+b)




# %%
