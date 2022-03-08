#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : grammar_test.py
    @Time       : 2022/3/8 14:52
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：
"""
import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 1, 1], [1, 1, 1]])
    b = np.array([[1, 0, 1], [1, 0, 0]])
    c = (b == 0)
    a[c] = 0
    pass


