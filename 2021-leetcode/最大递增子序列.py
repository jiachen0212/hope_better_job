#coding=utf-8
import numpy as np

a = [10, 80, 6, 3, 4, 7, 1, 5, 11, 2, 12, 30, 13]
concoll = [0]
for i in range(len(a) - 1):
    if a[i] < a[i+1]:
        flage = True
        count = concoll[-1] + 1
        concoll.append(count)
        print concoll
        flage = False
    else:
        concoll.append(1)
maxcou = max(concoll)
print maxcou