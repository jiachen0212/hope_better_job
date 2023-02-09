# coding=utf-8
# 有序数组，平方和后不重复数值的个数
# 用 set 一行搞定

a = [-10, -10, -5, 0, 1, 5, 8, 10]
print(len(set([x**2 for x in a])))