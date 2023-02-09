# coding=utf-8
# 字符串解码
'''
str = '1111'
能转换成的结果有“AAAA”，“LAA”，“ALA”，“AAL”和“LL”，返回5
规定“1”转换为“A”，“2”转换为“B”……“26”转换为“Z”
'''

def num(str1):
    if not str1:
        return 0
    cur = 1 if str1[-1] != '0' else 0
    nex = 1
    for i in range(len(str1)-2, -1, -1):
        if str1[i] == '0':
            nex = cur
            cur = 0
        else:
            tmp = cur
            if int(str1[i]) * 10 + int(str1[i+1]) < 27: # str1[i]+str1[i+1]可以组成字母
                cur += nex
            nex = tmp
    return cur

print num('1111')