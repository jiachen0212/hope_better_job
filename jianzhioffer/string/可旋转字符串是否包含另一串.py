# coding=utf-8
# 判断一可旋转的字符串是否包含另一串

'''
思路：
旋转字符串这种题，就是把原来的string扩大一倍，如果在这个扩的字符串里有那就有。
'''

import sys
try:
    while True:

        line1 = sys.stdin.readline().strip()
        line2 = sys.stdin.readline().strip()

        line3 = sys.stdin.readline().strip()
        line4 = sys.stdin.readline().strip()

        line5 = sys.stdin.readline().strip()
        line6 = sys.stdin.readline().strip()

        line1 += line1
        line3 += line2
        line5 += line5
        res = []
        if line2 in line1:
            res.append(1)
        else:
            res.append(0)
        if line4 in line3:
            res.append(1)
        else:
            res.append(0)
        if line6 in line5:
            res.append(1)
            print(res)
            break
        else:
            res.append(0)
            print(res)
            break

except:
    pass