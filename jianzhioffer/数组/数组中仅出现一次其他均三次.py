
#coding=utf-8
'''
x ^ 0 = x;
x ^ x = 0;
'''

class Solution(object):
    def singleNumber(self, s):
        a,b=0,0
        for i in range(len(s)):
            a ^= s[i]
            b ^= (a&s[i])
            a ^= (b&s[i])
        return b
