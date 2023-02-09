# coding=utf-8
# 回文数判断
'''
Example 1:
Input: 121
Output: true

Example 2:
Input: -121
Output: false
'''

# int 转成 str 先, 会快捷很多～
class Solution(object):
    def isPalindrome(self, x):
        res = True
        tempstr = str(x)
        if tempstr[0] == '-':
            res = False
        else:
            llen = len(tempstr)
            if llen > 1:
                for i in range(llen / 2):
                    if tempstr[i] != tempstr[llen - 1 - i]:
                        res = False
        return res

s = Solution()
res = s.isPalindrome('121.121')
print(res)