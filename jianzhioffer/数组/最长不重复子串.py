# coding=utf-8
# 最长的,没有重复字符的,子字符串序列长度
# 用字典,key是string中的字符,val是字符出现的下标值


# O(n)的方法
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        res = 0
        if not s:
            return res
        d = {}    # key 是s中的字符 value是这个字符出现的index  index会后序被刷新
        start = 0
        for i in range(len(s)):
            if s[i] in d and d[s[i]] >= start:
                start = d[s[i]] + 1  # 新的不重复子串的头要在d[s[i]]的基础上后移一位
            tmp = i + 1 - start
            d[s[i]] = i
            res = tmp if tmp > res else res
        return res

s = Solution()
res = s.lengthOfLongestSubstring('arabcacfr')
print res, '===='




