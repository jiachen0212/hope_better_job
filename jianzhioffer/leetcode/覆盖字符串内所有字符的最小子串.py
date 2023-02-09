# coding=utf-8
# leetcode
'''
输入: S = "ADOBECODEBANC", T = "ABC"
输出: "BANC"
'''
import collections

class Solution(object):
    def minWindow(self, s, t):
        # collections.Counter(t) 直接把t里面的字符个数统计好存成字典格式
        need, missing = collections.Counter(t), len(t)

        i = I = J = 0
        for j, c in enumerate(s, 1):
            missing -= need[c] > 0  # 表示要找的字符还有几个没找到
            need[c] -= 1

            if not missing:  # 所有字母都找到了
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1   # 滑动窗左边  后移
                if not J or j - i <= J - I:
                    I, J = i, j
        return s[I:J]

s = Solution()
res = s.minWindow('ADOBECODEBAN', 'ABCD')
print(res)



# 用滑动窗做   先右滑把所有的t中的值都找到  再左边界缩进
# 时间复杂度 O(mn)
# ac
class Solution(object):
    def minWindow(self, s, t):
        ls = len(s)
        lt = len(t)
        if not s or not t or ls < lt:
            return ''
        min_size = ls+1
        l,r=0,0   # 滑窗的左右边界
        start,end = 0, ls-1   # 扫描s
        Map = {}
        for ch in t:
            Map[ch] = Map.get(ch, 0) + 1
        matched  =0
        while r < ls:
            Map[s[r]] = Map.get(s[r], 0) - 1
            # 如果当前遇到的字符在map中出现过，则匹配数+1
            # 没出现的就是复数  出现过的话 >=0
            matched = matched + 1 if Map[s[r]] >= 0 else matched

            if matched == lt:  # 要找的数都找到了 现在要来缩进窗口的左边界
                while Map[s[l]] < 0:
                    Map[s[l]] += 1
                    l += 1
                if r-l+1 < min_size:
                    min_size = r-l+1
                    start, end = l, r
            r += 1
        return '' if min_size > ls else s[start: end+1]


