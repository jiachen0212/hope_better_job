# coding=utf-8
# leetcode easy ac

class Solution(object):
    def generate(self, numRows):
        if numRows < 1:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1,1]]
        res = [[1], [1,1]]
        for n in range(3, numRows+1):
            tmp = [1]
            for i in range(n-1-1):
                cache = res[-1][i] + res[-1][i+1]
                tmp.append(cache)
            tmp.append(1)
            res.append(tmp)
        return res

s = Solution()
print s.generate(5)
