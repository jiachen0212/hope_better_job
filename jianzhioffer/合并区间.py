# coding=utf-8
# 合并区间
# https://www.cnblogs.com/grandyang/p/4370601.html

# 将起始位置和结束位置分别存到了两个不同的数组 begs 和 ends 中 且sort
# 遍历的时候，begs提前一位
#当begs 数组到了最后一个位置，或者begs 数组的 i+1 位置上的数字大于ends i
# 说明区间不连续了，加入res

'''
Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

leetcode ac
执行用时 :
84 ms, 在所有 Python 提交中击败了 94.72%的用户
内存消耗 :14.1 MB, 在所有 Python 提交中击败了100.00% 的用户
'''

class Solution(object):
    def merge(self, intervals):
        if len(intervals) == 1:  # in:[[1,4]]
            return intervals     # out: [[1,4]]

        # inputs = [[1,3],[2,6],[8,10],[15,18]]
        begs = []
        ends = []

        for bi in intervals:
            begs.append(bi[0])
            ends.append(bi[1])
        begs.sort()
        ends.sort()  # 排序

        l = len(begs)
        tmp1 = 0
        res = []
        for i in range(l-1):
            if begs[i+1] > ends[i]:  # beg要提前一位
                res.append([begs[tmp1], ends[i]])
                # print([begs[tmp1], ends[i]], '===', i)
                tmp1 = i+1  # tmp1存放区间起始点

            if i == l-2:
                res.append([begs[tmp1], ends[-1]])

        return res


s = Solution()
print(s.merge([[1,3],[3,7]]))
