# coding=utf-8
# 分割数组  左小右大   在完成这样的分组后返回 left 的长度
# leetcode 915
'''
输入：[5,0,3,8,6]
输出：3
解释：left = [5,0,3]，right = [8,6]
示例 2：

输入：[1,1,1,0,6,12]
输出：4
解释：left = [1,1,1,0]，right = [6,12]

维护两个变量 当前最大值和左数组最大值
'''
# 思路好巧妙！！！
class Solution(object):
    def partitionDisjoint(self, A):
        if not A:
            return 0
        leftmax = A[0]   # 左数组的最大值
        mmax = A[0]
        index = 0
        l = len(A)
        for i in range(l):
            mmax = max(mmax, A[i])  # 数组的当前最大值
            if A[i] < leftmax:
                leftmax = mmax
                index = i
        return index+1

s = Solution()
res = s.partitionDisjoint([5,0,3,8,6])
print res