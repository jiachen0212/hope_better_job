# coding=utf-8
# 行列有序矩阵，求第k小的数
'''
1,5,9
10,11,13
12,13,15
'''
# https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/comments/

# 每次的比较元素是左下角  matrix[i][j]<=mid的话  即它上面的i个都是小于mid的
# 右边的都是比左边大的
class Solution(object):
    def kthSmallest(self, matrix, k):
        m, n = len(matrix), len(matrix[0])
        lo,hi = matrix[0][0], matrix[-1][-1]

        while lo<=hi:
            mid = (lo+hi)/2
            i,j=m-1, 0   # 左下角开始找  注意！！
            count = 0
            while i>=0 and j<n:
                if matrix[i][j]<=mid:
                    count += (i+1)  # 这一列上面所有的数都是比mid小
                    j += 1  # 右移
                else:
                    i -= 1
            if count < k:   # mid选小了
                lo = mid+1
            else:
                hi = mid-1
        return lo

