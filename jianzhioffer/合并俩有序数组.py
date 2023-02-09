# -*- coding:utf-8 -*-
# 合并俩有序数组, 结果仍有序
'''
输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出: [1,2,2,3,5,6]

'''


# 用三个指针的，从后往前
# 从后往前避免了像从前往后那需要的数组元素整体后移的问题
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        p1,p2,p = m-1,n-1,m+n-1
        while p1>=0 and p2>=0:
            if nums1[p1]>nums2[p2]:
                nums1[p]=nums1[p1]
                p1 -= 1
                p -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
                p -=1
        if p2>=0:  # p1已经遍历完
            nums1[:p2+1] = nums2[:p2+1]
        # 如果是p2遍历完了p1没有遍历完，不用处理，它本来自己就是nums1
        # 该怎么就怎么
        return nums1

s = Solution()
num = s.merge([1,2,3,0,0,0],3,[2,5,6],3)
print num
