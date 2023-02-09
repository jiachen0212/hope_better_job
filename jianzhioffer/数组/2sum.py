# coding=utf-8
# 时间复杂度是O(n)
# 一个递增数组,找出两个数,它们的和为s. 有多对的话找出一对即可
# 方法: 使用两个指针,开始时候指向首末.若两个被指向的,数的和大于s,则把p2前移(因为位置在前面的数小于在后面的数)
# 若小于s则把p1后移.   即每次只在一个指针上进行调整,必定可以找到一对和为s的值


# 牛客 ac 版
# 注意这个是递增数组！！！
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        if not array:
            return []
        if len(array) < 2:
            return []
        p1 = 0
        p2 = len(array) - 1
        while p2 > p1:
            if array[p1] + array[p2] == tsum:
                return [array[p1], array[p2]]   # 可以直接退出，因为要的是乘积最小的那组。
                # return [p1,p2]   # 返回index
                # 两个数之和一定，这两个数的diff越小，乘积越大...
            elif array[p1] + array[p2] < tsum:
                p1 += 1
            else:
                p2 -= 1
        return []

S = Solution()
s = [1, 5, 5, 10, 12, 16]
res = S.FindNumbersWithSum(s, 10)
print res, '===='




##### 不用指针，且不要求是递增的数组
##### 另一个牛客大佬的ac版本
# ([2,2,4],4)  res:[2,2]
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        for i in array:
            if tsum-i in array:
                if tsum-i==i:
                    if array.count(i)>1:
                        return [i,i]
                else:
                    return [i,tsum-i]
        return []



# 这个是返回两值的index  leetcode ac
def twoSum(array, tsum):
    for i in array:
        if tsum -i in array:
            if tsum - i == i:
                if array.count(i) > 1:
                    id1 = array.index(i)
                    array.pop(id1)
                    id2 = array.index(i) + 1
                    return [id1, id2]
            else:
                id1 = array.index(i)
                id2 = array.index(tsum - i)
                return [id1, id2]
    return []

s = Solution()
res = s.FindNumbersWithSum([2,2,4],4)
print(res)