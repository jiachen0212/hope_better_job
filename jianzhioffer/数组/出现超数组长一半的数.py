#coding=utf-8

# 京东面试被问到，要求时复O(n),空复O(1)找出这个出现超一半的数
def find(nums):
    l = len(nums)
    res = 0
    count = 0
    for i in range(l):
        if count == 0:
            res = nums[i]
            count = 1
        else:
            if res == nums[i]:
                count += 1
            else:
                count -= 1
    return res


nums = [1,2,3,2,5,2,2,6,2,2,2]
print(find(nums), '====')



# 寻找数组中出现次数超过数组长度一半的元素值
# 方法一：使用两个数组辅助实现：数组一存储出现的不同元素,数组二存储对应的元素出现次数
# 方法二：快排实现：将数组进行大小排序,出现在数组中间位置的元素有可能是出现次数大于数组长度一半的元素
# 快排算法原理： 随意选一个数，把小于它的数放左边，大于的放右边。 然后左右边的数也一样这样操作，直到数组长为1就排完了。

############# 牛客 ac 版
# -*- coding:utf-8 -*-
class Solution:
    def quick_sort(self, s, left, right):
        if left >= right:
            return s
        key = s[left]
        low = left
        high = right
        while left < right:
            while left < right and s[right] >= key:
                right -= 1
            s[left] = s[right]
            while left < right and s[left] <= key:
                left += 1
            s[right] = s[left]
        s[left] = key
        self.quick_sort(s, low, left - 1)
        self.quick_sort(s, left + 1, high)
        return s

    def countnum(self, s, mid):
        count = 0
        for i, num in enumerate(s):
            if num == mid:
                count += 1
        return count

    def MoreThanHalfNum_Solution(self, numbers):
        l = len(numbers)
        mid = l >> 1
        s = self.quick_sort(numbers, 0, l - 1)
        res = s[mid]
        count = self.countnum(numbers, res)
        print(count, mid)
        if count <= mid:
            return 0
        return res




# 方法一：使用两个数组辅助实现,数组一存储出现的不同元素,数组二存储对应的元素出现次数
def fun1(s):
    l = len(s)
    val = []   # 存放不同的数字
    time = []  # 存放出现的次数
    for i in range(l):
        if s[i] not in val:  # 判断一个元素是否在list中
            val.append(s[i])
            time.append(1)   # time里添加一个计数位置
        else:
            ind = val.index(s[i])  # 查看这个重复出现的元素的索引
            time[ind] += 1  # 在对应次数位置上+1
    maxtime = max(time)
    if maxtime < l >> 1:  # 最大出现次数小于数字长的一半
        return False
    res = val[time.index(maxtime)]
    return res



# 方法二：用快排实现数组大小排序,出现在数组中间位置的元素必定是大于数组长度一半的元素
# 首先进行快排
def quick_sort(s, left, right):
    if left >= right:   # 数组仅含一个元素
        return s
    key = s[left]  # 设定基准数,可以是第一个也可以是其他位置的数
    low = left
    high = right  # 存储首末指针用于后序的递归
    while left < right:
        # 注意这里要先检查s[right]处的值.因为一半设置首元素是基准数,所以先检查s[left]的话指针一定会后移,可能导致部分元素值重覆盖.使得s变了..
        while left < right and s[right] >= key:
            right -= 1
        s[left] = s[right]  # 扫到的是个小于基准数的值,把它换到前面去
        while left < right and s[left] <= key:
            left += 1
        s[right] = s[left]  # 扫到的是个大于基准数的值,把它换到后面去
    s[left] = key  # 把key基准值放进来.
    # 以下再递归的分别处理基准数左右两边的序列
    quick_sort(s, low, left - 1)
    quick_sort(s, left + 1, high)
    return s

def countnum(s, mid):
    count = 0
    for i, num in enumerate(s):
        if num == mid:
            count += 1
    return count

def fun2(s):
    l = len(s)
    mid = l >> 1  # 位运算,实现除以2
    s = quick_sort(s, 0, l - 1)  # 最初的左右指针分别指向首和末
    # print s
    res = s[mid]  # 取排序后的中位数即为出现次数大于一半list长度的元素
    count = countnum(s, res)
    print(mid, count)
    if count <= mid:  # 要求的是超过，所以<=都认为不选在想要的结果
        return 0
    return res

# s = [2, 1, 7, 5, 7, 4, 7, 7, 6, 6]
s = [1,2,3,2,4,2,5,2,3]
res1 = fun1(s)
print '方法一:', res1
res2 = fun2(s)
print '方法二:', res2


