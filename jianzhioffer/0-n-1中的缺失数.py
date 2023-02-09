#coding=utf-8
# n-1长度的递增list,范围在0~n内取值.所以存在某一缺失值m,找出这个m
# 使用二分查找
# 方法: 二分,就是寻找到第一个下标与元素本身值不等的那个元素,下标值即是缺失值. 因为0元素下标0,1下标1...当缺失的是m值,那么list中的m位置的元素值变成了m+1..


def fun(s):
    if len(s) <= 0:
        return -1
    start = 0
    end = len(s) - 1
    while start <= end:
        mid = (start + end) >> 1  # 位移 除以2
        if s[mid] != mid:   # 说明缺失值还在mid的后面
            if mid == 0 or s[mid - 1] == mid - 1:  # 说明mid刚好就是第一个下标与本元素值不等的元素,则返回下标
                # 增加mid==0是因为0-1=-1超过边界...
                return mid
            end = mid - 1
            print 'start:', start, 'end:', end, 'mid:', (start + end) >> 1
        else:  # 表明缺失值在边
            start = mid + 1
            print 'start:', start, 'end:', end, 'mid:', (start + end) >> 1
    return start

res = fun([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print 'the lost val is:', res