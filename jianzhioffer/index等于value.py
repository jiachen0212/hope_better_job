#coding=utf-8
# 一个递增的list,找到val刚好等于下标的那个数
# 思路: 二分实现. 先mid==s[mid]的话,直接就找到了. 如果s[mid]>mid,根据list递增的特点,mid之后的每个元素都会满足s[i]>i,所以s[i]=i的值必在mid之前...一直二分下去即可..

def val_equal_index(s):
    if len(s) <= 0:
        return -1
    start = 0
    end = len(s) - 1
    while start <= end:
        mid = (start + end) >> 1
        if s[mid] == mid:
            return mid
        elif s[mid] > mid:
            print 'the val == index is in the left'
            end = mid - 1
        else:  # val=index在右边
            start = mid + 1

res = val_equal_index([-3, -1, 2, 4, 5, 6])
print res