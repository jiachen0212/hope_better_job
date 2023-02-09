#coding=utf-8
import random

def get_randomNumber(num):
    lists = []
    i = 0
    while i < num:
        lists.append(random.randint(0, 100))
        i += 1
    return lists

####################  fun1 ##################
# 二分思想做  看大于mid的个数是否k 然后动态调整l r

# 找出最大最小值
def max_min(s):
    mmax, mmin = 0,0
    for i in s:
        mmin = mmin if mmin < i else i
        mmax = mmax if mmax > i else i
    return mmin, mmax

def half_find(s, k):
    if len(s) <= k:
        return s
    Min, Max = max_min(s)
    while Max >= Min:
        mid = (Max + Min) / 2
        kk = [x for x in s if x >= mid]  # 获得s中大于等于mid值的数
        if len(kk) == k:
            return kk
        elif len(kk) > k:  # 表示mid选小了,让min变大以使得mid变大
            Min = mid
        else:
            Max = mid
    return kk

s = get_randomNumber(10)
print s
res = half_find(s, 5)
print res, '==='


####################  fun2 ##################
# 维护一个k大的最小堆

# 最小堆维护
# 最小堆的heap[0]是最大值，最小值在最末[-1]
def heap_adjust(data, root):
    if 2*root+1 < len(data):
        # 下面四行 找到更小的子
        if 2*root+2 < len(data) and data[2*root+2] < data[2*root+1]:
            k = 2*root+2
        else:
            k = 2*root+1
        # 把堆顶换成最小
        if data[k] < data[root]:
            data[k],data[root] = data[root],data[k]
            heap_adjust(data,k)  # 递归维护下一个层子

# 创建堆
def min_heap(data):
    ind = len(data)/2 - 1
    for i in range(ind, -1, -1):
        heap_adjust(data, i)
    return data

def main_topk(nums, k):
    nums_k = min_heap(nums[:k])  # 先用前k个数组成一个堆
    for i in range(k, len(nums)):
        if nums[i] > nums_k[0]: # nums_k[0]最小
            nums_k[0] = nums[i]
            nums_k = min_heap(nums_k)  # 更新维护这个堆
    return nums_k

res = main_topk(s, 5)
print res