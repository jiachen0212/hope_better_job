# coding=utf-8
# https://www.cnblogs.com/chengxiao/p/6194356.html
# 时复O(nlogn)  空复O(n)
# 分段有序再merge，递归思路
def merge(a, b):
    res = []
    p1 = p2 = 0
    while p1 < len(a) and p2 < len(b):
        if a[p1] < b[p2]:
            res.append(a[p1])
            p1 += 1
        else:
            res.append(b[p2])
            p2 += 1

    if p1 == len(a):
        for bb in b[p2:]:
            res.append(bb)
    else:
        for aa in a[p1:]:
            res.append(aa)
    return res

def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    middle = len(lists)/2
    left = merge_sort(lists[:middle])
    right = merge_sort(lists[middle:])
    return merge(left, right)


if __name__ == '__main__':
    a = [4, 7, 8, 3, 5, 9]
    print merge_sort(a)



##########################################
# 手摇算法实现 O(1)空复  时复nlog(2n)归并
# 使用3次reverse操作实现相邻区间的交换位置
def handreverse(a,i,index,j):
    c =[]
    c1 = []
    c2 = []
    c3 = []
    # 反转  index是转折点
    if i == 0:
        for each in a[index-1::-1]:
            c1.append(each)
    else:
        for each in a[index-1:i-1:-1]:
            c1.append(each)
    for each in a[j-1:index-1:-1]:
        c2.append(each)
    c3 = c1 + c2
    c3.reverse()
    c = a[0:i] + c3 + a[j:]
    a[:] = c[:]

def merge(a,lo,hi):
    mid = (lo + hi) / 2
    i = lo
    j = mid +1
    while i<j and j<=hi:
        while i<j and a[i]<=a[j]:
            i += 1
        index = j
        while j<=hi and a[j]<a[i]:
            j += 1
        handreverse(a,i,index,j)
        i = i + (j - index)
    return a
def sort(a,lo,hi):
    if lo < hi:
        mid = (lo + hi) /2
        sort(a,lo,mid)
        sort(a,mid+1,hi)
        merge(a,lo,hi)
    return a

a = [3, 7, 8, 4, 5, 9]
sort(a,0,5)
print a,'==='



