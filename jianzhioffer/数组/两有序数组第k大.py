# coding=utf-8
# 两有序数组的第k大

# 一种思路是用O(m+n)的空间复杂度，然后时间复杂度的话只需要O(k)吧，
# 就是从后往前扫k个就ok了

# 另一就是不需要额外的空复，然后时复O(log(m+n))
# 递归
# https://blog.csdn.net/hk2291976/article/details/51107778


# 代码还有bug.......
a = [5,4,3,2,1]
b = [1]
k = 2
m, n = len(a), len(b)

def fun(a,m,b,n,k):
    if n>m:
        return fun(b,n,a,m,k)   # 反正就假定m更长
    if k==1:
        return a[0] if a[0] > b[0] else b[0]
    if k==2:
        tmp = a[:2]+b[:2]
        tmp.sort()
        return tmp[-2]
    if k == m+n:
        return a[-1] if a[-1] < b[-1] else b[-1]
    if k == m+n-1:
        tmp = a[-2:]+b[-2:1]
        tmp.sort()
        return tmp[1]

    # 因为两数组可能是不定长的，所以先短的补一下
    mimest = a[-1] if a[-1] < b[-1] else b[-1]
    for i in range(m-n):
        b.append(mimest)

    c1 = k/2 + 1
    c2 = k - c1
    # print c1,c2
    while c1 and c2 and c1 < m and c2 < m:
        if b[c2-1] >= a[c1] and a[c1-1] >= b[c2]: # 确保左边是大于右边的
            # 注意这里写的是a[c1-1] < b[c2-1] 因为a b两段是有序的，只需要比较这俩中
            # 更小的那一个，就是我们要的排位在k的那个！！！  这里比较容易出错！
            return a[c1-1] if a[c1-1] <= b[c2-1] else b[c2-1]
        elif a[c1] > b[c2-1]:
            c1 += 1
            c2 -= 1
        else:
            c2 += 1
            c1 -= 1

for i in range(1, m+n+1):
    print fun(a,m,b,n,i)