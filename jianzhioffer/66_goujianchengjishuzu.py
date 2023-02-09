#coding=utf-8
# 构建乘积数组 A=[0,1,2....n-1] >> B[i]=A[1]*A[2]*...*A[i-1]*A[i+1]*...*A[n-1] 就是唯独不乘以A[i]
# 要求不能同除法,即B[i]不可以通过A[1]*..*A[n-1]/A[i]得到..
# 思路: 把B[i]分成0~i-1 C 和 i+1~n-1 D 两部分乘. C[i]=C(i-1)*A[i]  D[i]=D(i+1)*A[i] 可以看出C D都是可以用递归实现的~


def fun(a):
    c = [1 for x in range(len(a))]  # 创建一个和a等长的list
    for i in range(1, len(a)):
        c[i] = c[i-1] * a[i-1]
    temp = 1
    for j in range(len(a) - 2, -1, -1):
        temp *= a[j + 1]  # 这个temp存的其实是D
        # 其实只用到了c的后半段...
        c[j] *= temp  # c现在变成B了
    return c
s = [x for x in range(1, 5)] # [1 2 3 4 5]
res = fun(s)


