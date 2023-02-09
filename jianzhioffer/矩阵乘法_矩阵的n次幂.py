# coding=utf-8
# 幂运算时间复杂度：logn，矩阵运算时间复杂度：n^3.  所以总共是logn * n^3
def myPow(x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        # if n < 0:
        #     return 1/(myPow(x, n*-1))
        half = myPow(x, n/2)  # 一半
        rem = myPow(x, n%2)
        if half != 1 and rem != 1:
            return matrixMul(matrixMul(half, half), rem)
        else:
            return matrixMul(half, half)


# 矩阵乘法
def matrixMul(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    res[i][j] += A[i][k] * B[k][j]
        return res
    return ('输入矩阵有误！')


a = [[1,2], [3,4]]
res = myPow(a, 5)
print(res)