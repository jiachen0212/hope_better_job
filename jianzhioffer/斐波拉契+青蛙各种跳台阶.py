#coding=utf-8
# 斐波拉契/青蛙跳台阶
# 青蛙一次可跳一阶，也可能跳两阶。
# 所以跳上了n阶的为f(n), 如果第一次跳了一阶， 剩下就是f(n-1)种可能，
# 如果第一次跳了2阶，那么剩下就是f(n-2)种可能
# 所以f(n)=f(n-1) + f(n - 2)  就也是一个斐波拉契形式.

# 递归实现
def Fibnacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return Fibnacci(n-1) + Fibnacci(n-2)

# 非递归
# 较之迭代实现斐波拉契的方法节省很多时间
def Fibnacci(n):
    result = [0,1]
    if n <= 1:
        return result[n]
    for i in range(2,n+1):
        result.append(result[i-1]+result[i-2])
    return result[n]
res = Fibnacci(11)
print(res)



############################################################################################
# 青蛙一次可跳一个台阶或两个台阶 其实就是求斐波拉切数列
# 牛客ac
# -*- coding:utf-8 -*-
class Solution(object):
    def climbStairs(self, n):
        res = [0, 1]
        if n < 2:
            return res[n]
        a, b, i = 0, 1, 1
        while i <= n:
            c = a + b
            a = b
            b = c
            i += 1
        return c



# 升级版青蛙跳
# 一次可跳 1 2 3 .... n级， 求 n 级的跳法
# f(n) = f(n-1) + f(n-2) + ... + f(0)
# f(n-1) = f(n-2) + ... + f(0)
# 所以可知， f(n)=f(n-1)+f(n-1), 即 f(n)=2*f(n-1)
# 所以现在直接递归就好了～～～
# 牛客ac
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        if number == 1:
            return 1
        else:
            return self.jumpFloorII(number - 1) * 2




# 作业帮代码题：
# 1*3的砖  铺成20*3的种数
f=[0]*21
f[1],f[2],f[3]=1,1,2
for i in range(4,21):
    f[i]=f[i-1]+f[i-3]
print(f[20], '---')   # 1278