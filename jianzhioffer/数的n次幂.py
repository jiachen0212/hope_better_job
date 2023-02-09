#coding=utf-8
# 实现整数的乘方,考虑指数是负数,底数是0等情况..并使用右移运算实现除以2,更高效.使用位与运算判断是都为奇数.
# 最下面补充了一种位运算的方法，时间复杂度近O(logn)

def power(a, e):
    if a == 0 and e < 0:
        return 0
    if e == 0:
        return 1
    if e == 1:
        return a
    if e < 0:
        res = 1.0 / power(a, -e)   # 这里注意要用1.0而不是1,要不然就一直是return0了..
        return res
    res = power(a, e >> 1) # 使用位运算右移,实现高效除以2.
    res *= res  # 把迭代的乘底数替换成拆分成指数的一半乘法再平方. a^e=a^(e/2)*a^(e/2)
    if e & 0x1 == 1:  # 判断是否e是奇数   与00000...1做与运算,是奇数的话最后一位必定是1,则最终得到结果1.
        res *= a   # a^e=a^((e-1)/2)*a^((e-1)/2)*a
    if e < 0:
        res = 1 / power(a, -e)
    return res

res = power(95.123, 12)
print res



# leetcode ac
class Solution(object):
    def myPow(self, x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        if n < 0:
            return 1/(self.myPow(x, n*-1))
        half = self.myPow(x, n/2)  # 一半
        rem = self.myPow(x, n%2)
        return half *half * rem



# 二分法  牛客ac
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        if exponent == 0:
            return 1
        if exponent < 0:
            return 1 / (self.Power(base, -exponent))
        if exponent % 2 == 0:
            return self.Power(base, exponent / 2) * self.Power(base, exponent / 2)
        else:
            return self.Power(base, exponent / 2) * self.Power(base, exponent / 2) * base




## c++代码
# 位运算实现：时间复杂度近O(logn)
# m^!#  ==> m^1101 = m^0001 * m^0100 * m^1000
int pow(int n){
    int sum = 1;
    int tmp = m;
    while(n != 0){
        if(n & 1 == 1){
            sum *= tmp;
        }
        tmp *= tmp;
        n = n >> 1;
    }

    return sum;
}
