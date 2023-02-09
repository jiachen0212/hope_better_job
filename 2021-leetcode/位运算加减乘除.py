#coding=utf-8
# 实现两个数相加,但不用+-x/等四则运算符实现.. 那么就只能用位运算了...
# 首先考虑不进位的话: 11--0 10--1 01--1 00--0 所以可以抽象为是异或运算
# 然后是考虑进位: 11--1 10--0 01--0 00--0 所以可以抽象为与运算 并且得到的结果要向左移动一位..
# ok,上代码:

def add_without_jiajianchengchu(a, b):
    while b != 0:
        ssum = a ^ b  # 异或
        jinwei = (a & b) << 1  # 做与运算并左移一位

        # 好了现在要把ssum和jinwei的结果相加了.但是因为没有+可以用,所以这的加也只能按照前两步一样的方法来:即进行异或和进位计算,直到没有进位为止...
        a = ssum
        b = jinwei  # 所以是进行了个循环其实... 这一点很巧妙..
    return a   # b=0的话,a+b直接等于a啊,就直接return a

# res = add_without_jiajianchengchu(5, 3)
res = add_without_jiajianchengchu(-3, -4)
print res