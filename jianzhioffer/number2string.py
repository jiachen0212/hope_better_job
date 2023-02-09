#coding=utf-8
# 把数字翻译成字符:0~25---a~z. 求某个整数可以被翻译成的种数.
# 如12--ab或m.
# 结题思路是:使用递归并且由末尾开始从右到左翻译,避免重复运算.  i的定义是最左边为0开始计数.
# f(i) = f(i-1)+f(i-2)*g(i,i-1)   g(i,i-1)在i+1和i组成的两位数属于:10~25时候才等于1.


def fun(a):
    if a < 0:
        return 0  # 负数可被翻译成字符串的种数是 0种.
    a = str(a)  # 首先把int转str,方便获取每个位置上的数值
    if len(a) == 1:  # 仅个位数,那么也只要一种翻译方法
        return 1
    f1 = 0
    f2 = 1
    for i in range(len(a) - 2, -1, -1):
        # print a[i]+a[i+1]
        if int(a[i] + a[i+1]) >= 10 and int(a[i] + a[i+1]) <= 25:
            g = 1
        else:
            g = 0
        temp = f2
        f2 = f2 + g * f1
        f1 = temp
    return f2

res = fun(12258)
print res




