# coding=utf-8
# 连续自然数的和为1000

# 下面的代码可以得出3种，然后加上[1000] 这个，所以总共是4种


for n in range(2, 1000):
    x = (1000 - n*(n-1)/2) / n
    a = range(x, x + n)
    ssum = sum(a)
    if ssum == 1000:
        if x > 0:
            print(x, n)