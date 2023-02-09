# coding=utf-8
# 喝酒问题  3个瓶子换一瓶酒

def drink(n):
    if n < 3:
        return n
    else:
        return (n-n%3) + drink(n/3 + n%3)
        # num-num%3：能换啤酒的瓶子  num/3+num%3：新的啤酒瓶数
print drink(10)
