# coding=utf-8
# 是否2的整数次幂
# 有技巧  一行搞定
# a&(a-1) 2的幂只有最右边是1
# a&(a-1) 的功能是去掉a中最右边的1  那么a&(a-1)一次就==0
# 的话，也就是a只有一个1  也就是2的幂了

def fun(a):
    res = a&(a-1)  # 消除a中最右边的1
    if not res:
        return True
    return False

print fun(18)