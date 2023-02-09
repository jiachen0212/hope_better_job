# coding=utf-8
# 4sum==0
# 不需要去分几个零那么复杂，直接AB遍历求和  BC遍历求和
# 用字典帮助存储和出现的次数
# 求有多少种这样的组合


def foursum(A, B, C, D):
    count = 0
    ab_dict = dict()

    # 遍历 AB
    for a in A:
        for b in B:
            ab_dict[a+b] = ab_dict.get(a+b, 0) + 1
            # dict.get(key, 0)   # 有这个key就返回key对应的value值，没有的话就返回后面给的0

    # 接着遍历CD
    for c in C:
        for d in D:
            s = -(c+d)
            if s in ab_dict:
                count += ab_dict[s]
    return count

