#coding=utf-8

# 但是牛客不 ac

# 寻找1~N的所有整数中1出现的次数...包括个十百千位等..
# 结题思路: 个, 十, 百, 位等出现 1 的情况分开计算.
# 详细分析过程见<编程之美>P135

# 经过分析,出现1的次数需要一次性考虑连续的3个位置.
def count_one(n):
    count = 0
    factor = 1
    while n / factor != 0:
        lower = n - (n / factor) * factor  # 取到3个位置中的最低位
        cur = (n / factor) % 10  # 中间位置
        higher = n / (factor * 10)
        if cur == 0:
            count += higher * factor
        elif cur == 1:
            count += higher * factor + lower + 1
        else:
            count += (higher + 1) * factor
        factor *= 10
    return count

num = count_one(2)
print num
