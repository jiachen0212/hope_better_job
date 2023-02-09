# coding=utf-8
# 最小众倍数

'''
这个数组还是递增的
给定5个正整数, 它们的最小的众倍数是指的能够被其中至少3个数整除的最小正整数。
in: 1 2 3 4 5  out: 4
'''

def fun(nums):
    tmp = nums[2]  # 众倍数必定是>=nums[2]的
    while tmp:
        count = 0
        for j in range(5):
            if tmp % nums[j] == 0:
                count += 1
                # print nums[j], count
        if count >= 3:   # 注意这里是>=  不是==
            return tmp
        else:
            tmp += 1
            # print tmp

res = fun([1,3,5,7,11])
print res


