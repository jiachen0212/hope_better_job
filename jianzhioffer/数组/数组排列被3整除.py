# coding=utf-8
# 数组排列 被3整除

'''
dp[i][j]
对于第𝑖位数字，之前的能组成的余数为𝑗的序列数目。
https://www.cnblogs.com/caomingpei/p/11042268.html
这个dp变化思想还是蛮巧妙的
'''

def fun(nums):
    count = [0,0,0]
    for num in nums:
        if num%3 == 0:
            count[0] *= 2
            count[1] *= 2
            count[2] *= 2
            count[0] += 1
        elif num%3 == 1: # 2+1=%0                      0+1=%1      1+1=%2
            count = [count[2]+count[0], count[1]+count[0]+1, count[1]+count[2]]
        else:            # 1+2=%0             2+2=%1             0+2=%2
            count = [count[1]+count[0], count[2]+count[1], count[0]+count[2]+1]

    return count[0] % 1000000007

print fun([1,2,3])