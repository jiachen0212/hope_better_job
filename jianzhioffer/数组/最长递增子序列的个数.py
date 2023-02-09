# coding=utf-8
# https://blog.csdn.net/xuxuxuqian1/article/details/81071975

# 最长递增子序列的个数

# dp[] con[]  分别记录最长的长度和最长串的个数
'''
对于每个遍历到的数字nums[i]，我们再遍历其之前的所有数字nums[j]
当nums[i]小于等于nums[j]时，不做任何处理，因为不是递增序列。
反之，则判断dp[i]和dp[j]的关系，如果dp[i]等于dp[j] + 1，
说明nums[i]这个数字可以加在以nums[j]结尾的递增序列后面，
并且以nums[j]结尾的递增序列个数可以直接加到以nums[i]结尾的递增序列个数上。
如果dp[i]小于dp[j] + 1，说明我们找到了一条长度更长的递增序列，
那么我们此时将dp[i]更新为dp[j]+1，并且原本的递增序列都不能用了，
cnt[i]直接用cnt[j]来代替。
'''

def fun(nums):
    l = len(nums)
    dp = [1]*l
    con = [1]*l
    maxlen = 1
    res = 0
    for i in range(l):
        for j in range(i):
            if nums[i] > nums[j]:
                if dp[j]+1 == dp[i]:
                    con[i] += con[j]
                if dp[j]+1>dp[i]:
                    # 说明找到了更长的最长串 con得重置了
                    dp[i] = dp[j]+1
                    con[i]=con[j]
        maxlen = maxlen if maxlen > dp[i] else dp[i]
    for i in range(l):
        if dp[i] == maxlen:
            res += con[i]
    return res

