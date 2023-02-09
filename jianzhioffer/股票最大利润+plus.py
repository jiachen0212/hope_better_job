#coding=utf-8
# 股票的最大利润  逐个遍历list数值作为卖出价格s[i],买入价格为是s[:i]中最小的数时,利润最大.


# 一次买股票的最大利润
# def maxlirun(s):
#     if len(s) < 2:
#         return 0
#     thein = s[0]
#     theout = s[1]
#     maxlr = theout - thein
#     for i in range(1, len(s)):  # 遍历所有可能的卖出值 从1开始遍历
#         min_in = min(s[:i])
#         curlr = (s[i] - min_in)
#         if curlr > maxlr:
#             maxlr = curlr
#             print s[i], min_in
#     return maxlr

# res = maxlirun([9, 11, 8, 5, 7, 12, 16, 14])
# print res



# leetcode diff
'''
给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票
最多进行两次交易，只进行一次也行。
输入: [3,3,5,0,0,3,1,4]
输出: 6
解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。

'''
# 三维动态规划
# 卖出才算完成一次交易，单买入不算
class Solution:
    def maxProfit(self, prices):
        if not prices:
            return 0
        n = len(prices)
        dp = [[[0]*2 for _ in range(3)] for _ in range(n)]
        # dp[i][j][0/1] 0/1代表是否持有股票 ij为第i天交易了第j次
        for j in range(3):
            dp[0][j][0], dp[0][j][1] = 0, -prices[0]

        for i in range(1,n):
            for j in range(3):
                if not j: # j==0 即第i天没有进行交易
                    dp[i][j][0] = dp[i-1][j][0]
                else:  # 第i天进行了交易
                    dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j-1][1]+prices[i])
                    # dp[i-1][j][1]+prices[i] 表示i-1天是有的，i天卖出去了，所以+proces[i]
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j][0]-prices[i])
                # dp[i-1][j][0]-prices[i] i-1天没股票，i天买入，所以-prices[i]
        return max(dp[n-1][0][0], dp[n-1][1][0], dp[n-1][2][0])

s = Solution()
print(s.maxProfit([3,3,5,0,0,3,1,4]))
