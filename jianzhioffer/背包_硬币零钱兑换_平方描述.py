# coding=utf-8


# 返回可以有几种拼凑金额方法
# 如6  可以 1+5 1*6
a = [1, 5, 10, 20, 50, 100]
def dp(num):
    num = int(num)
    dp = [1 for i in range(0, num + 1)]
    for i in range(1, 6):  # 这个6是共有6种面额
        for j in range(1, num + 1):  # j为需要拼凑的金额值
            if j >= a[i]:
                dp[j] = dp[j] + dp[j - a[i]]
    return dp[-1]
num = raw_input()  # num为输入的金额值
res = dp(num)
print res,'==='



# 平方个描述次数
# 5=2^2+1^2
# 3=1^2+1^2+1^2
# leetcode ac 代码
class Solution(object):
    def numSquares(self, n):
        f = [i for i in range(n+1)]
        for i in range(n+1):
            j = 1
            while j*j <= i:
                f[i] = min(f[i], f[i-j*j]+1)
                j+=1
        return f[-1]

s = Solution()
print s.numSquares(19), '+++'



# dp 动态规划做
# 零钱兑换问题，这个是返回最少的兑换张数
'''
输入: coins = [1, 2, 5], amount = 11
输出: 3
解释: 11 = 5 + 5 + 1

输入: coins = [2], amount = 3
输出: -1
'''

class Solution(object):
    def coinChange(self, coins, amount):
        dp = [amount+100] * (amount+1)  # 这里先把最坏的兑换可能预设好
        dp[0]=0
        for i in range(1, amount+1):
            for coin in coins:
                if i >= coin:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        return dp[-1] if dp[-1] != amount+100 else -1

s = Solution()
ans = s.coinChange([1,2,5], 12)
print(ans)



# dfs版本  但是有点绕
'''
广度遍历(bfs),所得到一定是,可以用最少硬币达到的路径
执行实现一直添加visited 和 可走的路径cur就可以
'''
class Solution(object):
    def coinChange(self, coins, amount):
        res = 0  # 兑换结果值
        cur = [0] # 图可能的所有路径
        visited = set()  #  拼凑好的面额存在这记录
        # 注意是set  值不重复
        coins.sort()
        while cur:
            next_time = []
            res += 1
            for tmp in cur:
                for coin in coins:
                    sum_num = tmp + coin
                    if sum_num == amount:
                        return res
                    elif sum_num > amount:
                        break  # 跳出cons的循环  遍历下一个tmp
                    elif sum_num < amount and sum_num not in visited:
                        next_time.append(sum_num)
                        visited.add(sum_num)
            cur = next_time
        return -1 if amount else 0

s = Solution()
print s.coinChange([1,2,5], 11)



