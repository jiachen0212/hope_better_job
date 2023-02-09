#coding=utf-8


##### 要求递增子序列连续
# 时复 O(n)
a = [10, 80, 6, 3, 4, 7, 1, 5, 11, 2, 12, 30, 31]
concoll = [0]
for i in range(len(a) - 1):
    if a[i] < a[i+1]:
        count = concoll[-1] + 1
        concoll.append(count)
    else:
        concoll.append(1)
maxcou = max(concoll)
print maxcou



##### 递增子序列可不连续
'''
[1,6,2,3,7,5]  ==> [1,2,3,5]
返回的是序列的长度，不是递增序列
时复：O(n^2)
'''
# 动态规划就难在 更新条件怎么变！！！
class Solution(object):
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        l = len(nums)
        dp = [1]*l
        res = 1
        for i in range(l):
            for j in range(i):  # 注意是i
                if dp[j]+1>dp[i] and nums[i]>nums[j]:  # 状态转移方程
                    dp[i] = dp[j]+1
            res = max(res, dp[i])   # 这里注意 不是return dp[-1] 而是用一个变量更新最长子序列值啊
        return res

s = Solution()
res = s.lengthOfLIS([10, 80, 6, 3, 4, 7, 1, 5, 11, 2, 12, 30, 31])
print res

# 利用二分查找+dp思想！，时复O(nlogn)
'''
   dp[i]: 所有长度为i+1的递增子序列中, 最小的那个序列尾数.
        由定义知dp数组必然是一个递增数组, 可以用 maxL 来表示最长递增子序列的长度.
        对数组进行遍历, 依次判断每个数num将其插入dp数组相应的位置:
        1. num > dp[maxL], 表示num比所有已知递增序列的尾数都大, 将num添加入dp
           数组尾部, 并将最长递增序列长度maxL加1
        2. dp[i-1] < num <= dp[i], 只更新相应的dp[i] dp的最末元素可以变小
        但此时的maxlen是不变的
'''
# 这个真的太强了！！！
def fun(nums):
    l = len(nums)
    dp = [0]*l  # dp[i]:递增子串长度为i中，最尾元素最小的那个，的尾元素值
    maxlen = 0
    for i in range(l):   # 最外层时复是O(n)  内部的二分时复是O(logn)
        lo, hi = 0, maxlen   # 所以总的时复是O(nlogn)
        while lo < hi:
            mid = (lo+hi)/2
            if dp[mid] < nums[i]:
                lo = mid + 1
            else: # dp[mid]>=nums[i]
                hi = mid
        dp[lo] = nums[i]   # 以上代码是二分查找的把nums填入dp

        if lo == maxlen:  # 如果lo!=maxlen的话 证明插入的值在0～maxlen之间
            # 新来的值不比dp[-1]大，最大递增串也没能+1
            maxlen += 1
    return maxlen
print fun([10, 80, 6, 3, 4, 7, 1, 5, 11, 2, 12, 30, 31])