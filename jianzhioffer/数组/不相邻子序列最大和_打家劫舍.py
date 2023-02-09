# coding=utf-8
# 不相邻子序列最大和

# 打家劫舍
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0
        l = len(nums)
        if l < 3:
            return max(nums)

        dp = [0]*l
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, l):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
        return dp[-1]

s = Solution()
res = s.rob([2,3,2])
print res



# 打家劫舍升级版  第一和最末认为是相邻的
# 那么就分两种情况  抢第一和抢最末  然后取两情况的max
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0

        l = len(nums)
        if l < 3:
            return max(nums)

        # 不抢第一个
        dp1 = [0]*l  # dp[0]=0
        dp1[1] = nums[1]  # dp[1]直接是nums[1]
        # 因为不抢第一个，所以最后一个是可以遍历到的
        for i in range(2,l):
            dp1[i] = max(dp1[i-2]+nums[i], dp1[i-1])

        dp2 = [0]*l
        dp2[0]=nums[0]  # 考虑抢第一个
        dp2[1]=max(nums[:2])
        for i in range(2,l-1):
            dp2[i] = max(dp2[i-2]+nums[i], dp2[i-1])
        return max(dp1[-1], dp2[-2])




#### 再升级
# https://leetcode-cn.com/problems/house-robber-iii/submissions/
'''
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \
     3   1

输出: 7
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def rob(self, root):
        if root==None:
            return 0
        def helper(root):
            if root==None:
                return [0,0]
            left = helper(root.left)
            right = helper(root.right)
            # 有根节点root 就不能有root.left root.right
            rob = root.val + left[1] + right[1]  # 0表示包含当前节点  1表示不包含
            skip = max(left) + max(right)
            return [rob, skip]
        res = helper(root)
        return max(res)