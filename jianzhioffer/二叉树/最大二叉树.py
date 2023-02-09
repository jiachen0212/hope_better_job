# coding=utf-8
# 最大二叉树
'''
二叉树的根是数组中的最大元素。
左子树是通过数组中最大值左边部分构造出的最大二叉树。
右子树是通过数组中最大值右边部分构造出的最大二叉树。
通过给定的数组构建最大二叉树，并且输出这个树的根节点。

输入：[3,2,1,6,0,5]
输出：返回下面这棵树的根节点：

      6
    /   \
   3     5
    \    /
     2  0
       \
        1
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        return self.maxTree(nums, 0, len(nums)-1)

    def maxTree(self,lists, l, r):
        if l>r:
            return None
        ge = self.findMax(lists, l, r)  # 找到根节点
        root = TreeNode(lists[ge])
        root.left = self.maxTree(lists, l, ge-1)
        root.right = self.maxTree(lists, ge+1, r)
        return root

    def findMax(self, nums, l, r):
        mmax = -2**31
        maxind = r
        for i in range(l,r+1):
            if nums[i] > mmax:
                mmax = nums[i]
                maxind = i
        return maxind

