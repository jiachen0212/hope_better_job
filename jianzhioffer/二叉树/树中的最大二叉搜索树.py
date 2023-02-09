# coding=utf-8
# 树中最大的二叉搜素树
# 类似打家劫舍最后进阶那个思路

# https://www.cnblogs.com/grandyang/p/5188938.html

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def largestBSTSubtree(self, root):
        def helper(root):
            if root==None:
                return (2**31-1,-2**31,0)
            left = helper(root.left)  # left是有三个值的，前两个是左右节点的值，第三个是这个节点作为
            # 根节点的二叉搜索树节点个数
            right = helper(root.right) # right一样

            # left[1] 左边的最大，right[0]右边最小  所以可以构成搜索树
            if root.val > left[1] and root.val < right[0]:
                return (min(root.val, left[0]), max(root.val, right[1]), left[2]+right[2]+1)
            else:
                return (2**31-1,-2**31, max(left[2], right[2]))
        res = helper(root)
        return res[2]
