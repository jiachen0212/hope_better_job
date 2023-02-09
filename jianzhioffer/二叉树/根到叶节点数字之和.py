# coding=utf-8

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
'''
输入: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
输出: 1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495.
从根到叶子节点路径 4->9->1 代表数字 491.
从根到叶子节点路径 4->0 代表数字 40.
因此，数字总和 = 495 + 491 + 40 = 1026.

'''

class Solution(object):
    def sumNumbers(self, root):
        if root is None:
             return 0
        if not any([root.left, root.right]):
            return root.val
        left, right = 0, 0
        if root.left:
            root.left.val += root.val * 10
            left = self.sumNumbers(root.left)
        if root.right:
            root.right.val += root.val * 10
            right = self.sumNumbers(root.right)
        return left + right
