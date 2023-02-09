# coding=utf-8
# 二叉树中的最大路径和
'''
输入: [1,2,3]

       1
      / \
     2   3

输出: 6

输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42
'''



# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    res = float('-inf')
    def maxPathSum(self, root):
        self.getMax(root)
        return self.res

    def getMax(self,root):
        if not root:
            return 0
        # 如果子树路径和为负则应当置0,表示不要子树
        left = max(0, self.getMax(root.left))
        right = max(0, self.getMax(root.right))
        self.res = max(self.res, root.val + left + right)
        return max(left, right) + root.val  # getMax函数是只返回单边的！！！