# coding=utf-8
# 二叉树的前序遍历  非递归

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root):
        stack, result = [], []
        if root:
            stack.append(root)

        while stack:
            node = stack.pop()
            result.append(node.val)
            # 因为栈是先入后出，所以先放right 再放left
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return result
