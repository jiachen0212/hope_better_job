# coding=utf-8
# 非递归后序遍历

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def postorderTraversal(self, root):
        res = []
        stack = []
        curr = root

        while curr or stack:
            if curr:
                stack.append(curr)
                res.insert(0, curr.val)
                curr = curr.right
            else:
                curr = stack.pop()
                curr = curr.left
        return res