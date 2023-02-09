# coding=utf-8
# 二叉树每一层的最大值
# 层次遍历  用dfs遍历树

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def largestValues(self, root):
        res = []
        q = []
        if not root:
            return res
        q.append(root)
        while q:
            level_len = len(q)  # 每一层的节点个数
            mmax = -2**31
            for i in range(level_len):
                node = q.pop(0)  # dfs用堆  bfs用队列 所以是pop(0)先进先出
                mmax = mmax if mmax > node.val else node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(mmax)
        return res
