# coding=utf-8
#### 实现每个节点都有一个硬币

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 问题抽象为后序遍历，左右父，硬币都交到父节点那，有多少个，需要移动的步数就是个数-1
# -1是因为父节点上那个硬币是可以不动的
class Solution(object):
    def distributeCoins(self, node):
        res = [0]
        def move_count(node):
            if not node:
                return 0
            else:
                l = move_count(node.left)
                r = move_count(node.right)
                count_ =  node.val + l + r - 1
                res[0] += abs(count_)
                return count_

        move_count(node)
        return res[0]