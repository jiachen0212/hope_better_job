# coding=utf-8

'''
两个玩家，可以对所有未染色的邻进行染色   最后谁染色多谁获胜

统计x左节点总数、统计x右节点总数 3.从左，右，非x子树（父）的节点数中，
找到最大的数字，看是否大于n/2
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def btreeGameWinningMove(self, root, n, x):
        self.right,self.left = 0,0
        # max(self.left, self.right, n-1-self.left-self.right)
        # 是2号玩家可以染色的节点数
        return self.nodesCount(root, x) / 2 < max(self.left, self.right, n-1-self.left-self.right)


    # 计算x对应的节点下的邻个数
    def nodesCount(self, node, x):
        if not node:
            return 0
        l = self.nodesCount(node.left, x)
        r = self.nodesCount(node.right, x)
        if node.val == x:
            self.left = l
            self.right = r
        return l+r+1