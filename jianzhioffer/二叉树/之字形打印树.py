#coding=utf-8
# 之字形打印二叉树
# 层次遍历  然后分奇偶打印节点

# leetcode ac
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        level_queue = [root]
        all_levels = []
        while level_queue:
            nodes = []   # 注意这个nodes才是真的在添加节点  其他几个[]都是在遍历
            next_level = []
            for i in level_queue:
                nodes.append(i.val)
                if i.left:
                    next_level.append(i.left)
                if i.right:
                    next_level.append(i.right)
            level_queue = next_level
            all_levels.append(nodes)

        prints = []
        for ind, lel in enumerate(all_levels):
            if ind%2 == 0:
                prints.append(lel)
            else:
                prints.append(lel[::-1])
        return prints
