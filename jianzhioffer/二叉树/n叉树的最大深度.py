# coding=utf-8
# n叉树的最大深度

# Definition for a Node.
# class Node(object):
#     def __init__(self, val, children):
#         self.val = val
#         self.children = children

class Solution(object):
    def maxDepth(self, root):
        if not root:
            return 0
        queue = [root]
        level_size = 1
        height = 0
        while queue:
            node = queue.pop(0)
            level_size -= 1
            if len(node.children):  # n叉树这个节点有孩子
                queue.extend(node.children)
            if level_size == 0:
                height += 1
                level_size = len(queue)
        return height