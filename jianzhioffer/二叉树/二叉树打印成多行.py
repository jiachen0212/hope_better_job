# -*- coding:utf-8 -*-
##### 牛客 ac
##### 从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if not pRoot:
            return []
        nodeStack = [pRoot]
        result = []
        while nodeStack:
            res = []
            nextStack = []
            for i in nodeStack:
                res.append(i.val)
                if i.left:
                    nextStack.append(i.left)
                if i.right:
                    nextStack.append(i.right)
            nodeStack = nextStack
            result.append(res)
        return result
