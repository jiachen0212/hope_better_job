# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))   # pre.pop(0) 把此刻的根节点删掉了， 所以下面可以直接
        # 使用pre 而无需做别的index设置
        index = tin.index(root.val)   # 返回根节点在中序遍历中的位置，从而区分左右树
        root.left = self.reConstructBinaryTree(pre, tin[:index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root