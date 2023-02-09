# coding=utf-8
# 虹软笔试题

class Solution(object):
    # 计算以当前点为根结点时，树的最大深度
    def diameterOfBinaryTree(self, root):
        if not root:
            return 0
        max_dia_left = self.diameterOfBinaryTree(root.left)
        max_dia_right = self.diameterOfBinaryTree(root.right)
        max_dia = max(self.get_depth(root.left)+self.get_depth(root.right),max_dia_left,max_dia_right)
        # max: 1.当前结点最大距离；2.左、右子结点的最大距离
        return max_dia

    def get_depth(self,root):  #计算以当前结点为根时，树的最大深度；
        if not root:
            return 0
        else:
            return max(self.get_depth(root.left),self.get_depth(root.right))+1




# 二叉树两节点的最长路径  其实就是左右子树最大和
'''
          1
         / \
        2   3
       / \
      4   5
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。

leetcode easy
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def diameterOfBinaryTree(self, root):
        self.result = 0
        self.Deep(root)
        return self.result

    def Deep(self, root):
        if not root:
            return -1
        if root.left:
            left = self.Deep(root.left) + 1
        else:
            left = 0
        if root.right:
            right = self.Deep(root.right) + 1
        else:
            right = 0
        self.result = left+right if left+right > self.result else self.result
        return left if left > right else right