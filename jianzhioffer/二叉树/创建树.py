#coding=utf-8
# 创建二叉树

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
t1 = TreeNode(6)
t2 = TreeNode(5)
t3 = TreeNode(8)
t4 = TreeNode(2)
t5 = TreeNode(7)
t6 = TreeNode(4)
t7 = TreeNode(9)
t1.left = t2
t1.right = t3
t2.left = t4
t2.right = t5
t3.left = t6
t3.right = t7
# 传入根节点就可以使用这棵树了