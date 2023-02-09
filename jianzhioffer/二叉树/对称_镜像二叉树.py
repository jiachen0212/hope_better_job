#coding=utf-8
########### 牛客 ac
# 通过使用前序遍历和前序对称遍历,比较两结果是否一致,来判断这颗树是否是对称树
# 递归实现

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def qianxu_duichenqianxu(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False

        # 这一句是精髓...
        return self.qianxu_duichenqianxu(left.left, right.right) and self.qianxu_duichenqianxu(left.right, right.left)

    def isSymmetrical(self, pRoot):
        if not pRoot:
            return True
        return self.qianxu_duichenqianxu(pRoot.left, pRoot.right)
