#coding=utf-8
# 判断是否是平衡二叉树
# 建议使用后序遍历
# 一：空树
# 二：它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def getDepth(self, Root):
        if Root == None:
            return 0
        lDepth = self.getDepth(Root.left)
        rDepth = self.getDepth(Root.right)
        return max(lDepth, rDepth) + 1

    def IsBalanced_Solution(self, pRoot):
        if not pRoot:
            return True
        lDepth = self.getDepth(pRoot.left)
        rDepth = self.getDepth(pRoot.right)
        diff = lDepth - rDepth
        if diff < -1 or diff > 1:
            return False
        # self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
        # 这是继续check下面的子树
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)