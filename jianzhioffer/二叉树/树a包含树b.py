# 牛客ac

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def DoesTree1HaveTree2(self, pRoot1, pRoot2):
        if pRoot2 == None:
            return True
        if pRoot1 == None:
            return False
        if pRoot2.val != pRoot1.val:
            return False
        # 迭代是去比较这两颗子树的左右子树..
        return self.DoesTree1HaveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1HaveTree2(pRoot1.right, pRoot2.right)

    # 主函数
    def HasSubtree(self, pRoot1, pRoot2):
        if pRoot1 is None or pRoot2 is None:
            return False
        result = False
        if pRoot1.val == pRoot2.val:  # 这里表示找到了AB中节点值相等的点了,就调用上面的函数看它们下面的子树是否结构一致
            result = self.DoesTree1HaveTree2(pRoot1, pRoot2)
        if result == False:
            result = self.HasSubtree(pRoot1.left, pRoot2)  # 如果A的父节点和B的父节点值不一样,那就接着找A的左树
        if result == False:
            result = self.HasSubtree(pRoot1.right, pRoot2) # 就接着找A的右树
        return result