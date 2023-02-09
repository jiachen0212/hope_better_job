# coding=utf-8
# 给中序遍历，验证是否是二叉搜索树

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # using inorder - binary search tree will be ascending order
        stack = []
        cur = root
        pre = None
        while len(stack) or cur:
            if cur:
                stack.append(cur)
                cur = cur.left  # 左节点一直压入栈
            else:
                p = stack.pop()
                # pop()弹出最末,pre=p,这个p是上一个最末，也即当前p的左孩子
                if pre and p.val <= pre.val:  # 父<=左子，那就false了
                    return False
                pre = p
                cur = p.right
        return True