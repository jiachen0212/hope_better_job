# coding=utf-8
# 完全二叉树节点个数

# 利用完全二叉树特性，使用递归 时间复杂度：O(logn*logn)
# 满二叉树节点总个数为2**n-1    2^0+2^1+2^2+...2^n = (1-2^n)/ (1-2) = 2^n - 1

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def countNodes(self, root):
        # 返回当前节点下，树的高度
        def height(t):
            h = -1
            while t:
                h += 1
                t = t.left
            return h

        h = height(root)  # 树总高度

        nodes = 0
        while root:
            if height(root.right) == h - 1:
                nodes += 2**h
                root = root.right
            else:
                nodes += 2**(h - 1)
                root = root.left
            h -= 1
        return nodes



t1 = TreeNode(1)
t2 = TreeNode(2)
t3 = TreeNode(3)
t4 = TreeNode(4)
t5 = TreeNode(5)
t6 = TreeNode(6)

t1.left = t2
t1.right = t3
t2.left = t4
t2.right = t5
t3.left = t6
s = Solution()
res = s.countNodes(t1)
print(res)



