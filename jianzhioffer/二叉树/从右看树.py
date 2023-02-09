# coding=utf-8
# 从右向左看树
# 层次遍历二叉树，然后取每一层最后的那个节点就是所求

##### leetcode ac
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        d = {}
        def f(root, i):   # i为树的深度
            if root:
                d[i] = root.val
                f(root.left, i+1)
                f(root.right, i+1)  # right要放在后面，因为要把前面的left覆盖掉
        f(root, 0)
        return list(d.values())






# 非递归呢？
# 层次遍历
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        temp = []   # 存放每一层节点
        stack = []
        last = nextLast = root
        stack.append(root)
        while stack:
            node = stack.pop(0)
            temp.append(node.val)
            if node.left:
                stack.append(node.left)
                nextLast = node.left
            if node.right:
                stack.append(node.right)
                nextLast = node.right
            if node == last:
                res.append(temp[-1])
                temp = []   # 清空，接着放下一层的节点
                last = nextLast
        return res



class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
# init 树
t1 = TreeNode(1)
t2 = TreeNode(2)
t3 = TreeNode(3)
t4 = TreeNode(4)
t5 = TreeNode(5)
t1.left = t2
t1.right = t3
t2.left = t4
t2.right = t5
s = Solution()
res = s.rightSideView(t1)
print(res)