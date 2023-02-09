# coding=utf-8
# 二叉树生成字符串
# 递归

class Solution:
    def tree2str(self, t):
        res = ''
        if not t:
            return res
        res += str(t.val)  # 根节点先加进去
        if t.left:
            res += '(' + self.tree2str(t.left) + ')'
        else:   # 没有左孩子？ 那加一个代表左边没有的()
            if t.right:
                res += '()'
        if t.right:
            res += '(' + self.tree2str(t.right) + ')'
        return res


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
t1 = TreeNode(1)
t2 = TreeNode(2)
t3 = TreeNode(3)
t4 = TreeNode(4)
t1.left = t2
t1.right = t3
t2.right = t4

s = Solution()
res = s.tree2str(t1)
print(res)

