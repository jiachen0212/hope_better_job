#coding=utf-8
# 给定某值, 返回二叉树中,节点值之和等于该值的所有路径
# 递归

####### 牛客 ac
class Solution:
    def FindPath(self, root, expectNumber):
        if not root:
            return []
        ret = []  # 返回二维列表, 内部每个单维列表表示找到的路径
        path = []
        self.Find(root, expectNumber, ret, path)
        return ret

    def Find(self, root, target, ret, path):
        if not root:
            return
        path.append(root.val)   # 最上面的父节点开始
        isLeaf = (not root.left and not root.right)   # 判断是否已经到叶节点了
        if isLeaf and (target == root.val):
            ret.append(path[:])  # 这里这一步要千万注意是path[:] 而不是直接 path.
            # 假如是:ret.append(path), 结果是错的. 因为Python可变对象都是引用传递.
        # 还没达到叶节点,往下遍历左右子节点.
        if root.left:
            self.Find(root.left, target - root.val, ret, path)
        if root.right:
            self.Find(root.right, target - root.val, ret, path)
        path.pop()   # 父节点下面不管有没有左右子，这个环节都被加入到路径了，所以要pop掉它


