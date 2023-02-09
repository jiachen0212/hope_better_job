# coding=utf-8
# trim 修剪二叉搜索树
'''
给定一个二叉搜索树，同时给定最小边界L 和最大边界 R。通过修剪二叉搜索树，使得所有节点的值在[L, R]中 (R>=L) 。你可能需要改变树的根节点，所以结果应当返回修剪好的二叉搜索树的新的根节点。

'''

## 递归
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        if root is None:
            return None
        if root.val >= L and root.val <= R :
            root.left = self.trimBST(root.left, L, R)
            root.right = self.trimBST(root.right, L, R)
            return root
        if root.val < L :
            return self.trimBST(root.right, L, R)
        # root.val > R
        return self.trimBST(root.left, L, R)



### 非递归
# 其实还蛮好理解的
class Solution:
    def trimBST(self, root, L, R):
        node = root
        while node and not (L<=node.val<=R): # 根节点不在LR范围，要被删掉
            if node.val<L:
                node = node.right
            elif node.val>R:
                node = node.left  # 所以重新找根节点
        root = node
        head_queue = [root]  # 存放"根节点" 先进先出
        while head_queue:
            head = head_queue.pop(0)
            if not head:
                continue
            node = head.left # 找左节点
            while node and node.val<L:  # 左节点比L小，那左节点只能用它的右兄弟代替了
                node = node.right
            head.left = node
            head_queue.append(head.left) # (整个)左子节点加入队列

            node = head.right # 找右节点
            while node and node.val>R:
                node = node.left
            head.right=node
            head_queue.append(head.right)
        return root



class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
t1 = TreeNode(3)
t2 = TreeNode(0)
t3 = TreeNode(4)
t4 = TreeNode(2)
t5 = TreeNode(1)

t1.left = t2
t1.right = t3
t2.right = t4
t4.left = t5
s = Solution()
res = s.trimBST(t1, 1,3)
print(res.val)



