# 二叉树转双向链表   牛客 ac
# 先中序遍历，将所有的节点保存到一个列表中。对这个list[:-1]进行遍历，每个节点的right设为下一个节点，下一个节点的left设为上一个节点。

# -*- coding:utf-8 -*-
# 中序遍历   左父右
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        if not pRootOfTree:
            return
        self.arr = []

        self.midTraversal(pRootOfTree)  # 中序遍历
        for i,v in enumerate(self.arr[:-1]):
            # 每个节点的right设为下一个节点，下一个节点的left设为上一个节点。
            v.right = self.arr[i + 1]
            self.arr[i + 1].left = v
        return self.arr[0]


    # 中序遍历
    def midTraversal(self, root):
        if not root:
            return
        self.midTraversal(root.left)
        self.arr.append(root)
        self.midTraversal(root.right)