#coding=utf-8
# 用队列，先进先出。 某节点的子节点在该节点被打印出后存入队列

# 牛客 ac
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        queue = []   # 队列
        result = []  # 打印的节点
        if root == None:
            return result
        queue.append(root)
        while queue:
            newNode = queue.pop(0)  # 即每次打印队列的首元素
            result.append(newNode.val)
            # 有左右子树的话,依次遍历这些树,然后加入到队列
            if newNode.left != None:
                queue.append(newNode.left)
            if newNode.right != None:
                queue.append(newNode.right)
        return result





#####
    # 给定二叉树的前序遍历和中序遍历,获得该二叉树
    def getBSTwithPreTin(self, pre, tin):
        if len(pre) == 0 | len(tin) == 0:
            return None
        root = treeNode(pre[0])
        for order, item in enumerate(tin):
            if root .val == item:
                root.left = self.getBSTwithPreTin(pre[1:order+1], tin[:order])
                root.right = self.getBSTwithPreTin(pre[order+1:], tin[order+1:])
                return root

class treeNode:
    def __init__(self, x):
        self.left = None
        self.right = None
        self.val = x

if __name__ == '__main__':
    flag = "printTreeNode"
    solution = Solution()
    preorder_seq = [1, 2, 4, 7, 3, 5, 6, 8]
    middleorder_seq = [4, 7, 2, 1, 5, 3, 8, 6]
    treeRoot1 = solution.getBSTwithPreTin(preorder_seq, middleorder_seq)
    if flag == "printTreeNode":
        newArray = solution.PrintFromTopToBottom(treeRoot1)
        print(newArray)

'''
               1
           /        \
         2            3
       /             /  \
      4              5   6
       \                /
        7              8

'''