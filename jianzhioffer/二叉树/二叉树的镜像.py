#coding=utf-8
# 得到一颗二叉树的对称树

# 定义树
class treeNode:
    def __init__(self, x):
        self.left = None
        self.right = None
        self.val = x

class Solution:
    # 一直从上到下迭代的把左右子树互换就好..
    def turnToMirror(self, root):
        if root == None:
            return
        root.right, root.left = root.left, root.right
        self.turnToMirror(root.left)
        self.turnToMirror(root.right)
        return root

    # 给定二叉树的前序遍历和中序遍历,获得该二叉树
    def getBSTwithPreTin(self, pre, tin):
        if len(pre)==0 | len(tin)==0:
            return None
        root = treeNode(pre[0])
        for order, item in enumerate(tin):
            if root.val == item:  # order是当前的父节点   前序: 父左右  中序: 左父右
                root.left = self.getBSTwithPreTin(pre[1:order+1], tin[:order])  # 迭代的生成左树
                root.right = self.getBSTwithPreTin(pre[order+1:], tin[order+1:])  # 迭代的生成右树
                return root

if __name__ == '__main__':
    flag = "turnToMirror"
    solution = Solution()
    preorder_seq = [1, 2, 4, 7, 3, 5, 6, 8]
    middleorder_seq = [4, 7, 2, 1, 5, 3, 8, 6]
    treeRoot1 = solution.getBSTwithPreTin(preorder_seq, middleorder_seq)
    if flag == "turnToMirror":
        solution.turnToMirror(treeRoot1)
        print treeRoot1