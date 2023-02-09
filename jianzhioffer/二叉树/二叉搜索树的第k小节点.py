#coding=utf-8
# 二叉搜索树：左节点--根节点--右节点  依次增大   左父右 正好也是中序遍历
# 给定一棵二叉搜索树，请找出其中的第k小的结点。

# 递归实现
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def kthSmallest(self, root, k):
        if not k:
            return
        nodes = []
        self.inOrder(root, nodes)
        if len(nodes)<k:
            return None
        return nodes[k-1]
    # 左父右，中序遍历，且遍历后就是有序的，因为是搜索二叉树，左<父<右
    def inOrder(self, root, nodes):
        if not root:
            return
        if root.left:
            self.inOrder(root.left, nodes)
        nodes.append(root.val)  # 父节点value加入
        if root.right:
            self.inOrder(root.right, nodes)