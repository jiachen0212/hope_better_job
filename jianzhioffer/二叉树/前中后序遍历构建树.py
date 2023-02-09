#coding=utf-8


class BTree:
    def __init__(self, c='/0', l=None, r=None):
        '''
        Initializes the node's data
        '''
        self.root = c
        self.left = l
        self.right = r

# 前序遍历
def preorderTraverse(bt):
    if bt:
        return '%s%s%s' % (bt.root, preorderTraverse(bt.left), preorderTraverse(bt.right))
    return ''

# 中序遍历
def inorderTraverse(bt):
    if bt:
        return '%s%s%s' % (inorderTraverse(bt.left), bt.root, inorderTraverse(bt.right))
    return ''

# 后序遍历
def postorderTraverse(bt):
    if bt:
        return '%s%s%s' % (postorderTraverse(bt.left), postorderTraverse(bt.right), bt.root)
    return ''


def printBTree(bt, depth):
    ch = bt.root if bt else '*'
    if (depth > 0):
        # print '%s%s%s' % ((depth - 1) * '  ', '--', ch)
        print ch
    else:
        print ch
    if not bt:
        return
    printBTree(bt.left, depth + 1)
    printBTree(bt.right, depth + 1)


# 根据前序和中序遍历结果重构这棵二叉树
def buildBTreeFromPreIn(preo, ino):
    if (preo == '' or ino == ''):
        return None
    pos = ino.find(preo[0])  # 在中序遍历中，找到根节点，从而可左右划分出左右子树
    if (pos < 0):
        return None
    # 前序: 父左右  中序: 左父右
    return BTree(preo[0], buildBTreeFromPreIn(preo[1:pos + 1], ino[0:pos]), # 迭代的找，前序的根节点，在中序中的位置,迭代重构出树
                 buildBTreeFromPreIn(preo[pos + 1:], ino[pos + 1:]))
                # (preo[1:pos + 1], ino[0:pos]) 左树
                # (preo[pos + 1:], ino[pos + 1:])) 右树


# 根据中序和后序遍历结果重构这棵二叉树
def buildBTreeFromInPost(ino, po):
    if (ino == '' or po == ''):
        return None
    pos = ino.find(po[-1])
    if (pos < 0):
        return None
    return BTree(po[-1], buildBTreeFromInPost(ino[0:pos], po[0:pos]), buildBTreeFromInPost(ino[pos + 1:], po[pos:-1]))



# * 表示没有左/右子节点。 **表示到根节点了已经。
if __name__ == '__main__':
    preo = 'aefhbkilgdjc'
    ino = 'debac'
    po = 'dabec'
    # bt = buildBTreeFromPreIn(preo, ino)
    # print 'Build from preorder & inorder'
    # print 'Preorder: %s' % (preorderTraverse(bt))
    # print 'Inorder: %s' % (inorderTraverse(bt))
    # print 'Postorder: %s' % (postorderTraverse(bt))
    # print 'The BTree is (* means no such a node):'
    # printBTree(bt, 0)
    print('=====')
    bt = buildBTreeFromInPost(ino, po)
    print 'Build from inorder & postorder'
    print 'Preorder: %s' % (preorderTraverse(bt))
    print 'Inorder: %s' % (inorderTraverse(bt))
    print 'Postorder: %s' % (postorderTraverse(bt))
    print 'The BTree is (* means no such a node):'
    printBTree(bt, 0)







# 牛客ac
# 前中序遍历得到二叉树
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))   # pre.pop(0) 把此刻的根节点删掉了， 所以下面可以直接
        # 使用pre 而无需做别的index设置
        index = tin.index(root.val)   # 返回根节点在中序遍历中的位置，从而区分左右树
        root.left = self.reConstructBinaryTree(pre, tin[:index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root
