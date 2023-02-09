#coding=utf-8
# 序列化和反序列化二叉树

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        tree, ch = [root], []  # ch为二叉树对应的序列
        while len(tree) > 0:
            temp = tree.pop(0)
            if temp is None:
                ch.append('$,')  # 空指针
            else:
                ch.append(str(temp.val) + ',')  # 节点间使用#间隔
                # 依次遍历左右子树
                tree.append(temp.left)
                tree.append(temp.right)
        return ''.join(ch)

    # 反序列化
    def deserialize(self, s):
        s1, i = s.split(','), 0  # 这里其实用了切片迭代.. i在迭代..
        if s1[i] == '$':
            return None  # 第一个节点就是空的,表明是空树
        root = TreeNode(int(s1[i]))
        tree = [root]  # 先是只有根节点的树
        while len(tree) > 0:
            te = tree.pop(0)
            i += 1
            if s1[i] != '$':  # 没遇到空指针就继续往下迭代
                k = TreeNode(int(s1[i]))
                te.left = k  # 左子树加进来
                tree.append(k)
            i += 1
            if s1[i] != '$':
                k = TreeNode(int(s1[i]))
                te.right = k  # 右子树加进来
                tree.append(k)
        return root
