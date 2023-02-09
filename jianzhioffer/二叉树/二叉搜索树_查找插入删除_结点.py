#coding=utf-8

import queue

class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
    def settag(self,tag=None):
        self.tag = tag

    # 插入结点
    def InsertNode(root,treenode):
        tmp = root
        if(root is None):
            root = treenode
        while(tmp is not None):
            if(treenode.val == tmp.val):
                break
            elif(treenode.val < tmp.val):
                if(tmp.left is None):
                    tmp.left = treenode
                else: tmp = tmp.left
            else:
                if(tmp.right is None):
                    tmp.right = treenode
                else: tmp = tmp.right
        return root

    # 查找结点
    def search(root,key):
        tmp = root
        while(tmp is not None and key != tmp.val):
            if(key < tmp.val):
                tmp = search(tmp.left, key)
            else:
                tmp = search(tmp.right, key)
        return tmp

    # 获取结点的父亲结点
    def getparent(root,treenode):
        tmp = root
        if(root == treenode): return None
        while(tmp is not None):
            if(tmp.right == treenode or tmp.left == treenode):
                return tmp
            elif(treenode.val < tmp.val):
                tmp = tmp.left
            else:
                tmp = tmp.right
        return None

    # 合并删除结点
    def mergedeletenode(root,treenode):
        tmp = treenode
        parentnode = getparent(root, tmp)
        if(parentnode is not None):
            if(treenode.right is None and treenode.left is None):
                if(parentnode.right == treenode): parentnode.right = None
                else: parentnode.left = None
            elif(treenode.right is not None and treenode.left is None):
                if(parentnode.right == treenode): parentnode.right = tmp.right
                else: parentnode.left = tmp.right
            elif(treenode.right is None and treenode.left is not None):
                if(parentnode.right == treenode): parentnode.right = tmp.left
                else: parentnode.left = tmp.left
            else:
                tmp = treenode.left
                while(tmp.right!=None):
                    tmp = tmp.right
                tmp.right = treenode.right
                if(parentnode.right == treenode): parentnode.right = treenode.left
                else: parentnode.left = treenode.left
        else:#删除根节点
            if(treenode.right is None and treenode.left is None):
                root = None
            elif(treenode.right is not None and treenode.left is None):
                root = tmp.right
            elif(treenode.right is None and treenode.left is not None):
                root = tmp.left
            else:
                tmp = tmp.left
                while(tmp.right is not None):
                    tmp = tmp.right
                tmp.right = treenode.right
                root = treenode.left
        return root




# test
if __name__=='__main__':
    nodes = [5,1,7,4,6,9,8]
    root = None
    for i in range(len(nodes)):
        node = TreeNode(nodes[i])
        root = InsertNode(root, node)
    PreOrderWithoutRecursion(root)
    print()
    InOrderWithoutRecursion(root)
    print()
    tmp = search(root, 5)
    print(tmp.val)
    root = mergedeletenode(root, tmp)    # 删除结点 5
    PreOrderWithoutRecursion(root)
    print()
    InOrderWithoutRecursion(root)