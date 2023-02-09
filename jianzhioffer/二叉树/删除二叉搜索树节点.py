# coding=utf-8
# 删除二叉搜索树的一个节点key

# 递归
class Solution(object):
    def deleteNode(self, root, key):
        if not root:
            return None;
        if root.val > key:  # 要删除的树肯定只在左孩子，因为左<父<右
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:   # 要删除的就是根节点?
            if not root.left or not root.right:
                root = root.left if root.left else root.right
            else: # 根节点有左孩子也有右孩子
                cur = root.right
                while cur.left:
                    cur = cur.left   # 一直找到根节点右孩子的最左，这个值肯定是右边最小的
                cur.left = root.left   # 把根的左子树搬过来...即使cur在右边是最小，但它也还是比所有的左都大的！！！
                return root.right
        return root





####递归2   这个最后面有点不好理解
class Solution(object):
    def deleteNode(self, root, key):
        if not root:
            return None;
        if root.val > key:  # 要删除的树肯定只在左孩子，因为左<父<右
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:   # 要删除的就是根节点?
            if not root.left or not root.right:
                root = root.left if root.left else root.right
            else: # 根节点有左孩子也有右孩子
                cur = root.right
                while cur.left:
                    cur = cur.left   # 一直找到根节点右孩子的最左，这个值肯定是右边最小的
                root.val = cur.val   # 把这个右边的最小值放上去做根节点值，
                # 可以保证其都小于右边的值
                root.right = self.deleteNode(root.right, cur.val)
        return root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 非递归
class Solution:
    def deleteNode(self, root, key):
        parent = None
        node = root
        while (node and node.val != key):
            parent = node
            if node.val > key:
                node = node.left
            else:
                node = node.right

        # parent始终为父节点  这句话好好理解！！

        # not found
        if not node:
            return root
        # 跳出上面的while循环就是找到了这个key了，放在node节点变量中。

        # node does't has child
        elif not node.left and not node.right:
            # not root
            if parent:  # 意思就是这个是找到的key的父节点
                if parent.left == node:
                    parent.left = None   # ok，找到了key，删掉你
                else:
                    parent.right = None  # key是右节点？那我也删掉你
                return root
            # root
            return None    # 如果找到的key就是根节点，那就删掉然后retune的就是none了

        # node  has two children
        elif node.left and node.right:   #用node与左子树最大的节点交换（或者用node与右子树最小的节点交换） 最后删除最小或者最大的节点
            pre_parent = node
            pre = node.left
            while pre.right:
                pre_parent = pre
                pre = pre.right
            if pre_parent != node:      #经过while循环 导致pre_parent不是node
                pre_parent.right = pre.left    #将左子树最大节点用最大节点的左孩子代替 因为是最大的所以没有右孩子
                node.val = pre.val             #将最大节点的值与要删除的节点的值做替换
            else:
                node.val = pre.val      #值交换
                node.left = pre.left    #将最大值（因为没有右子树所以就是pre）删除 删除的方法就是：用该节点的左子树代替
                # 把左子树推上去就可以了
            return root

        # node only has one child
        else:
            if parent:
                if parent.left == node:
                    parent.left = node.left or node.right
                else:
                    parent.right = node.left or node.right
                return root
            else:
                return node.left or node.right