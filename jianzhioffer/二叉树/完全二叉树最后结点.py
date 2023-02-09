# coding=utf-8
# 完全二叉树：就是左树一定是满的，右树可能是空的...

# 算法思想：首先遍历完全二叉树的左分支，求出完全二叉树的高度depth, 
# 然后对于每个子树的根节点，先从根节点的右孩子开始，然后从此节点遍历该节点的左孩子，
# 等遍历完成后，进行判断此时临时高度等于二叉树的高度，且节点无右孩子时候，
# 则输出该节点，否则右侧还有节点，则遍历右子树，
# 若临时高度小于二叉树的高度，则遍历根节点的左孩子。

# https://blog.csdn.net/wodedipang_/article/details/96427407


class TreeNode(object):

    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def printlastnode(self, root):
        if not root:
            return None
        depth = 0
        tmp = root
        while tmp:        # 计算二叉树的高度
            depth += 1
            tmp = tmp.left

        level = 0
        tempdepth = 0
        while root:
            level += 1
            if level == depth:
                break
            curnode = root
            if curnode.right:    # 先遍历右孩子，若无右孩子，则line59
                parent = curnode
                curnode = curnode.right
                tempdepth = level + 1   # 设置临时高度
                while curnode.left:    # 从根节点右树起一直循环往其左孩子遍历
                    tempdepth += 1
                    parent = curnode
                    curnode = curnode.left
                # 若临时统计高度小于二叉树高度，证明右边的树更短，则root=root.left
                # 得去根节点的左孩子那边找最后的节点
                if tempdepth < depth:
                    root = root.left

                # tempdepth == depth, 且无右孩子
                elif not curnode.right or parent.right == curnode:
                    return curnode
                else:  # 高度是相等的，但curnode还有右孩子
                    root = root.right   # 那就把根节点的右节点当作根节点再来一遍

            else:   # 根节点没有右孩子？好吧那就遍历左孩子
                root = root.left

        return root


if __name__ == "__main__":
    sol = Solution()
    t1 = TreeNode(1)
    t2 = TreeNode(2)
    t3 = TreeNode(3)
    t4 = TreeNode(4)
    t5 = TreeNode(5)
    t6 = TreeNode(666)
    t1.left = t2
    t1.right = t3
    t2.left = t4
    t2.right = t5
    t3.left = t6

    res = sol.printlastnode(t1)
    if res:
        print("lastest node = %s" % res.val)