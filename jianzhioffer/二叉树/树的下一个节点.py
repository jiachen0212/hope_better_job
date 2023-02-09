#coding=utf-8
####  牛客 ac 版
# 中序遍历下的 下一个节点   左根右
class Solution:
    def GetNext(self, pNode):
        if pNode is None:
            return None
        # 存在右子树，那么下一个节点就是右子树的最左子节点    (左中右撒～)
        if pNode.right:  # 存在右子树
            temp = pNode.right
            while temp.left:  # 一直沿着左指针,指到最左边那个节点
                temp = temp.left
            return temp
        elif pNode.next is None:
            return None
        elif pNode.next.left == pNode:  # 没有右子树但是是左节点,那么下一个点就是父节点就直接是next
            return pNode.next
        else:
            while pNode.next:  # 没有右子树并且又是右节点,就一直返回直到该点是左节点位置
                if pNode.next.left != pNode:
                    pNode = pNode.next
                else:
                    return pNode.next
            return None
