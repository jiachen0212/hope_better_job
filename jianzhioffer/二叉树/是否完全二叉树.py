# coding=utf-8
# 是否完全二叉树
# 给根节点，判断是否是完全二叉树
'''
1. 左无右有，False
2. 到了叶节点状态了但后续还有左右孩子  False
3. 左有右有，加入队列接着往下看
'''

# 也许是  DFS打天下吧
class Solution(object):
    def isCompleteTree(self, head):
        if not head:
            return True
        isLeaf = False   # 进行是否叶节点判断的flag
        queue = []
        queue.append(head)
        while queue:
            head = queue.pop(0)
            left = head.left
            right = head.right
            if (not left and right) or (isLeaf and (left or right)):
                # （not left） and  right 右边存在 左边不存在
                #  或者是进入到全是叶节点状态后但你有左或者右
                # 这两种情况都会返回F
                return False
            if left:
                queue.append(left)
            if right:
                queue.append(right)
            if not left or not right: # 注意这一步,只要左右有一边没有,
            # 证明即将进入叶节点状态
                isLeaf = True
        return True

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
t1 = TreeNode(1)
t2 = TreeNode(2)
t3 = TreeNode(3)
t4 = TreeNode(4)
t5 = TreeNode(5)
t6 = TreeNode(6)
t1.left = t2
t1.right = t3
t2.left = t4
t3.right = t5
t3.left = t6


s = Solution()
ans = s.isCBT(t1)
print(ans)