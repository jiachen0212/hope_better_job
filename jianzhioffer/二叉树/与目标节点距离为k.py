# coding=utf-8
# 二叉树种与target目标节点的距离为k的节点
#  leetcode
# DFS建图 BFS搜距离


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        from collections import defaultdict
        graph = defaultdict(set)
        # 建图
        def dfs(root):
            if root.left :
                graph[root.val].add(root.left.val)
                graph[root.left.val].add(root.val)
                dfs(root.left)
            if root.right:
                graph[root.val].add(root.right.val)
                graph[root.right.val].add(root.val)
                dfs(root.right)
        dfs(root)
        # print(graph)  # 节点之间的链接
        '''
        defaultdict(<type 'set'>, {0: set([1]), 1: set([0, 8, 3]), 2: set([4, 5, 7]), 3: set([1, 5]), 4: set([2]), 5: set([2, 3, 6]), 6: set([5]), 7: set([2]), 8: set([1])})
        '''
        # 以下就是bfs的遍历了
        cur = [target.val]
        visited ={target.val}
        while K:
            next_time = []
            while cur:
                tmp = cur.pop()
                for node in graph[tmp]:  # node是tmp节点的所有邻接点
                    if node not in visited:
                        visited.add(node)
                        next_time.append(node)
            K -= 1
            cur = next_time
        return cur


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
t1 = TreeNode(3)
t2 = TreeNode(5)
t3 = TreeNode(1)
t4 = TreeNode(6)
t5 = TreeNode(2)
t6 = TreeNode(0)
t7 = TreeNode(8)
t8 = TreeNode(7)
t9 = TreeNode(4)

t1.left = t2
t1.right = t3
t2.left = t4
t2.right = t5
t3.left = t6
t3.right = t7
t5.left = t8
t5.right = t9

s = Solution()
res = s.distanceK(t1, t2, 2)
print(res)