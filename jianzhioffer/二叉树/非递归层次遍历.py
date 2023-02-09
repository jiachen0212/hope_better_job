# coding=utf-8
# 非递归层次遍历

# BFS + 双端队列实现

class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        result = []
        queue = []
        queue.append(root)

        while queue:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):   # 因为只对这一层进行遍历
                node = queue.pop(0)
                current_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(current_level)
        return result

