# coding=utf-8
# leetcode ac
'''
中序遍历后，然后前后往前一直add
'''

# 代码很精简！   右父左 一直add
class Solution(object):
    def bstToGst(self, root):
        self.ans = [0]
        temp = root
        self._midOrder(temp)
        return root
    def _midOrder(self, node):
        if not node:
            return
        if node.right:
            self._midOrder(node.right)
        node.val += self.ans[-1]
        self.ans[-1] = node.val
        if node.left:
            self._midOrder(node.left)