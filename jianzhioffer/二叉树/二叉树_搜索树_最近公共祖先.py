# coding=utf-8
# 二叉树两节点的最低公共祖先

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        dic={root:None}

        # 中间的help函数
        def bfs(node):
            if node:
                if node.left:
                    dic[node.left]=node
                if node.right:
                    dic[node.right]=node
                bfs(node.left)
                bfs(node.right)

        bfs(root)   # 用字典存储每个节点的最近父节点

        l1,l2=p,q
        while(l1!=l2):  # dic的get函数 取得到就取 取不到就会返回False
            l1=dic.get(l1) if l1 else q
            l2=dic.get(l2) if l2 else p
        return l1


# 递归版本
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if root in (None, p, q): return root
        left, right = (self.lowestCommonAncestor(kid, p, q)
                       for kid in (root.left, root.right))
        return root if left and right else left or right





# 二叉搜索树的最低公共祖先
# 可利用搜索树的特点  左<父<右
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        # p q 都比根小  那就都在左子树
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root  # p q 在一左一右的话  那最近的公共祖先直接就是根节点了
