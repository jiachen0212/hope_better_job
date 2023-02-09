# coding=utf-8
# 不同的二叉搜索树

# 卡特兰数
# 给定n个节点,求可以组成的树的颗数
# https://www.cnblogs.com/ShaneZhang/p/4102581.html
# 思路是：固定根节点，然后左右子树节点直接是n-1, n-1二分即可。
# 注意左右子树的情况是相乘：f(n-1-i)*f(i)

# f(n) = f(n-1) + f(n-2)f(1) + f(n-3)f(2) + ... + f(1)f(n-2) + f(n-1)

# dp来做
class Solution(object):
    def numTrees(self, n):
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2,n+1):
            for j in range(i):
                dp[i] += dp[j]*dp[i-j-1]
        return dp[-1]

s = Solution()
print s.numTrees(3)



# https://leetcode-cn.com/problems/unique-binary-search-trees-ii/comments/
# 进阶  把树输出来
class Solution(object):
    def generateTrees(self, n):
        def helper(tree):
            # tree 为有序数组
            ans = []
            # 遍历可能的根结点
            for i, val in enumerate(tree):
                # left、right 分别为左右子树包含的结点
                left, right = tree[:i], tree[i+1:]  # i是根节点
                # 若左子树为 NULL，则令其为 [None]
                for ltree in helper(left) or [None]:
                    # 若右子树为 NULL，则令其为 [None]
                    for rtree in helper(right) or [None]:
                        root = TreeNode(val)
                        root.left, root.right = ltree, rtree
                        ans.append(root)
            return ans
        return helper(range(1, n+1))