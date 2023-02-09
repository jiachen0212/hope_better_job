#coding=utf-8
# 二叉搜索树：左子树所有节点都小于父，右子树都大于父
# 后序遍历：左右父

##########update 20190704 牛客ac
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        size = len(sequence)
        if size == 0:
            return False
        size -= 1
        index = 0
        while size:
            # sequence[size] size=len-1即最末那个元素，也即根节点
            while index < size and sequence[index] < sequence[size]:
                index += 1
            while index < size and sequence[index] > sequence[size]:
                index += 1
            if index < size: # 上两行遍历完了还没到最末？那肯定有不符合大小的节点，return false
                return False
            index = 0
            size -= 1   # 根节点位置修正
        return True



s = Solution()
res = s.VerifySquenceOfBST([1,8,5,15,10])
print(res)

