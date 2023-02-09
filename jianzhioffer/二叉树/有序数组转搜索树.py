# coding=utf-8
# 有序数组转二叉搜索树

# 找到中点值作为根节点 然后左右分别递归


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def sortedArrayToBST(self, nums):
        if not nums:
            return None
        l, r = 0, len(nums)-1
        mid = (l+r)/2
        root = TreeNode(nums[mid])
        nums1=nums[0:mid]
        nums2=nums[mid+1:]
        root.left=self.sortedArrayToBST(nums1)
        root.right=self.sortedArrayToBST(nums2)
        return root